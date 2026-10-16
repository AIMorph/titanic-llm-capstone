"""LLM interface: parse natural language -> validate -> run model -> explain."""
import json
import os
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.preprocess import ALL_FEATURES

REQUIRED = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
FRIENDLY_NAMES = {
    "Pclass": "ticket class (1st, 2nd or 3rd)",
    "Sex": "sex (male or female)",
    "Age": "age",
    "SibSp": "number of siblings/spouses traveling with you",
    "Parch": "number of parents/children traveling with you",
}
NUMERIC_RANGES = {"Age": (0, 100), "SibSp": (0, 10), "Parch": (0, 10), "Fare": (0, 600)}

CLARIFY_PREFIX = "I need a bit more information"
OUT_OF_SCOPE_MESSAGE = (
    "I can only estimate Titanic survival probability for a passenger profile. "
    "Describe a passenger (ticket class, sex, age, family aboard) and I will run the model."
)

PARSE_PROMPT = """You extract passenger features from a user's message for a Titanic survival model.
Reply with JSON ONLY, no other text, in exactly this shape:
{"in_scope": true or false,
 "features": {"Pclass": 1|2|3|null, "Sex": "male"|"female"|null, "Age": number|null,
              "SibSp": integer|null, "Parch": integer|null, "Fare": number|null,
              "Embarked": "S"|"C"|"Q"|null}}
Rules:
- in_scope is true only if the user describes a passenger (themselves or someone else) to evaluate for Titanic survival. Otherwise false and all features null.
- Use null for anything the user did not state. NEVER guess or invent values.
- SibSp = siblings + spouses aboard. Parch = parents + children aboard.
- "traveling alone" means SibSp = 0 and Parch = 0.
- Pclass: first class = 1, second = 2, third = 3.
- Embarked: Southampton = S, Cherbourg = C, Queenstown = Q."""

EXPLAIN_PROMPT = """You explain a Titanic survival model's output to a beginner.
Write 3 to 5 plain sentences. Use ONLY the facts provided. Include: the survival probability as a percent,
what it means in context, and caveats (this is a statistical model trained on 891 historical passengers,
not a real-life prediction; any values not provided were filled with typical values).
Do not invent which factors mattered most."""


def get_client():
    load_dotenv()
    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL")
    llm_model = os.getenv("NEBIUS_MODEL")
    if not (api_key and base_url and llm_model):
        raise RuntimeError(
            "Set NEBIUS_API_KEY, NEBIUS_BASE_URL and NEBIUS_MODEL in your .env file."
        )
    return OpenAI(api_key=api_key, base_url=base_url), llm_model


def extract_json(text):
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _to_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_features(raw):
    """Return (clean_features, problems). Missing values stay None."""
    clean = {name: None for name in ALL_FEATURES}
    problems = []

    if raw.get("Pclass") is not None:
        n = _to_number(raw["Pclass"])
        if n in (1, 2, 3):
            clean["Pclass"] = int(n)
        else:
            problems.append("ticket class must be 1, 2 or 3")

    if raw.get("Sex") is not None:
        sex = str(raw["Sex"]).strip().lower()
        if sex in ("male", "female"):
            clean["Sex"] = sex
        else:
            problems.append("sex must be male or female")

    for name, (low, high) in NUMERIC_RANGES.items():
        if raw.get(name) is not None:
            n = _to_number(raw[name])
            if n is not None and low <= n <= high:
                clean[name] = n
            else:
                problems.append(f"{name} must be a number between {low} and {high}")

    if raw.get("Embarked") is not None:
        port = str(raw["Embarked"]).strip().upper()
        if port in ("S", "C", "Q"):
            clean["Embarked"] = port
        else:
            problems.append("embarkation port must be S, C or Q")

    return clean, problems


def parse_query(client, llm_model, text):
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": PARSE_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    data = extract_json(response.choices[0].message.content)
    if data is None:
        return {"in_scope": False, "features": {}}
    return {
        "in_scope": bool(data.get("in_scope", False)),
        "features": data.get("features") or {},
    }


def predict_survival(model, features):
    row = {k: (np.nan if features.get(k) is None else features[k]) for k in ALL_FEATURES}
    frame = pd.DataFrame([row])
    return float(model.predict_proba(frame)[0][1])


def explain_prediction(client, llm_model, features, probability):
    provided = {k: v for k, v in features.items() if v is not None}
    not_provided = [k for k, v in features.items() if v is None]
    facts = {
        "survival_probability_percent": round(probability * 100, 1),
        "values_provided": provided,
        "values_not_provided": not_provided,
    }
    try:
        response = client.chat.completions.create(
            model=llm_model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": EXPLAIN_PROMPT},
                {"role": "user", "content": json.dumps(facts)},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return (
            f"The model estimates a {probability * 100:.1f}% chance of survival for this "
            "passenger profile. This is a statistical estimate from 891 historical "
            "passengers, not a real-life prediction."
        )


def handle_query(text, client, llm_model, model):
    try:
        parsed = parse_query(client, llm_model, text)
    except Exception:
        return "Sorry, I could not reach the language model. Check your API settings and try again."

    if not parsed["in_scope"]:
        return OUT_OF_SCOPE_MESSAGE

    features, problems = validate_features(parsed["features"])
    missing = [FRIENDLY_NAMES[name] for name in REQUIRED if features[name] is None]

    if problems or missing:
        message = f"{CLARIFY_PREFIX} to run the model."
        if problems:
            message += " Please check: " + "; ".join(problems) + "."
        if missing:
            message += " Please tell me your: " + ", ".join(missing) + "."
        return message

    probability = predict_survival(model, features)
    return explain_prediction(client, llm_model, features, probability)


def combine_with_history(message, history):
    """If the last bot reply was a clarifying question, merge the earlier user messages."""
    parts = [message]
    for item in reversed(history):
        if item["role"] == "assistant":
            if str(item["content"]).startswith(CLARIFY_PREFIX):
                continue
            break
        parts.append(str(item["content"]))
    return "\n".join(reversed(parts))