from types import SimpleNamespace

from src.interface import (
    CLARIFY_PREFIX,
    OUT_OF_SCOPE_MESSAGE,
    extract_json,
    handle_query,
    parse_query,
    validate_features,
)


class FakeClient:
    """Pretends to be the LLM client and always returns the same text."""

    def __init__(self, content):
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DummyModel:
    def predict_proba(self, frame):
        raise AssertionError("Model must not run on incomplete input")


def test_parse_extracts_correct_features():
    reply = (
        '```json\n{"in_scope": true, "features": {"Pclass": 1, "Sex": "female", '
        '"Age": 28, "SibSp": 0, "Parch": 0, "Fare": null, "Embarked": null}}\n```'
    )
    parsed = parse_query(FakeClient(reply), "fake-model", "I'm a 28 year old woman in first class, alone")
    features, problems = validate_features(parsed["features"])
    assert parsed["in_scope"] is True
    assert features["Pclass"] == 1
    assert features["Sex"] == "female"
    assert features["Age"] == 28
    assert problems == []


def test_incomplete_input_asks_for_missing_info():
    reply = (
        '{"in_scope": true, "features": {"Pclass": 2, "Sex": "male", "Age": null, '
        '"SibSp": null, "Parch": null, "Fare": null, "Embarked": null}}'
    )
    answer = handle_query("I'm a man in second class", FakeClient(reply), "fake-model", DummyModel())
    assert answer.startswith(CLARIFY_PREFIX)
    assert "age" in answer


def test_out_of_scope_query_is_rejected():
    reply = '{"in_scope": false, "features": {}}'
    answer = handle_query("What is the capital of France?", FakeClient(reply), "fake-model", DummyModel())
    assert answer == OUT_OF_SCOPE_MESSAGE


def test_invalid_values_are_flagged():
    features, problems = validate_features({"Pclass": 7, "Sex": "male", "Age": 250})
    assert features["Pclass"] is None
    assert features["Age"] is None
    assert len(problems) == 2
    assert extract_json("no json here") is None