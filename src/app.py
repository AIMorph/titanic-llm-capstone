"""Gradio chat app for the Titanic survival predictor."""
import joblib
import gradio as gr

from src.interface import combine_with_history, get_client, handle_query

MODEL_PATH = "models/best_model.joblib"


def build_app():
    model = joblib.load(MODEL_PATH)
    client, llm_model = get_client()

    def respond(message, history):
        text = combine_with_history(message, history)
        return handle_query(text, client, llm_model, model)

    return gr.ChatInterface(
        fn=respond,
        type="messages",
        title="Titanic Survival Predictor",
        description="Describe a passenger in plain English and the trained model estimates their survival probability.",
        examples=[
            "I'm a 28-year-old woman in first class traveling alone.",
            "I'm a 40-year-old man in third class with 2 kids and my wife.",
            "What is the capital of France?",
        ],
    )


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)