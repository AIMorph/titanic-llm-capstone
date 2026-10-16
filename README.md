# Titanic Survival Predictor (LLM-Powered)

## Project Description
A chat application where a user describes a Titanic passenger in plain English and a trained machine learning model estimates survival probability. An LLM turns the text into model inputs and explains the result. It is built for learners and anyone curious about how an ML model can sit behind a natural language interface. It solves the problem of using a model without knowing its feature names or input format.

**Dataset:** Titanic passenger data (`titanic.csv` from github.com/datasciencedojo/datasets), 891 rows. Prediction task: binary classification of `Survived`.

**LLM provider:** Google Gemini API (free tier) through its OpenAI-compatible endpoint, model `gemini-3.8-flash`. The environment variables keep the names `NEBIUS_*` from the course template, but they hold Gemini values.

## Setup
1. Install Python 3.11 and Git.
2. Clone and enter the repo:
```
   git clone https://github.com/AIMorph/titanic-llm-capstone.git
   cd titanic-llm-capstone
```
3. Create the environment and install dependencies (Windows PowerShell):
```
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
```
4. API keys: create a free key at https://aistudio.google.com/api-keys, then copy `.env.example` to `.env` and fill in:
```
   NEBIUS_API_KEY=your-gemini-key
   NEBIUS_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   NEBIUS_MODEL=gemini-3.8-flash
```
5. Get the data:
```
   Invoke-WebRequest -Uri "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv" -OutFile "data\titanic.csv"
```
6. Train and select the best model:
```
   python -m src.train
   python -m src.compare_runs
```

## Usage
```
python -m src.app
```
Open http://127.0.0.1:7860 and type, for example: "I'm a 28-year-old woman in first class traveling alone."

Docker:
```
docker build -t titanic-llm-app .
docker run --rm --env-file .env -p 7860:7860 titanic-llm-app
```
Run tests: `pytest tests/ -v`

View experiments: `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open http://127.0.0.1:5000

## Architecture
1. The user's message goes to the LLM, which returns structured JSON of passenger features (or flags the query as out of scope).
2. Code validates the values and asks a clarifying question if required features are missing or invalid.
3. The best trained scikit-learn pipeline (`models/best_model.joblib`, chosen with `mlflow.search_runs()`) predicts survival probability.
4. The LLM explains the probability in plain language with caveats.

Training hyperparameters live in `configs/config.yaml`. Experiments are tracked in MLflow (`sqlite:///mlflow.db`). Preprocessing (median/most-frequent imputation, scaling, one-hot encoding) is part of the saved pipeline and is fitted on training data only, so there is no data leakage.

## Results
Six runs, evaluated on a held-out 20% test set and ranked by F1:

| Run | Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| rf_300_nodepth | random_forest | 0.8156 | 0.8103 | 0.6812 | 0.7402 | 0.8318 |
| logreg_c01 | logistic_regression | 0.8101 | 0.8070 | 0.6667 | 0.7302 | 0.8440 |
| logreg_c1 | logistic_regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8427 |
| rf_100_depth5 | random_forest | 0.8101 | 0.8723 | 0.5942 | 0.7069 | 0.8409 |
| gb_lr01 | gradient_boosting | 0.7933 | 0.7857 | 0.6377 | 0.7040 | 0.8192 |
| gb_lr005 | gradient_boosting | 0.7989 | 0.8235 | 0.6087 | 0.7000 | 0.8222 |

**Best model:** `rf_300_nodepth` (random forest, 300 trees, no depth limit). It has the highest accuracy and F1 and the best balance of precision and recall among the six. Always predicting "did not survive" would score about 61.6% accuracy, so all models beat that baseline clearly.

**Findings:** The two logistic regression runs had the highest ROC AUC (about 0.84), even though the random forest won on F1 and accuracy. The models are close, so the choice of model mattered less than the features and preprocessing.

## Reflection
The project contained many of the prior sprint components but allowed me to see how all parts come together in an MLOps.  I learned how Docker, containers, LLM, NLP create a pipeline for models to generate accurate predictions based on multiple experiments through validation (tests).  Understanding what are the correct pieces of information need to properly run a MLOps make the project difficult for any beginner to comprehend as a whole. The most difficult part of the MLOps process would have to be the preprocessing, training & testing. With time and patience however these will come more fluent as i complete future projects/models.

## Demo
https://youtu.be/m0YlkjmfesI