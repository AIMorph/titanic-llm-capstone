"""Data loading and preprocessing for the Titanic survival model."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Survived"


def load_data(path):
    return pd.read_csv(path)


def split_xy(df):
    """Return copies of the feature columns and the target (original df is untouched)."""
    return df[ALL_FEATURES].copy(), df[TARGET].copy()


def make_split(df, test_size, random_state):
    """Train/test split BEFORE any fitting, so no data leakage."""
    X, y = split_xy(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_preprocessor():
    """Numeric: median-fill + standard scaling. Categorical: most-frequent-fill + one-hot.

    Fitted only on training data inside the model Pipeline.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])