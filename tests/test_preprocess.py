import numpy as np
import pandas as pd
import pytest

from src.preprocess import build_preprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": [22.0, np.nan, 35.0, 54.0, 2.0, 27.0, 14.0, 40.0],
        "SibSp": [1, 0, 1, 0, 3, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 1, 2, 0, 0],
        "Fare": [7.25, 71.28, np.nan, 51.86, 21.07, 11.13, 30.07, 8.05],
        "Pclass": [3, 1, 1, 1, 3, 3, 2, 3],
        "Sex": ["male", "female", "female", "male", "male", "female", "female", "male"],
        "Embarked": ["S", "C", None, "S", "S", "Q", "C", "S"],
    })


def test_missing_values_are_filled(sample_df):
    out = build_preprocessor().fit_transform(sample_df)
    assert not np.isnan(out.astype(float)).any()


def test_categoricals_are_one_hot_encoded(sample_df):
    pre = build_preprocessor()
    pre.fit(sample_df)
    names = list(pre.get_feature_names_out())
    assert "cat__Sex_male" in names
    assert "cat__Sex_female" in names
    assert any(n.startswith("cat__Embarked_") for n in names)


def test_numeric_features_are_scaled(sample_df):
    out = build_preprocessor().fit_transform(sample_df).astype(float)
    numeric_block = out[:, :4]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-6)


def test_original_dataframe_not_modified(sample_df):
    before = sample_df.copy(deep=True)
    build_preprocessor().fit_transform(sample_df)
    pd.testing.assert_frame_equal(sample_df, before)