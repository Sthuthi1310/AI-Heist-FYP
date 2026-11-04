"""Unit tests for preprocessing and data preparation."""
import unittest
import pandas as pd
import numpy as np

import config
from utils.preprocessing import (
    load_raw_data,
    clean_and_prepare,
    build_column_preprocessor,
    prepare_dataset_splits,
    get_sample_employee
)

class TestPreprocessing(unittest.TestCase):
    def test_load_raw_data(self):
        df = load_raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 1000)
        self.assertIn("Attrition", df.columns)

    def test_clean_and_prepare(self):
        sample = get_sample_employee("high")
        sample["Attrition"] = "Yes"
        df = pd.DataFrame([sample])
        X, y = clean_and_prepare(df)
        
        self.assertNotIn("Attrition", X.columns)
        self.assertEqual(len(X.columns), len(config.ALL_FEATURE_COLUMNS))
        self.assertEqual(int(y.iloc[0]), 1)

    def test_preprocessor_pipeline(self):
        preprocessor = build_column_preprocessor()
        sample = get_sample_employee("low")
        df = pd.DataFrame([sample])
        X_trans = preprocessor.fit_transform(df)
        self.assertIsInstance(X_trans, np.ndarray)
        self.assertGreater(X_trans.shape[1], len(config.NUMERICAL_FEATURES))

    def test_prepare_splits(self):
        splits = prepare_dataset_splits(save_splits=False)
        self.assertIn("X_victim_train", splits)
        self.assertIn("X_test", splits)
        self.assertIn("X_attack_pool", splits)
        
        # Attacker pool must not have target column
        self.assertNotIn(config.TARGET_COLUMN, splits["X_attack_pool"].columns)

if __name__ == "__main__":
    unittest.main()
