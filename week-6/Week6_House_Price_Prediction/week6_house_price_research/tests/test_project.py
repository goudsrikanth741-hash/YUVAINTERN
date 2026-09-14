
import unittest, numpy as np, pandas as pd
from src.data_loader import load_dataset, FEATURE_NAMES, TARGET_NAME
from src.feature_engineering import engineer_features
from src.pipeline import load_or_train, predict

class TestProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df,_=load_dataset(verbose=False)
        cls.bundle=load_or_train()

    def test_schema_and_no_nan(self):
        self.assertTrue(all(c in self.df.columns for c in FEATURE_NAMES+[TARGET_NAME]))
        self.assertEqual(int(self.df[FEATURE_NAMES+[TARGET_NAME]].isna().sum().sum()),0)

    def test_feature_engineering(self):
        x=engineer_features(self.df[FEATURE_NAMES].head(10))
        self.assertGreaterEqual(x.shape[1],20)
        self.assertFalse(x.isna().any().any())

    def test_models_loaded(self):
        self.assertGreaterEqual(len(self.bundle["models"]),3)
        self.assertIn(self.bundle["best_model"],self.bundle["models"])

    def test_prediction_smoke(self):
        row=self.df[FEATURE_NAMES].iloc[[0]]
        pred=predict(self.bundle,row)
        self.assertEqual(len(pred),1)
        self.assertTrue(np.isfinite(pred[0]))

if __name__=="__main__": unittest.main()
