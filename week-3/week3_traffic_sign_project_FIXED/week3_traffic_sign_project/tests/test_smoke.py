import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import torch
from src.models import BaselineCNN, build_model
from src.metrics import compute_metrics
from src.data import stratified_indices

class SmokeTests(unittest.TestCase):
    def test_baseline_forward(self):
        m = BaselineCNN(43)
        y = m(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(y.shape), (2, 43))

    def test_resnet_forward(self):
        m = build_model("resnet18", 43, pretrained=False)
        y = m(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(y.shape), (2, 43))

    def test_stratified_split(self):
        targets = np.repeat(np.arange(4), 10)
        tr, va = stratified_indices(targets, 0.2, 42)
        self.assertEqual(len(set(tr) & set(va)), 0)
        self.assertEqual(len(tr) + len(va), len(targets))

    def test_metrics(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.array([0, 1, 2, 1, 1, 2])
        probs = np.eye(3)[p] * 0.9 + 0.05
        m = compute_metrics(y, p, probs, 3)
        for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro_ovr"]:
            self.assertIn(k, m)

if __name__ == "__main__":
    unittest.main()
