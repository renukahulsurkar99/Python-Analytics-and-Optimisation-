"""Synthetic boundary tests; no patient data are required."""

import unittest

from project_config import choose_threshold


class ThresholdTests(unittest.TestCase):
    def test_best_precision_satisfies_recall(self):
        threshold, precision, recall = choose_threshold([0, 0, 1, 1], [0.01, 0.2, 0.8, 0.9], 1.0)
        self.assertGreater(threshold, 0.2)
        self.assertLessEqual(threshold, 0.8)
        self.assertEqual((precision, recall), (1.0, 1.0))

    def test_all_low_scores_have_a_feasible_fallback(self):
        self.assertEqual(choose_threshold([0, 1], [0.001, 0.002], 1.0), (0.0, 0.5, 1.0))

    def test_all_zero_scores_do_not_return_none(self):
        self.assertEqual(choose_threshold([0, 0, 1], [0.0, 0.0, 0.0], 0.75), (0.0, 1 / 3, 1.0))

    def test_no_positive_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive examples"):
            choose_threshold([0, 0], [0.1, 0.5], 0.8)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Probabilities"):
            choose_threshold([0, 1], [0.1, float("nan")], 0.8)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "equally sized"):
            choose_threshold([0, 1], [0.5], 0.8)

    def test_invalid_recall_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Required recall"):
            choose_threshold([0, 1], [0.1, 0.5], 1.1)


if __name__ == "__main__":
    unittest.main()
