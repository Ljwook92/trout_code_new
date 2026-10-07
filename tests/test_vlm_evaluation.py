import tempfile
from pathlib import Path
import unittest

import pandas as pd

from evaluate_trout_vlm import decision, evaluation_rows, summarize


class VLMEvaluationTests(unittest.TestCase):
    def test_predictions_and_invalid_json(self):
        self.assertEqual(decision('{"decision":"bad"}', "quality"), 1)
        self.assertEqual(decision('{"prediction":3}', "age"), 3)
        for raw in ['{"prediction":true}', '{"prediction":4}', '{"prediction":"2"}']:
            with self.assertRaises(ValueError):
                decision(raw, "age")

    def test_direct_gt_only_and_fish_leakage(self):
        table = pd.DataFrame({"scale_id": ["a", "b"], "fish_key": ["f1", "f2"],
                              "path": ["a.png", "b.png"], "split": ["train", "validation"],
                              "quality_gt": ["readable"] * 2, "age4": [1, 2],
                              "age_label_source": ["expert_direct"] * 2})
        self.assertEqual(evaluation_rows(table, "validation").scale_id.tolist(), ["b"])
        table.loc[1, "age_label_source"] = "fish_propagated"
        with self.assertRaises(ValueError):
            evaluation_rows(table, "validation")
        table.loc[1, "fish_key"] = "f1"
        with self.assertRaises(ValueError):
            evaluation_rows(table, "validation")

    def test_rejected_readable_is_not_removed_from_pipeline(self):
        records = [{"model": "lora", "quality_gt": "readable", "quality_prediction": 1,
                    "age_gt": 2, "age_prediction": 2, "pipeline_gt": 2, "pipeline_prediction": 4},
                   {"model": "lora", "quality_gt": "bad", "quality_prediction": -1,
                    "age_gt": -1, "age_prediction": -1, "pipeline_gt": 4, "pipeline_prediction": -1}]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            summarize(records, out)
            result = pd.read_csv(out / "comparison.csv").set_index("task")
            self.assertEqual(result.loc["pipeline", "accuracy"], 0)
            self.assertEqual(result.loc["age_gt_readable", "accuracy"], 1)
            cm = pd.read_csv(out / "lora_pipeline_confusion.csv", index_col=0)
            self.assertEqual(cm.to_numpy().sum(), 2)
            self.assertEqual(cm["abstain"].sum(), 1)


if __name__ == "__main__":
    unittest.main()
