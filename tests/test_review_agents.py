import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from trout_review_agents import ReviewWorkflow, write_reports, main, sha
from train_trout_vlm import MODEL, REVISION


POLICY = {"quality_margin": 0.05, "age_margin": 0.05, "quality_pixels": 200704,
          "age_pixels": 802816, "review_pixels": 1605632, "max_calls": 6}


class FakeTool:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def inspect(self, path, task, pixels, rotation, contrast):
        self.calls.append((path, task, pixels, rotation, contrast))
        pred, margin = next(self.responses)
        return {"prediction": pred, "margin": margin, "ranking_weights": [], "calibrated_probability": False}


class ReviewTests(unittest.TestCase):
    def test_cli_dry_run_and_test_requires_frozen_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "scale.png"
            Image.new("RGB", (32, 32), "white").save(image)
            labels = root / "labels.csv"
            pd.DataFrame({"scale_id": ["s1"], "fish_key": ["f1"], "path": [str(image)],
                          "split": ["validation"], "quality_gt": ["readable"], "age4": [2],
                          "age_label_source": ["expert_direct"]}).to_csv(labels, index=False)
            adapters = {}
            for task, pixels in [("quality", 200704), ("age", 802816)]:
                run = root / task
                adapter = run / "best_adapter"
                adapter.mkdir(parents=True)
                (adapter / "adapter_model.safetensors").write_bytes(b"fake checkpoint")
                (run / "training_config.json").write_text(json.dumps({"task": task, "model": MODEL,
                    "revision": REVISION, "labels_sha256": sha(labels), "max_pixels": pixels}))
                adapters[task] = str(adapter)
            argv = ["trout_review_agents.py", "--labels", str(labels), "--quality-adapter", adapters["quality"],
                    "--age-adapter", adapters["age"], "--out", str(root / "out"), "--dry-run"]
            with patch("sys.argv", argv):
                main()
            self.assertFalse((root / "out").exists())
            with patch("sys.argv", argv + ["--split", "test"]):
                with self.assertRaisesRegex(ValueError, "Freeze validation"):
                    main()

    def run_workflow(self, responses, policy=None):
        tool = FakeTool(responses)
        result = ReviewWorkflow(tool, policy or POLICY).run("anonymous.png")
        self.assertTrue(all(call[0] == "anonymous.png" for call in tool.calls))
        return result, tool

    def test_clear_bad_stops_before_age(self):
        result, tool = self.run_workflow([(1, .5)])
        self.assertEqual(result["prediction"], 4)
        self.assertEqual(len(tool.calls), 1)

    def test_consistent_age_accepts(self):
        result, tool = self.run_workflow([(0, .5), (2, .4), (2, .3)])
        self.assertEqual(result["prediction"], 2)
        self.assertEqual(result["calls"], 3)
        self.assertEqual(tool.calls[2][3], 90)

    def test_quality_review_disagreement_refers_without_age(self):
        result, tool = self.run_workflow([(0, .01), (1, .5)])
        self.assertEqual(result["status"], "expert_review")
        self.assertTrue(all(call[1] == "quality" for call in tool.calls))

    def test_quality_review_same_class_can_pass(self):
        result, tool = self.run_workflow([(0, .01), (0, .5), (1, .5), (1, .5)])
        self.assertEqual(result["prediction"], 1)
        self.assertEqual(tool.calls[1][2], POLICY["review_pixels"])

    def test_age_disagreement_not_resolved_by_majority(self):
        result, tool = self.run_workflow([(0, .5), (2, .5), (3, .5), (3, .5), (3, .5)])
        self.assertEqual(result["prediction"], -1)
        self.assertEqual(result["calls"], 5)
        self.assertEqual(tool.calls[-1][4], 1.15)

    def test_budget_and_low_margins_refer(self):
        policy = {**POLICY, "max_calls": 3}
        result, _ = self.run_workflow([(0, .5), (2, .01), (2, .01)], policy)
        self.assertEqual(result["prediction"], -1)
        self.assertEqual(result["calls"], 3)

    def test_invalid_tool_output_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            self.run_workflow([(6, .5)])

    def test_referral_queue_hides_gt_and_metrics_count_abstention(self):
        result, _ = self.run_workflow([(0, .5), (2, .5), (3, .5), (3, .5), (3, .5)])
        record = {**result, "scale_id": "s1", "fish_key": "f1", "path": "anonymous.png", "gt": 2, "seconds": 1}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_reports([record], out)
            metrics = json.loads((out / "metrics.json").read_text())
            self.assertEqual(metrics["accuracy_abstention_as_wrong"], 0)
            self.assertEqual(metrics["expert_referral_rate"], 1)
            self.assertNotIn("gt", (out / "expert_review.csv").read_text().splitlines()[0].split(","))


if __name__ == "__main__":
    unittest.main()
