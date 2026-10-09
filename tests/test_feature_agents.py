import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from trout_feature_agents import run_workflow, metrics, apply_feedback, export_review
from train_trout_feature_qwen import sha
from prepare_trout_training_labels import prepare_labels


POLICY = {"quality_margin": 0.05, "age_margin": 0.05, "max_calls": 8,
          "contrast": 1.2, "highres_size": 0}


class Tools:
    def __init__(self, quality, age=None):
        self.script = {"quality": iter(quality), "age": iter(age or [])}
        self.calls = []

    def inspect(self, image, task, view, policy):
        self.calls.append((task, view))
        value = next(self.script[task])
        if isinstance(value, tuple):
            pred, margin = value
        else:
            pred, margin = value, 0.2
        return {"prediction": pred, "ranking_prediction": pred, "margin": margin,
                "error": None if pred >= 0 else "Invalid JSON", "raw": "test response"}


class FeatureAgentTests(unittest.TestCase):
    def test_consistent_readable_and_age_use_four_calls(self):
        tools = Tools([0, 0], [2, 2])
        result = run_workflow(tools, Image.new("RGB", (32, 32)), POLICY)
        self.assertEqual(result["prediction"], 2)
        self.assertEqual(result["status"], "age_accepted")
        self.assertEqual(len(result["traces"]), 4)
        self.assertNotIn("gt", result)

    def test_bad_stops_before_age(self):
        tools = Tools([1, 1])
        result = run_workflow(tools, None, POLICY)
        self.assertEqual(result["prediction"], 4)
        self.assertEqual(tools.calls, [("quality", "original"), ("quality", "rotation")])

    def test_strong_quality_disagreement_refers_without_age(self):
        tools = Tools([0, 1, 1])
        result = run_workflow(tools, None, POLICY)
        self.assertEqual(result["prediction"], -1)
        self.assertEqual(result["reason"], "quality_strong_view_disagreement")
        self.assertTrue(all(task == "quality" for task, _ in tools.calls))

    def test_low_margin_can_request_contrast_and_resolve(self):
        tools = Tools([(0, 0.01), 0, 0], [3, 3])
        result = run_workflow(tools, None, POLICY)
        self.assertEqual(result["prediction"], 3)
        self.assertIn(("quality", "contrast"), tools.calls)

    def test_unresolved_age_requests_optional_highres_then_refers(self):
        tools = Tools([0, 0], [1, 2, 2, 2])
        result = run_workflow(tools, None, {**POLICY, "highres_size": 448})
        self.assertEqual(result["status"], "expert_review")
        self.assertEqual(result["quality_prediction"], 0)
        self.assertEqual(len(result["traces"]), 6)
        self.assertIn(("age", "full_frame_highres"), tools.calls)

    def test_budget_exhaustion_cannot_accept_age_without_two_views(self):
        tools = Tools([0, 0], [1])
        result = run_workflow(tools, None, {**POLICY, "max_calls": 3})
        self.assertEqual(len(tools.calls), 3)
        self.assertEqual(result["reason"], "age_inspection_budget_exhausted")
        self.assertEqual(result["prediction"], -1)

    def test_schema_failures_do_not_become_bad(self):
        result = run_workflow(Tools([-1, -1, -1]), None, POLICY)
        self.assertEqual(result["quality_prediction"], -1)
        self.assertEqual(result["status"], "expert_review")

    def test_coverage_does_not_count_false_bad_as_accepted_age(self):
        records = [{"gt": 0, "prediction": 4, "quality_prediction": 1, "status": "bad",
                    "reason": "agreed", "traces": [{}, {}], "seconds": 1},
                   {"gt": 4, "prediction": 4, "quality_prediction": 1, "status": "bad",
                    "reason": "agreed", "traces": [{}, {}], "seconds": 1}]
        summary = metrics(records)
        self.assertEqual(summary["readable_age_coverage"], 0)
        self.assertEqual(summary["bad_pass_rate"], 0)
        self.assertEqual(summary["accuracy_abstention_as_wrong"], 0.5)

    def test_feedback_train_only_preserves_old_table_and_quality_is_not_propagated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.new("RGB", (32, 32)).save(path)
            master = pd.DataFrame({"scale_id": ["a", "b", "c", "d"], "fish_key": ["f1", "f1", "f2", "f3"],
                                  "path": [str(path)] * 4, "label": [1, 1, 2, 6]})
            manifest = master[["scale_id", "fish_key"]].copy()
            manifest["split"] = ["train", "train", "validation", "test"]
            table = prepare_labels(master, manifest)
            original = table.copy(deep=True)
            annotation = pd.DataFrame([{"scale_id": "a", "fish_key": "f1", "path": str(path),
                "image_sha256": sha(path), "quality_gt": "bad", "expert_age": None,
                "reviewer": "expert", "reviewed_at": "2026-10-09"}])
            result = apply_feedback(table, annotation)
            self.assertTrue(table.equals(original))
            self.assertEqual(result.loc[0, "quality_gt"], "bad")
            self.assertEqual(result.loc[1, "quality_gt"], "readable")
            self.assertEqual(result.loc[0, "pre_review_quality_gt"], "readable")
            pd.testing.assert_frame_equal(result[["scale_id", "fish_key", "split"]],
                                          table[["scale_id", "fish_key", "split"]])
            annotation.loc[0, ["scale_id", "fish_key"]] = ["c", "f2"]
            with self.assertRaisesRegex(ValueError, "Validation/test"):
                apply_feedback(table, annotation)

    def test_feedback_requires_human_identity_and_matching_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.new("RGB", (16, 16)).save(path)
            master = pd.DataFrame({"scale_id": ["a"], "fish_key": ["f"], "path": [str(path)], "label": [1]})
            manifest = pd.DataFrame({"scale_id": ["a"], "fish_key": ["f"], "split": ["train"]})
            table = prepare_labels(master, manifest)
            annotation = pd.DataFrame([{"scale_id": "a", "fish_key": "f", "path": str(path),
                "image_sha256": sha(path), "quality_gt": "readable", "expert_age": 2,
                "reviewer": None, "reviewed_at": "2026-10-09"}])
            with self.assertRaisesRegex(ValueError, "reviewer"):
                apply_feedback(table, annotation)
            annotation.loc[0, "reviewer"] = "expert"
            annotation.loc[0, "image_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "identity"):
                apply_feedback(table, annotation)
            annotation.loc[0, "image_sha256"] = sha(path)
            result = apply_feedback(table, annotation)
            self.assertEqual(result.loc[0, "age4"], 2)
            self.assertTrue(result.loc[0, "age_train_eligible"])

    def test_review_export_hides_existing_gt_and_emits_blank_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            path = out / "image.png"
            Image.new("RGB", (32, 32)).save(path)
            export_review([{"scale_id": "a", "fish_key": "f", "path": str(path), "image_sha256": sha(path),
                "status": "expert_review", "gt": 3, "reason": "disagreement", "quality_prediction": 0,
                "age_prediction": -1, "traces": []}], out)
            annotations = pd.read_csv(out / "expert_annotations.csv")
            self.assertTrue(annotations.quality_gt.isna().all())
            self.assertNotIn("gt", annotations.columns)
            self.assertNotIn('&quot;gt&quot;', (out / "expert_review.html").read_text())


if __name__ == "__main__":
    unittest.main()
