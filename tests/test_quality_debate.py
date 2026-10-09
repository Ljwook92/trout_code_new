import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from train_trout_quality_agents import training_rows
from trout_agents import sha
from trout_quality_debate import QualityDebate, prepare_feedback, public_history, quality_rows, validate_evidence, write_reports


def assessment(pred=0, margin=.2, state=None):
    states = ["preserved"] * 3
    if pred == 1:
        states[0] = "damaged"
    if state:
        states[0] = state
    return {"prediction": pred, "margin": margin, "evidence": {"checks": [
        {"criterion": c, "state": s, "observation": "Visible structure inspected",
         "bbox": [.2, .2, .4, .4] if s == "damaged" else None}
        for c, s in zip(["center", "material", "visibility"], states)], "peer_response": "Compared physical evidence"}}


class FakeTools:
    def __init__(self, results):
        self.results, self.calls = iter(results), []

    def assess(self, path, role, pixels, history):
        self.calls.append((path, role, pixels, copy.deepcopy(history)))
        return copy.deepcopy(next(self.results))


class QualityDebateTests(unittest.TestCase):
    policy = {"pixels": 200704, "review_pixels": 802816, "max_rounds": 3, "margin": .05}

    def test_independent_readable_agreement_and_bad_stop(self):
        for pred in [0, 1]:
            tools = FakeTools([assessment(pred), assessment(pred)])
            result = QualityDebate(tools, self.policy).run("anonymous.png")
            self.assertEqual(result["prediction"], pred)
            self.assertEqual(result["age_allowed"], pred == 0)
            self.assertEqual(len(tools.calls), 2)
            self.assertEqual(tools.calls[0][3], [])
            self.assertEqual(tools.calls[1][3], [])

    def test_changed_decision_requires_stability_and_synchronous_history(self):
        tools = FakeTools([assessment(1), assessment(0), assessment(0), assessment(0), assessment(0), assessment(0)])
        result = QualityDebate(tools, self.policy).run("anonymous.png")
        self.assertEqual(result["prediction"], 0)
        self.assertEqual(len(result["rounds"]), 3)
        self.assertEqual(tools.calls[2][3], tools.calls[3][3])
        self.assertEqual(tools.calls[2][2], 802816)
        self.assertEqual(len(tools.calls[2][3]), 1)

    def test_low_margin_uncertainty_and_unsupported_decision_refer(self):
        for item in [assessment(0, .01), assessment(0, state="uncertain"), assessment(1, state="preserved")]:
            result = QualityDebate(FakeTools([item] * 6), self.policy).run("anonymous.png")
            self.assertEqual(result["prediction"], -1)
            self.assertFalse(result["age_allowed"])

    def test_schema_error_and_unlocalized_damage_refer(self):
        item = assessment(1)
        item["evidence"]["checks"][0]["bbox"] = None
        with self.assertRaisesRegex(ValueError, "localized"):
            validate_evidence(item["evidence"])
        result = QualityDebate(FakeTools([item] * 6), self.policy).run("anonymous.png")
        self.assertEqual(result["prediction"], -1)
        self.assertIsNotNone(result["rounds"][0]["assessments"][0]["error"])

    def test_raw_schema_failure_is_preserved_and_last_change_cannot_pass(self):
        failure = {"prediction": -1, "margin": 0, "evidence": None, "raw_evidence": "invalid JSON", "error": "bad schema"}
        result = QualityDebate(FakeTools([failure] * 6), self.policy).run("anonymous.png")
        self.assertEqual(result["rounds"][0]["assessments"][0]["raw_evidence"], "invalid JSON")
        items = [assessment(0), assessment(1), assessment(1), assessment(0), assessment(0), assessment(0)]
        result = QualityDebate(FakeTools(items), self.policy).run("anonymous.png")
        self.assertEqual(result["prediction"], -1)

    def test_history_whitelist_excludes_gt_and_identity(self):
        item = {**assessment(), "role": "bad", "quality_gt": "bad", "fish_key": "secret", "path": "hidden"}
        history = public_history([{"round": 1, "assessments": [item], "quality_gt": "bad"}])
        text = json.dumps(history)
        self.assertNotIn("quality_gt", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("hidden", text)

    def test_rows_and_feedback_train_only_and_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "scale.png"
            Image.new("RGB", (64, 64)).save(image)
            table = pd.DataFrame([{"scale_id": split, "fish_key": split + "fish", "path": str(image),
                "split": split, "quality_gt": "readable", "quality_source": "expert_direct"}
                for split in ["train", "validation", "test"]])
            labels = root / "labels.csv"
            table.to_csv(labels, index=False)
            train, val = training_rows(table, "bad")
            self.assertEqual(list(train.scale_id), ["train"])
            self.assertEqual(list(val.scale_id), ["validation"])
            leak = table.copy()
            leak["fish_key"] = "same"
            with self.assertRaisesRegex(ValueError, "leakage"):
                quality_rows(leak, "train")
            run = root / "run"
            run.mkdir()
            cfg = {"split": "train", "labels_sha256": sha(labels)}
            (run / "policy_settings.json").write_text(json.dumps(cfg))
            prediction = QualityDebate(FakeTools([assessment(1), assessment(1)]), self.policy).run(str(image))
            record = {**prediction, "scale_id": "train", "fish_key": "trainfish", "path": str(image),
                      "quality_gt": "readable", "image_sha256": sha(image)}
            (run / "predictions.jsonl").write_text(json.dumps(record) + "\n")
            feedback = root / "feedback.jsonl"
            self.assertEqual(prepare_feedback(labels, run, feedback), 2)
            trained, val = training_rows(table, "bad", feedback, sha(labels))
            self.assertEqual(len(trained), 2)
            self.assertTrue(trained.answer.eq('{"decision": "readable"}').all())
            self.assertNotIn('"quality_gt"', trained.iloc[1]["prompt"])
            # GT corrects the target, NOT the agent's own observation text.
            self.assertFalse(json.loads(feedback.read_text().splitlines()[0])["reasoning_target"])
            cfg["split"] = "validation"
            (run / "policy_settings.json").write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                prepare_feedback(labels, run, root / "invalid.jsonl")
            metrics = write_reports([record], run)
            self.assertEqual(metrics["readable_rejected_as_bad"], 1)
            referred = {**record, "prediction": -1, "status": "expert_review", "age_allowed": False}
            metrics = write_reports([referred], run)
            self.assertEqual(metrics["accuracy_abstention_as_wrong"], 0)
            self.assertNotIn("quality_gt", (run / "expert_review.jsonl").read_text())

    def test_training_and_inference_cli_dry_runs_without_model(self):
        from train_trout_quality_agents import main as train_main
        from trout_quality_debate import DECISION_PROMPT, main as run_main, system_prompt
        from train_trout_vlm import MODEL, REVISION
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "scale.png"
            Image.new("RGB", (64, 64)).save(image)
            labels = root / "labels.csv"
            pd.DataFrame([{"scale_id": split, "fish_key": split, "path": str(image), "split": split,
                          "quality_gt": "readable", "quality_source": "expert_direct"}
                          for split in ["train", "validation", "test"]]).to_csv(labels, index=False)
            with patch("sys.argv", ["train", "--labels", str(labels), "--out", str(root / "trainout"), "--dry-run"]):
                train_main()
            self.assertFalse((root / "trainout").exists())
            for role in ["bad", "readable"]:
                folder = root / role / "best_adapter"
                folder.mkdir(parents=True)
                (folder / "adapter_model.safetensors").write_bytes(b"fake, never loaded")
                (folder.parent / "training_config.json").write_text(json.dumps({"task": "quality", "quality_role": role,
                    "model": MODEL, "revision": REVISION, "labels_sha256": sha(labels),
                    "system": system_prompt(role), "prompt": DECISION_PROMPT}))
            argv = ["run", "run", "--labels", str(labels), "--bad-adapter", str(root / "bad/best_adapter"),
                    "--readable-adapter", str(root / "readable/best_adapter"), "--out", str(root / "eval"), "--dry-run"]
            with patch("sys.argv", argv):
                run_main()
            self.assertFalse((root / "eval").exists())
            with patch("sys.argv", argv + ["--split", "test"]):
                with self.assertRaisesRegex(ValueError, "frozen-policy"):
                    run_main()


if __name__ == "__main__":
    unittest.main()
