import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from audit_trout_retrieval import generate, sha


class RetrievalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old, self.new, self.memory = [self.root / name for name in ["old", "new", "memory"]]
        for folder in [self.old, self.new, self.memory]:
            folder.mkdir()
        target, reference = self.root / "target.png", self.root / "reference.png"
        Image.new("RGB", (80, 60), "yellow").save(target)
        Image.new("RGB", (90, 70), "purple").save(reference)
        entry = {"scale_id": "train_1", "fish_key": "train_fish", "path": str(reference), "split": "train",
                 "quality_class": 0, "age_class": 2, "quality_source": "expert_direct",
                 "age_label_source": "expert_direct", "image_sha256": sha(reference)}
        (self.memory / "entries.jsonl").write_text(json.dumps(entry) + "\n")
        policy = {"quality_margin": .05, "age_margin": .05, "quality_pixels": 200704,
                  "age_pixels": 802816, "review_pixels": 1605632, "max_calls": 10,
                  "feedback_rounds": 2, "labels_sha256": "labels", "adapter_hashes": {"age": "a", "quality": "q"},
                  "scoring": "ranking", "memory": None}
        meta = {"labels_sha256": "labels", "adapter_hashes": policy["adapter_hashes"],
                "artifact_hashes": {"entries.jsonl": sha(self.memory / "entries.jsonl")}}
        (self.memory / "memory_config.json").write_text(json.dumps(meta))
        (self.old / "policy_settings.json").write_text(json.dumps(policy))
        policy["memory"] = {"config_sha256": sha(self.memory / "memory_config.json"), "top_k": 2}
        (self.new / "policy_settings.json").write_text(json.dumps(policy))
        for folder in [self.old, self.new]:
            (folder / "run_config.json").write_text(json.dumps({"split": "validation", "training_configs": {}}))
        trace = {"task": "age", "view": "feedback_round_1", "prediction": 2, "margin": .12,
                 "pixels": 1605632, "rotation": 0, "contrast": 1.15}
        self.before = {"scale_id": "target<script>", "fish_key": "heldout", "path": str(target),
                       "gt": 2, "image_sha256": sha(target), "prediction": -1, "status": "expert_review",
                       "reason": "unresolved", "traces": [trace]}
        trace_with_hit = {**trace, "retrieved_training_references": [{"scale_id": "train_1", "fish_key": "train_fish",
                          "path": str(reference), "class": 2, "label_source": "expert_direct",
                          "cosine_similarity": .9, "model_correct": True}]}
        self.after = {**self.before, "prediction": 2, "status": "age_accepted", "traces": [trace_with_hit]}
        self.save_records()

    def save_records(self):
        for folder, row in [(self.old, self.before), (self.new, self.after)]:
            (folder / "predictions.jsonl").write_text(json.dumps(row) + "\n")

    def run_audit(self, **kwargs):
        return generate(self.old, self.new, self.memory, self.root / "report", **kwargs)

    def test_changed_case_and_references_and_escape(self):
        summary = self.run_audit()
        self.assertEqual(summary["changed_transitions"], {"referral -> correct": 1})
        document = (self.root / "report/retrieval_audit.html").read_text()
        self.assertIn("target&lt;script&gt;", document)
        self.assertNotIn("target<script>", document)
        self.assertIn("data:image/jpeg;base64,", document)
        with (self.root / "report/review_annotations_template.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["growth_pattern_similarity"], "")
        with self.assertRaises(FileExistsError):
            self.run_audit()

    def test_unchanged_and_all_cases(self):
        self.before.update(prediction=2, status="age_accepted")
        self.save_records()
        summary = self.run_audit()
        self.assertEqual(summary["changed_cases"], 0)
        other = generate(self.old, self.new, self.memory, self.root / "all", all_cases=True)
        self.assertEqual(other["displayed_cases"], 1)

    def test_gt_or_policy_mismatch(self):
        self.after["gt"] = 3
        self.save_records()
        with self.assertRaisesRegex(ValueError, "GT/image changed"):
            self.run_audit()
        self.after["gt"] = 2
        self.save_records()
        path = self.new / "policy_settings.json"
        policy = json.loads(path.read_text())
        policy["age_margin"] = .01
        path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, "setting changed"):
            self.run_audit()

    def test_image_mutation(self):
        Image.new("RGB", (80, 60), "red").save(self.after["path"])
        with self.assertRaisesRegex(ValueError, "Image bytes changed"):
            self.run_audit()
        self.assertFalse((self.root / "report").exists())

    def test_wrong_reference_label_and_fish_overlap(self):
        self.after["traces"][0]["retrieved_training_references"][0]["class"] = 3
        self.save_records()
        with self.assertRaisesRegex(ValueError, "label source differs"):
            self.run_audit()
        self.after["fish_key"] = self.before["fish_key"] = "train_fish"
        self.save_records()
        with self.assertRaisesRegex(ValueError, "fish overlap"):
            self.run_audit()


if __name__ == "__main__":
    unittest.main()
