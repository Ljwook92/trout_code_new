import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

import trout_agents as agents


class FakeVLM:
    resolved_revision = "test"

    def __init__(self, quality="readable"):
        self.calls = []
        self.quality = quality

    def generate(self, prompt, images):
        self.calls.append((prompt, images))
        value = {"confidence": 0.5, "evidence": "Observed annulus candidate; uncertain."}
        if "TRAINING feedback" in prompt:
            value["candidate_rule"] = "Check whether an annulus candidate persists across sectors."
        elif "Assess whether annuli" in prompt:
            value["decision"] = self.quality
        elif "Independently assess" in prompt:
            value["match"] = "uncertain"
        else:
            value["prediction"] = 0
        return json.dumps(value)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.refs = self.root / "references"
        self.refs.mkdir()
        slides = []
        for slide in range(8, 14):
            image = self.refs / f"slide_{slide:02d}.png"
            Image.new("RGB", (32, 32), "gray").save(image)
            slides.append({"slide": slide, "image": image.name, "sha256": agents.sha(image), "meaning": "Example."})
        agents.atomic_json(self.refs / "manifest.json", {"slides": slides})
        rows = []
        for i, split in enumerate(["train", "train", "validation", "test"]):
            image = self.root / f"scale{i}.png"
            Image.new("RGB", (32, 32), "white").save(image)
            rows.append({"scale_id": f"scale{i}", "fish_key": f"fish{i}", "path": str(image),
                         "label": [1, 6, 2, 3][i], "target": [1, 4, 2, 3][i], "split": split})
        self.master, self.manifest = self.root / "master.csv", self.root / "split.csv"
        pd.DataFrame(rows).to_csv(self.master, index=False)
        pd.DataFrame(rows)[["scale_id", "fish_key", "target", "split"]].to_csv(self.manifest, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def args(self, split="train", feedback=False, memory=None):
        values = ["run", "--master", str(self.master), "--manifest", str(self.manifest),
                  "--references", str(self.refs), "--out", str(self.root / split), "--split", split]
        if feedback:
            values += ["--feedback"]
        if memory:
            values += ["--memory", str(memory)]
        return agents.parser().parse_args(values)

    def test_gate_and_independent_roles(self):
        refs = agents.load_references(self.refs)
        backend = FakeVLM()
        prediction, traces = agents.VisualAgents(backend, refs, self.refs).predict(self.root / "scale0.png")
        self.assertEqual(prediction, 0)
        self.assertEqual([r["role"] for r in traces], ["quality", "age_0", "age_1", "age_2", "age_3", "judge"])
        for prompt, images in backend.calls[:5]:
            self.assertNotIn("expert_gt", prompt)
            self.assertNotIn("previous_evidence", prompt)
            self.assertEqual(images[-1][0], "TARGET: unannotated scale to classify; GT is hidden.")
        for decision, expected in [("bad", 4), ("uncertain", -1)]:
            backend = FakeVLM(decision)
            pred, trace = agents.VisualAgents(backend, refs, self.refs).predict(self.root / "scale0.png")
            self.assertEqual(pred, expected)
            self.assertEqual(len(backend.calls), 1)

    def test_feedback_provenance_resume_and_test_blindness(self):
        backend = FakeVLM()
        args = self.args(feedback=True)
        agents.run(args, backend)
        records = agents.read_jsonl(args.out / "predictions.jsonl")
        memory = agents.read_jsonl(args.out / "reflection_memory.jsonl")
        self.assertEqual(len(records), 2)
        self.assertEqual(len(memory), 2)
        self.assertFalse(memory[0]["expert_verified"])
        config = json.loads((args.out / "run_config.json").read_text())
        cohort = agents.load_cohort(self.master, self.manifest)
        self.assertEqual(agents.load_memory(args.out / "reflection_memory.jsonl", cohort, config["data_signature"]), [])
        self.assertEqual(len(agents.load_memory(args.out / "reflection_memory.jsonl", cohort,
                                               config["data_signature"], True)), 2)
        resumed = FakeVLM()
        agents.run(args, resumed)
        self.assertEqual(resumed.calls, [])
        with self.assertRaises(ValueError):
            agents.run(self.args("test", feedback=True), FakeVLM())
        test_backend = FakeVLM()
        agents.run(self.args("test", memory=args.out / "reflection_memory.jsonl"), test_backend)
        self.assertFalse(any("expert_gt" in prompt or "TRAINING feedback" in prompt for prompt, _ in test_backend.calls))
        memory[0]["source_split"] = "test"
        malicious = self.root / "bad_memory.jsonl"
        agents.append_jsonl(malicious, memory[0])
        with self.assertRaises(ValueError):
            agents.load_memory(malicious, cohort, config["data_signature"], True)
        args.mode = "single"
        with self.assertRaises(ValueError):
            agents.run(args, FakeVLM())

    def test_split_labels_and_reference_checks(self):
        table = pd.read_csv(self.manifest)
        table.loc[2, "fish_key"] = "fish0"
        table.to_csv(self.manifest, index=False)
        with self.assertRaises(ValueError):
            agents.load_cohort(self.master, self.manifest)
        (self.refs / "slide_08.png").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            agents.load_references(self.refs)

    def test_schema_errors_and_abstention_metrics(self):
        self.assertEqual(agents.parse_response('```json\n{"x":1}\n```'), {"x": 1})
        with self.assertRaises(ValueError):
            agents.validate_response("judge", {"prediction": 4, "confidence": .5, "evidence": "x"})
        with self.assertRaises(ValueError):
            agents.validate_response("single", {"prediction": True, "confidence": .5, "evidence": "x"})
        out = self.root / "metrics"
        out.mkdir()
        for i, (gt, pred, error) in enumerate([(0, 0, None), (1, -1, "decode"), (4, 4, None)]):
            agents.append_jsonl(out / "predictions.jsonl", {"scale_id": str(i), "fish_key": str(i),
                "split": "test", "gt": gt, "prediction": pred, "error": error, "seconds": 1})
        metrics = agents.evaluate(out)
        self.assertAlmostEqual(metrics["accuracy"], 2/3)
        self.assertEqual(metrics["error_count"], 1)
        self.assertEqual(metrics["readable_coverage"], .5)
        cm = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
        self.assertEqual(cm.to_numpy().sum(), 3)


if __name__ == "__main__":
    unittest.main()
