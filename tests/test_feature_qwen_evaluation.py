import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from evaluate_trout_feature_qwen import greedy_response, prompt_tokens, final_prediction, inspect_bundle, predict_gated
from evaluate_trout_vlm import summarize
from train_trout_feature_qwen import FeatureProjector, MODEL, REVISION, PROTOCOL, SYSTEM, PROMPTS
from prepare_trout_training_labels import prepare_labels


class Tokenizer:
    eos_token_id = 255

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("ascii"))

    def decode(self, tokens, skip_special_tokens=True):
        return bytes(tokens).decode("ascii")

    def convert_tokens_to_ids(self, text):
        return 255

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(m["role"] + ":" + m["content"] + "\n" for m in messages) + "assistant:"


class ScriptedModel(torch.nn.Module):
    def __init__(self, response):
        super().__init__()
        self.embed = torch.nn.Embedding(256, 16)
        self.script = list(response.encode("ascii")) + [255]
        self.calls = 0

    def get_input_embeddings(self):
        return self.embed

    def forward(self, **kwargs):
        if "pixel_values" in kwargs or "labels" in kwargs or "input_ids" in kwargs:
            raise AssertionError("Inference must use feature embeddings only, without an answer")
        length = kwargs["inputs_embeds"].shape[1]
        if kwargs["position_ids"].shape != (3, 1, length):
            raise AssertionError("Incorrect RoPE shape")
        scores = torch.zeros(1, 1, 256)
        scores[0, 0, self.script[self.calls]] = 10
        self.calls += 1
        return SimpleNamespace(logits=scores)


class FeatureEvaluationTests(unittest.TestCase):
    def test_bad_and_unknown_gate_never_call_age(self):
        for quality in [1, -1]:
            calls = []
            def predict(image, task):
                calls.append(task)
                if task != "quality":
                    raise AssertionError("Age must not be invoked")
                return quality, "quality response", None
            gate, age, skipped = predict_gated(None, predict)
            self.assertEqual(calls, ["quality"])
            self.assertEqual(age, (-1, None, None))
            self.assertTrue(skipped)

    def test_readable_gate_calls_age_and_schema_error_does_not(self):
        calls = []
        def predict(image, task):
            calls.append(task)
            return (0, "readable", None) if task == "quality" else (2, "age", None)
        gate, age, skipped = predict_gated(None, predict)
        self.assertEqual(calls, ["quality", "age"])
        self.assertEqual(age[0], 2)
        self.assertFalse(skipped)
        _, age, skipped = predict_gated(None, lambda image, task: (0, "broken", "schema error"))
        self.assertTrue(skipped)
        self.assertEqual(age[0], -1)

    def test_gated_age_metrics_separate_skipped_and_passed_readable(self):
        records = [{"model": "feature_qwen", "quality_gt": "readable", "age_gt": 0,
                    "quality_prediction": 1, "age_prediction": -1, "pipeline_gt": 0, "pipeline_prediction": 4},
                   {"model": "feature_qwen", "quality_gt": "readable", "age_gt": 1,
                    "quality_prediction": 0, "age_prediction": 1, "pipeline_gt": 1, "pipeline_prediction": 1},
                   {"model": "feature_qwen", "quality_gt": "bad", "age_gt": -1,
                    "quality_prediction": 1, "age_prediction": -1, "pipeline_gt": 4, "pipeline_prediction": 4}]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            summarize(records, out, age_gated=True)
            result = pd.read_csv(out / "comparison.csv").set_index("task")
            self.assertEqual(result.loc["age_all_readable_gated", "accuracy"], 0.5)
            self.assertEqual(result.loc["age_gate_passed_readable", "accuracy"], 1)
            self.assertEqual(result.loc["age_gate_passed_readable", "support"], 1)
            self.assertNotIn("age_gt_readable", result.index)

    def test_greedy_feature_only_generation_and_stop(self):
        tokenizer = Tokenizer()
        model = ScriptedModel('{"prediction": 2}')
        projector = FeatureProjector(16, grid=1)
        raw, error = greedy_response(model, projector, torch.randn(1, 1, 512),
                                    prompt_tokens(tokenizer, "age"), tokenizer)
        self.assertEqual(json.loads(raw), {"prediction": 2})
        self.assertIsNone(error)

    def test_incomplete_generation_abstains(self):
        raw, error = greedy_response(ScriptedModel('{"prediction": 2}'),
            FeatureProjector(16, grid=1), torch.randn(1, 1, 512), ([1], [2]), Tokenizer(), max_new_tokens=2)
        self.assertIsNotNone(error)
        self.assertEqual(len(raw), 2)

    def test_pipeline_gate_overrides_age(self):
        self.assertEqual(final_prediction(1, 2), 4)
        self.assertEqual(final_prediction(0, 2), 2)
        self.assertEqual(final_prediction(-1, 2), -1)
        self.assertEqual(final_prediction(0, -1), -1)

    def test_bundle_identity_and_saved_partitions(self):
        master = pd.DataFrame({"scale_id": list("abcdefgh"), "fish_key": list("abcdefgh"),
            "path": ["image.png"] * 8, "label": [0, 1, 2, 3] * 2})
        manifest = master[["scale_id", "fish_key"]].copy()
        manifest["split"] = ["train"] * 4 + ["validation"] * 4
        table = prepare_labels(master, manifest)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            adapter = run / "best_bundle/qwen_adapter"
            adapter.mkdir(parents=True)
            for path in [run / "encoder.pt", run / "best_bundle/projector.pt",
                         adapter / "adapter_model.safetensors", adapter / "adapter_config.json",
                         run / "best_bundle/selection.json"]:
                path.write_bytes(b"test fixture")
            config = {"protocol": PROTOCOL, "task": "age", "model": MODEL, "revision": REVISION,
                      "labels_sha256": "hash", "qwen_pixel_input": False, "pseudo_labels": False,
                      "distillation": False, "system": SYSTEM, "prompt": PROMPTS["age"],
                      "image_size": 224, "grid": 7}
            (run / "training_config.json").write_text(json.dumps(config))
            table.to_csv(run / "used_rows.csv", index=False)
            inspect_bundle(run, "age", table, "hash")
            with self.assertRaises(ValueError):
                inspect_bundle(run, "age", table, "changed-labels")
            with self.assertRaises(ValueError):
                inspect_bundle(run, "quality", table, "hash")
            table.loc[0, "split"] = "test"
            table.to_csv(run / "used_rows.csv", index=False)
            with self.assertRaises(ValueError):
                inspect_bundle(run, "age", prepare_labels(master, manifest), "hash")


if __name__ == "__main__":
    unittest.main()
