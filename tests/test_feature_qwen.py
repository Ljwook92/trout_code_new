import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from train_trout_feature_qwen import (
    FeatureProjector, FrozenScaleEncoder, answer_tokens, backbone_state,
    check_encoder_provenance, expert_rows, feature_batch,
    class_sampling_weights, epoch_rows, validation_metrics, selection_key,
)
from prepare_trout_training_labels import prepare_labels


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 50 for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(m["role"] + ":" + m["content"] + "\n" for m in messages)
        return text + ("assistant:" if add_generation_prompt else "")


class FeatureQwenTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_only_response_supervised_and_gradient_reaches_projector_and_lora(self):
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        config = Qwen2_5_VLConfig(
            text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 1, "num_attention_heads": 2,
                         "num_key_value_heads": 2,
                         "rope_scaling": {"rope_type": "default", "mrope_section": [2, 2, 4]}},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 2, "out_hidden_size": 32},
            image_token_id=61, video_token_id=62, vision_start_token_id=60)
        base = Qwen2_5_VLForConditionalGeneration(config)
        model = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj", "v_proj"],
                                               task_type="CAUSAL_LM"))
        projector = FeatureProjector(32, grid=1)
        tokens = answer_tokens(CharacterTokenizer(), "age", '{"prediction": 2}')
        batch = feature_batch(model, projector, torch.randn(1, 1, 512), tokens)
        before, tail, masked = tokens
        self.assertTrue(batch["labels"][:, :len(before) + 1 + masked].eq(-100).all())
        self.assertEqual(batch["labels"][0, len(before) + 1 + masked:].tolist(), tail[masked:])
        self.assertNotIn("pixel_values", batch)
        def forbidden(*args, **kwargs):
            raise AssertionError("Qwen vision encoder was called")
        base.model.visual.forward = forbidden
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(projector.net[1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0
                            for name, p in model.named_parameters() if "lora_" in name))
        from evaluate_trout_feature_qwen import greedy_response
        from tests.test_feature_qwen_evaluation import Tokenizer
        model.eval()
        raw, _ = greedy_response(model, projector, torch.randn(1, 1, 512),
                                 ([1], [2]), Tokenizer(), max_new_tokens=1)
        self.assertIsInstance(raw, str)

    def test_frozen_encoder_matches_pooled_backbone_and_preserves_batchnorm(self):
        from torchvision.models import resnet18
        base = resnet18(weights=None).eval()
        base.fc = torch.nn.Identity()
        state = base.state_dict()
        encoder = FrozenScaleEncoder(state, grid=7)
        inputs = torch.randn(1, 3, 224, 224)
        initial = encoder.layers[1].running_mean.clone()
        encoder.train()
        features = encoder(inputs)
        self.assertEqual(tuple(features.shape), (1, 50, 512))
        self.assertFalse(features.requires_grad)
        self.assertTrue(torch.equal(initial, encoder.layers[1].running_mean))
        with torch.no_grad():
            torch.testing.assert_close(features[:, 0], base(inputs))
        nested = {"backbone." + k: v for k, v in state.items()}
        nested["head.weight"] = torch.randn(4, 512)
        extracted = backbone_state({"model_state_dict": nested, "feature_dim": 512})
        self.assertEqual(set(extracted), set(state))
        with self.assertRaises(ValueError):
            backbone_state({"backbone": "resnet50"})

    def test_manifest_binding_and_fish_leak_rejected(self):
        manifest = pd.DataFrame({"scale_id": ["a", "b", "c"], "fish_key": ["f1", "f2", "f3"],
                                 "target": [0, 1, 4], "split": ["train", "validation", "test"]})
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            manifest.to_csv(run / "split_manifest.csv", index=False)
            digest = hashlib.sha256(pd.util.hash_pandas_object(
                manifest.sort_values("scale_id"), index=False).values.tobytes()).hexdigest()
            settings = {"split_sha256": digest, "cohort_sha256": "cohort"}
            (run / "search_config.json").write_text(json.dumps(settings))
            signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            checkpoint = {"search_signature": signature}
            check_encoder_provenance(manifest, run, checkpoint)
            wrong = manifest.copy()
            wrong.loc[1, "split"] = "train"
            with self.assertRaises(ValueError):
                check_encoder_provenance(wrong, run, checkpoint)
            with self.assertRaises(ValueError):
                check_encoder_provenance(manifest, run, {"search_signature": "other"})

    def test_expert_only_training_excludes_propagated_age_and_unknown_quality(self):
        master = pd.DataFrame({"scale_id": list("abcdefghi"),
            "fish_key": ["f0", "f1", "f2", "f3", "f0", "v0", "v1", "v2", "v3"],
            "path": ["image.png"] * 9, "label": [0, 1, 2, 3, None, 0, 1, 2, 3]})
        manifest = master.drop(index=4)[["scale_id", "fish_key"]].copy()
        manifest["split"] = ["train"] * 4 + ["validation"] * 4
        table = prepare_labels(master, manifest,
            pd.DataFrame({"scale_id": ["e"], "quality_gt": ["readable"]}))
        train, val = expert_rows(table, "age")
        self.assertEqual(set(train.scale_id), set("abcd"))
        self.assertEqual(len(val), 4)
        table.loc[0, "split"] = "test"
        with self.assertRaises(ValueError):
            expert_rows(table, "age")

    def test_balanced_sampling_uses_train_counts_and_is_reproducible(self):
        frame = pd.DataFrame({"scale_id": [f"s{i}" for i in range(100)],
                              "answer": ["common"] * 90 + ["rare"] * 10,
                              "split": ["train"] * 100})
        weights = class_sampling_weights(frame)
        mass = weights.groupby(frame.answer).sum()
        self.assertAlmostEqual(mass["common"], mass["rare"])
        a = epoch_rows(frame, 100, True)
        b = epoch_rows(frame, 100, True)
        self.assertTrue(a.equals(b))
        self.assertEqual(len(a), len(frame))
        self.assertTrue(set(a.scale_id) <= set(frame.scale_id))
        self.assertTrue(a["split"].eq("train").all())
        self.assertGreater(a.answer.eq("rare").sum(), 25)
        ordinary = epoch_rows(frame, 100, False)
        self.assertEqual(set(ordinary.scale_id), set(frame.scale_id))
        self.assertEqual(ordinary.answer.eq("rare").sum(), 10)

    def test_generated_validation_metrics_count_abstentions_as_wrong(self):
        metrics = validation_metrics([0, 1, 2, 3], [0, 1, 1, -1], "age")
        self.assertEqual(metrics["validation_accuracy"], 0.5)
        self.assertEqual(metrics["validation_balanced_accuracy"], 0.5)
        self.assertEqual(metrics["validation_abstention_rate"], 0.25)
        self.assertEqual(metrics["validation_recall_3"], 0)
        quality = validation_metrics([0, 0, 0, 1, 1], [0, 1, -1, 0, 1], "quality")
        self.assertEqual(quality["validation_bad_pass_rate"], 0.5)
        self.assertAlmostEqual(quality["validation_readable_coverage"], 1 / 3)
        with self.assertRaises(ValueError):
            validation_metrics([0, 1, 2, 3], [0], "age")

    def test_selection_uses_macro_f1_and_loss_only_breaks_ties(self):
        low_f1 = {"validation_macro_f1": 0.70, "validation_loss": 0.05}
        high_f1 = {"validation_macro_f1": 0.80, "validation_loss": 0.10}
        self.assertGreater(selection_key(high_f1, "validation_macro_f1"),
                           selection_key(low_f1, "validation_macro_f1"))
        self.assertGreater(selection_key(low_f1, "validation_answer_loss"),
                           selection_key(high_f1, "validation_answer_loss"))
        tie = {"validation_macro_f1": 0.80, "validation_loss": 0.09}
        self.assertGreater(selection_key(tie, "validation_macro_f1"),
                           selection_key(high_f1, "validation_macro_f1"))


if __name__ == "__main__":
    unittest.main()
