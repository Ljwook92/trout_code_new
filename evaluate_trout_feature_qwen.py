"""Evaluate frozen SimCLR/projector/Qwen bundles; no pixels or GT enter Qwen."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps

from evaluate_trout_vlm import decision, evaluation_rows, summarize
from train_trout_feature_qwen import (
    MODEL, REVISION, PROTOCOL, MARKER, SYSTEM, PROMPTS,
    FeatureProjector, FrozenScaleEncoder, expert_rows, sha,
)


def inspect_bundle(run, task, table, label_hash):
    config = json.loads((run / "training_config.json").read_text())
    if (config.get("protocol") != PROTOCOL or config.get("task") != task
            or config.get("model") != MODEL or config.get("revision") != REVISION
            or config.get("labels_sha256") != label_hash):
        raise ValueError("Feature bundle protocol/task/model/labels differ from this evaluation")
    if (config.get("qwen_pixel_input") is not False or config.get("pseudo_labels") is not False
            or config.get("distillation") is not False or config.get("system") != SYSTEM
            or config.get("prompt") != PROMPTS[task] or config.get("image_size") != 224
            or config.get("grid") not in [1, 7]):
        raise ValueError("Unsupported feature input or supervision configuration")
    bundle = run / "best_bundle"
    files = {"encoder": run / "encoder.pt", "projector": bundle / "projector.pt",
             "adapter": bundle / "qwen_adapter" / "adapter_model.safetensors",
             "adapter_config": bundle / "qwen_adapter" / "adapter_config.json",
             "selection": bundle / "selection.json"}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    train, val = expert_rows(table, task)
    expected = pd.concat([train, val])[["scale_id", "fish_key", "split"]].sort_values("scale_id").reset_index(drop=True)
    saved = pd.read_csv(run / "used_rows.csv")
    saved = saved[["scale_id", "fish_key", "split"]].sort_values("scale_id").reset_index(drop=True)
    if not expected.equals(saved):
        raise ValueError("Saved training fish/image partitions do not match the current table")
    return config, {key + "_sha256": sha(path) for key, path in files.items()}


def prompt_tokens(tokenizer, task):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": MARKER + "\n" + PROMPTS[task]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    before, after = prefix.split(MARKER)
    return (tokenizer.encode(before, add_special_tokens=False),
            tokenizer.encode(after, add_special_tokens=False))


@torch.inference_mode()
def greedy_response(model, projector, features, tokens, tokenizer, max_new_tokens=48):
    before, after = tokens
    embed = model.get_input_embeddings()
    device = features.device
    left = embed(torch.tensor([before], device=device))
    right = embed(torch.tensor([after], device=device))
    inputs = torch.cat([left, projector(features).to(left.dtype), right], dim=1)
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    stop_ids.discard(None)
    generated = []
    # Bounded cache-free decoding keeps explicit text RoPE identical to training.
    # Only generated tokens are appended: there is no ground-truth answer prefix.
    for _ in range(max_new_tokens):
        length = inputs.shape[1]
        positions = torch.arange(length, device=device).view(1, 1, -1).expand(3, 1, -1)
        output = model(inputs_embeds=inputs,
                       attention_mask=torch.ones((1, length), dtype=torch.long, device=device),
                       position_ids=positions, use_cache=False, logits_to_keep=1)
        token = int(output.logits[0, -1].argmax())
        if token in stop_ids:
            break
        generated.append(token)
        inputs = torch.cat([inputs, embed(torch.tensor([[token]], device=device))], dim=1)
    else:
        return tokenizer.decode(generated, skip_special_tokens=True), "Maximum generation length reached"
    return tokenizer.decode(generated, skip_special_tokens=True), None


def final_prediction(quality, age):
    return 4 if quality == 1 else age if quality == 0 else -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--quality-run", type=Path, required=True)
    parser.add_argument("--age-run", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--confirm-test", action="store_true", help="Explicit final held-out test evaluation")
    parser.add_argument("--limit", type=int, default=0, help="0: full partition; otherwise exploratory pilot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("Limit cannot be negative")
    if args.split == "test" and not args.confirm_test:
        raise ValueError("Use validation first; --confirm-test is required for final test evaluation")
    table = pd.read_csv(args.labels)
    rows = evaluation_rows(table, args.split)
    if "quality_source" not in rows or not rows.quality_source.isin(
            ["expert_direct", "expert_quality_csv"]).all():
        raise ValueError("Evaluation quality must be image-specific expert GT")
    if args.limit:
        rows = rows.sample(frac=1, random_state=100).head(args.limit).reset_index(drop=True)
    if not rows.path.map(lambda p: Path(str(p)).is_file()).all():
        raise FileNotFoundError("Missing evaluation images")
    label_hash = sha(args.labels)
    runs = {"quality": args.quality_run, "age": args.age_run}
    configs, hashes, encoders, projectors = {}, {}, {}, {}
    for task, run in runs.items():
        config, digest = inspect_bundle(run, task, table, label_hash)
        configs[task], hashes[task] = config, digest
        state = torch.load(run / "encoder.pt", map_location="cpu", weights_only=True)
        encoders[task] = FrozenScaleEncoder(state["backbone_state_dict"], config["grid"])
        projectors[task] = FeatureProjector(config["hidden_size"], config["grid"])
        projectors[task].load_state_dict(torch.load(
            run / "best_bundle/projector.pt", map_location="cpu", weights_only=True), strict=True)
    print(f"split={args.split} scales={len(rows)}; feature input only; frozen models", flush=True)
    print(rows.quality_gt.value_counts().to_string(), flush=True)
    if args.dry_run:
        print("Dry run passed: both bundles/weights/GT/splits/images checked; no Qwen loaded or output written")
        return
    if args.out.exists():
        raise FileExistsError("Choose a new output directory to preserve prior predictions")
    if not torch.cuda.is_available():
        raise RuntimeError("An allocated CUDA GPU is required")
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    from torchvision import transforms as T

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(runs["quality"] / "best_bundle/qwen_adapter"),
                                    adapter_name="quality", is_trainable=False)
    model.load_adapter(str(runs["age"] / "best_bundle/qwen_adapter"), adapter_name="age", is_trainable=False)
    model.to("cuda")
    base.model.visual.to("cpu")
    model.eval()
    hidden = model.get_input_embeddings().weight.shape[1]
    for task in runs:
        if configs[task]["hidden_size"] != hidden:
            raise ValueError("Projector/Qwen embedding dimensions differ")
        encoders[task].to("cuda").eval()
        projectors[task].to("cuda").eval()
    tokenizers = {task: AutoTokenizer.from_pretrained(run / "best_bundle/qwen_adapter")
                  for task, run in runs.items()}
    tokens = {task: prompt_tokens(tokenizers[task], task) for task in runs}
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def predict(image_tensor, task):
        model.set_adapter(task)
        model.eval()
        # Match training: the frozen ResNet runs in float32, outside autocast.
        with torch.inference_mode():
            features = encoders[task](image_tensor)
            with torch.autocast("cuda", dtype=dtype):
                raw, error = greedy_response(model, projectors[task], features, tokens[task], tokenizers[task])
        if error is not None:
            return -1, raw, error
        try:
            return decision(raw, task), raw, None
        except ValueError as exc:
            return -1, raw, str(exc)

    args.out.mkdir(parents=True)
    (args.out / "evaluation_config.json").write_text(json.dumps({
        "protocol": PROTOCOL, "split": args.split, "limit": args.limit,
        "labels_sha256": label_hash, "training_configs": configs, "bundle_hashes": hashes,
        "decoding": "greedy_free_generation_no_cache_max_48_tokens", "test_feedback": False,
        "gt_used_for_prediction": False, "qwen_pixel_input": False,
        "note": "Oracle-readable age scoring is separate from the predicted quality gate."}, indent=2))
    records = []
    for row in rows.to_dict("records"):
        with Image.open(row["path"]) as image:
            image_tensor = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to("cuda")
        quality, qraw, qerror = predict(image_tensor, "quality")
        # Infer age for EVERY image, irrespective of GT or gate, to isolate model performance.
        # Age on GT-bad scales is not meaningful and is excluded from age-only metrics.
        age, araw, aerror = predict(image_tensor, "age")
        readable = row["quality_gt"] == "readable"
        gt = int(row["age4"]) if readable else 4
        record = {"model": "feature_qwen", "scale_id": row["scale_id"], "fish_key": row["fish_key"],
                  "quality_gt": row["quality_gt"], "age_gt": gt if readable else -1,
                  "quality_prediction": quality, "age_prediction": age,
                  "pipeline_gt": gt, "pipeline_prediction": final_prediction(quality, age),
                  "quality_raw": qraw, "age_raw": araw, "quality_error": qerror,
                  "age_error": aerror, "image_sha256": sha(row["path"])}
        records.append(record)
        with (args.out / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"{row['scale_id']}: gate={quality} age={age} final={record['pipeline_prediction']} GT={gt}", flush=True)
    summarize(records, args.out)


if __name__ == "__main__":
    main()
