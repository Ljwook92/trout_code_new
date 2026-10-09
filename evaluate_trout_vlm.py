"""Evaluate frozen quality/age LoRA adapters with paired optional base predictions."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from train_trout_vlm import MODEL, REVISION, transform_image
from trout_agents import parse_response


def evaluation_rows(table, split):
    required = {"scale_id", "fish_key", "path", "split", "quality_gt", "age4", "age_label_source"}
    if not required <= set(table) or table.scale_id.isna().any() or table.scale_id.duplicated().any():
        raise ValueError("Invalid supervised label table")
    if table.fish_key.isna().any() or table.groupby("fish_key")["split"].nunique().gt(1).any():
        raise ValueError("Missing fish identities or fish leakage")
    rows = table[table["split"].eq(split) & table.quality_gt.isin(["readable", "bad"])].copy()
    if rows.empty:
        raise ValueError("No expert quality labels in the requested partition")
    readable = rows.quality_gt.eq("readable")
    if not rows.loc[readable, "age_label_source"].eq("expert_direct").all():
        raise ValueError("Held-out ages must be direct expert GT")
    if not rows.loc[readable, "age4"].isin(range(4)).all():
        raise ValueError("Invalid held-out age GT")
    return rows.sort_values("scale_id").reset_index(drop=True)


def decision(raw, task):
    value = parse_response(raw)
    if task == "quality":
        if value.get("decision") not in {"readable", "bad"}:
            raise ValueError("Expected readable or bad")
        return 1 if value["decision"] == "bad" else 0
    pred = value.get("prediction")
    if type(pred) is not int or pred not in range(4):
        raise ValueError("Expected integer age prediction 0..3")
    return pred


def summarize(records, out, age_gated=False):
    data = pd.DataFrame(records)
    results = []
    for model, rows in data.groupby("model"):
        readable = rows.quality_gt.eq("readable")
        age_names = ["0", "1", "2", "3 or older"]
        tasks = [
            ("quality", rows, (rows.quality_gt == "bad").astype(int), rows.quality_prediction, ["readable", "bad"]),
            ("age_all_readable_gated" if age_gated else "age_gt_readable",
             rows[readable], rows.loc[readable, "age_gt"], rows.loc[readable, "age_prediction"], age_names),
            ("pipeline", rows, rows.pipeline_gt, rows.pipeline_prediction, ["0", "1", "2", "3 or older", "bad"]),
        ]
        if age_gated:
            passed_readable = readable & rows.quality_prediction.eq(0)
            tasks.insert(2, ("age_gate_passed_readable", rows[passed_readable],
                            rows.loc[passed_readable, "age_gt"], rows.loc[passed_readable, "age_prediction"], age_names))
        for task, subset, truth, pred, names in tasks:
            if subset.empty:
                continue
            truth, pred = truth.astype(int), pred.astype(int)
            labels = list(range(len(names)))
            recalls = [(pred[truth.eq(c)] == c).mean() for c in labels if truth.eq(c).any()]
            results.append({"model": model, "task": task, "support": len(subset),
                            "accuracy": accuracy_score(truth, pred), "balanced_accuracy": sum(recalls) / len(recalls),
                            "macro_f1": f1_score(truth, pred, labels=labels, average="macro", zero_division=0),
                            "abstention_rate": float(pred.eq(-1).mean())})
            report = classification_report(truth, pred, labels=labels, target_names=names, digits=4, zero_division=0)
            (out / f"{model}_{task}_report.txt").write_text(report)
            # Include the abstention column so errors do not disappear from the matrix.
            cm = confusion_matrix(truth, pred, labels=labels + [-1])[:len(labels), :]
            last_column = "skipped_or_abstain" if task == "age_all_readable_gated" else "abstain"
            pd.DataFrame(cm, index=names, columns=names + [last_column]).to_csv(out / f"{model}_{task}_confusion.csv")
        bad = ~readable
        accepted = rows.quality_prediction.eq(0)
        rates = {"readable_coverage": float(accepted[readable].mean()) if readable.any() else None,
                 "bad_pass_rate": float(accepted[bad].mean()) if bad.any() else None,
                 "accepted_readable_age_accuracy": float(rows.loc[readable & accepted, "age_prediction"].eq(
                     rows.loc[readable & accepted, "age_gt"]).mean()) if (readable & accepted).any() else None}
        (out / f"{model}_coverage.json").write_text(json.dumps(rates, indent=2))
    result = pd.DataFrame(results)
    result.to_csv(out / "comparison.csv", index=False)
    print(result.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--quality-adapter", type=Path, required=True)
    parser.add_argument("--age-adapter", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=0, help="0 means all; limited pilot is exploratory")
    parser.add_argument("--compare-base", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rotation", type=float, default=0, help="Fixed evaluation perturbation in degrees")
    parser.add_argument("--brightness", type=float, default=1)
    parser.add_argument("--contrast", type=float, default=1)
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("Limit cannot be negative")
    import math
    if not all(math.isfinite(x) for x in [args.rotation, args.brightness, args.contrast]) or min(args.brightness, args.contrast) <= 0:
        raise ValueError("Invalid evaluation perturbation")
    table = pd.read_csv(args.labels)
    rows = evaluation_rows(table, args.split)
    if args.limit:
        rows = rows.sample(frac=1, random_state=100).head(args.limit).reset_index(drop=True)
    configs = {}
    label_hash = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    for task, folder in [("quality", args.quality_adapter), ("age", args.age_adapter)]:
        config = json.loads((folder.parent / "training_config.json").read_text())
        if config["task"] != task or config["labels_sha256"] != label_hash:
            raise ValueError("Adapter task or label table differs from training")
        if config["model"] != MODEL or config["revision"] != REVISION:
            raise ValueError("Adapter base model/revision differs")
        for filename in ["adapter_config.json", "adapter_model.safetensors"]:
            if not (folder / filename).is_file():
                raise FileNotFoundError(folder / filename)
        configs[task] = config
    if not rows.path.map(lambda p: Path(str(p)).is_file()).all():
        raise FileNotFoundError("Missing evaluation images")
    print(f"split={args.split} scales={len(rows)}; limited={bool(args.limit)}", flush=True)
    print(rows.quality_gt.value_counts().to_string(), flush=True)
    if args.dry_run:
        print("Dry run passed. No GPU model loaded.")
        return
    if args.out.exists():
        raise FileExistsError("Choose a new output directory; previous predictions are preserved")
    import torch
    from PIL import Image, ImageOps
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Allocated CUDA GPU required")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16, attn_implementation="eager").to("cuda:0")
    model = PeftModel.from_pretrained(base, str(args.quality_adapter), adapter_name="quality", is_trainable=False)
    model.load_adapter(str(args.age_adapter), adapter_name="age", is_trainable=False)
    model.config.use_cache = True
    processors = {task: AutoProcessor.from_pretrained(
        MODEL, revision=REVISION, use_fast=False, min_pixels=56 * 56, max_pixels=cfg["max_pixels"])
        for task, cfg in configs.items()}

    def predict(path, task, mode):
        processor, config = processors[task], configs[task]
        model.set_adapter(task)
        model.eval()
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        image = transform_image(image, args.rotation, args.brightness, args.contrast)
        messages = [{"role": "system", "content": config["system"]}, {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": config["prompt"]}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
        with (model.disable_adapter() if mode == "base" else nullcontext()), torch.inference_mode():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=64)
        raw = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        try:
            pred, error = decision(raw, task), None
        except ValueError as exc:
            pred, error = -1, str(exc)
        return pred, raw, error

    args.out.mkdir(parents=True)
    provenance = {"split": args.split, "limit": args.limit, "labels_sha256": label_hash,
                  "compare_base": args.compare_base, "model": MODEL, "revision": REVISION,
                  "training_configs": configs, "test_feedback": False,
                  "age_inference_policy": "predicted_readable_only_no_GT_override",
                  "fixed_perturbation": {"rotation": args.rotation, "brightness": args.brightness,
                                         "contrast": args.contrast}}
    for task, folder in [("quality", args.quality_adapter), ("age", args.age_adapter)]:
        provenance[task + "_adapter_sha256"] = hashlib.sha256((folder / "adapter_model.safetensors").read_bytes()).hexdigest()
    (args.out / "evaluation_config.json").write_text(json.dumps(provenance, indent=2))
    records = []
    for mode in (["base", "lora"] if args.compare_base else ["lora"]):
        for row in rows.to_dict("records"):
            q, qraw, qerror = predict(row["path"], "quality", mode)
            # Only the predicted readable gate may invoke age inference; GT cannot bypass it.
            age, araw, aerror = (-1, None, None)
            if q == 0 and qerror is None:
                age, araw, aerror = predict(row["path"], "age", mode)
            gt = int(row["age4"]) if row["quality_gt"] == "readable" else 4
            record = {"model": mode, "scale_id": row["scale_id"], "fish_key": row["fish_key"],
                      "quality_gt": row["quality_gt"], "age_gt": gt if gt != 4 else -1,
                      "quality_prediction": q, "age_prediction": age, "pipeline_gt": gt,
                      "pipeline_prediction": 4 if q == 1 else age if q == 0 else -1,
                      "quality_raw": qraw, "age_raw": araw, "quality_error": qerror, "age_error": aerror,
                      "age_skipped": q != 0 or qerror is not None,
                      "image_sha256": hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()}
            records.append(record)
            with (args.out / "predictions.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            age_display = "skipped" if record["age_skipped"] else age
            print(f"{mode} {row['scale_id']}: gate={q}, age={age_display}, final={record['pipeline_prediction']}, GT={gt}", flush=True)
    summarize(records, args.out, age_gated=True)


if __name__ == "__main__":
    main()
