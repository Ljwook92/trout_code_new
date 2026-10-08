"""Bounded quality/age/review workflow using frozen LoRA tools and expert referral."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import pandas as pd

from evaluate_trout_vlm import evaluation_rows
from train_trout_vlm import MODEL, REVISION, transform_image

VERSION = "bounded_lora_review_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReviewWorkflow:
    """Explicit inspect/reinspect/stop policy. Tools never receive expert labels."""
    def __init__(self, tool, policy):
        self.tool, self.policy = tool, policy

    def run(self, path):
        traces = []
        actions = []
        def inspect(task, view, pixels, rotation=0, contrast=1):
            if len(traces) >= self.policy["max_calls"]:
                return None
            start = time.monotonic()
            response = self.tool.inspect(path, task, pixels, rotation, contrast)
            if type(response.get("prediction")) is not int or response["prediction"] not in (range(2) if task == "quality" else range(4)):
                raise ValueError("Tool returned an invalid class")
            if not math.isfinite(response.get("margin", float("nan"))) or not 0 <= response["margin"] <= 1:
                raise ValueError("Tool returned invalid ranking margin")
            trace = {"role": "quality" if task == "quality" else "age", "task": task, "view": view,
                     "pixels": pixels, "rotation": rotation, "contrast": contrast,
                     "seconds": time.monotonic() - start, **response}
            traces.append(trace)
            return trace

        def finish(prediction, reason):
            ages = [t for t in traces if t["task"] == "age"]
            qualities = [t for t in traces if t["task"] == "quality"]
            return {"prediction": prediction, "status": "expert_review" if prediction == -1 else
                    "bad" if prediction == 4 else "age_accepted", "reason": reason,
                    "first_age_prediction": ages[0]["prediction"] if ages else -1,
                    "first_quality_prediction": qualities[0]["prediction"] if qualities else -1,
                    "calls": len(traces), "review_actions": actions, "traces": traces}

        q = inspect("quality", "original", self.policy["quality_pixels"])
        qlimit = self.policy["quality_margin"]
        if q["margin"] < qlimit:
            actions.append({"role": "quality", "action": "request_full_frame_highres",
                            "reason": "low_quality_ranking_margin"})
            reviewed = inspect("quality", "full_frame_highres", self.policy["review_pixels"])
            if reviewed is None:
                return finish(-1, "quality_review_budget_exhausted")
            if reviewed["prediction"] != q["prediction"] or reviewed["margin"] < qlimit:
                return finish(-1, "quality_unresolved_or_disagreement")
            q = reviewed
        if q["prediction"] == 1:
            return finish(4, "bad_with_sufficient_ranking_margin")

        age = inspect("age", "original", self.policy["age_pixels"])
        if age is None:
            return finish(-1, "age_budget_exhausted")
        rotated = inspect("age", "rotation_check", self.policy["age_pixels"], rotation=90)
        if rotated is None:
            return finish(-1, "review_budget_exhausted")

        def consistent():
            ages = [t for t in traces if t["task"] == "age"]
            return len({t["prediction"] for t in ages}) == 1 and all(
                t["margin"] >= self.policy["age_margin"] for t in ages)

        if not consistent():
            actions.append({"role": "review", "action": "request_full_frame_highres",
                            "reason": "age_disagreement_or_low_margin"})
            inspect("age", "full_frame_highres", self.policy["review_pixels"])
            if not consistent():
                actions.append({"role": "review", "action": "request_contrast_check",
                                "reason": "age_still_unresolved"})
                inspect("age", "contrast_check", self.policy["review_pixels"], contrast=1.15)
        if not consistent():
            actions.append({"role": "review", "action": "refer_to_expert",
                            "reason": "no_evidence_to_override_disagreement"})
            return finish(-1, "age_unresolved_or_disagreement")
        return finish(age["prediction"], "consistent_views_not_a_guarantee_of_correctness")


class LoRATools:
    def __init__(self, quality_adapter, age_adapter, configs):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("Allocated CUDA GPU required")
        self.torch, self.configs, self.processors = torch, configs, {}
        self.processor_class = AutoProcessor
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=torch.float16, attn_implementation="eager").to("cuda:0")
        self.model = PeftModel.from_pretrained(base, str(quality_adapter), adapter_name="quality", is_trainable=False)
        self.model.load_adapter(str(age_adapter), adapter_name="age", is_trainable=False)
        self.model.config.use_cache = False

    def inspect(self, path, task, pixels, rotation=0, contrast=1):
        from PIL import Image, ImageOps
        torch = self.torch
        if pixels not in self.processors:
            self.processors[pixels] = self.processor_class.from_pretrained(
                MODEL, revision=REVISION, use_fast=False, min_pixels=56 * 56, max_pixels=pixels)
        processor, config = self.processors[pixels], self.configs[task]
        self.model.set_adapter(task)
        self.model.eval()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image = transform_image(image, angle=rotation, contrast=contrast)
        messages = [{"role": "system", "content": config["system"]}, {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": config["prompt"]}]}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_ids = processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"]
        n = prefix_ids.shape[1]
        candidates = ([{"decision": "readable"}, {"decision": "bad"}] if task == "quality" else
                      [{"prediction": i} for i in range(4)])
        scores = []
        # Rank fixed label strings; never fabricate expert reasoning or use self-reported confidence.
        for candidate in candidates:
            completed = processor.apply_chat_template(messages + [{"role": "assistant", "content": json.dumps(candidate)}],
                                                       tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[completed], images=[image], return_tensors="pt")
            if not torch.equal(inputs["input_ids"][:, :n], prefix_ids):
                raise ValueError("Response token prefix mismatch")
            inputs = inputs.to("cuda:0")
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False)
                logits = output.logits[:, n - 1:-1, :].float()
                targets = inputs["input_ids"][:, n:]
                score = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean().item()
            if not math.isfinite(score):
                raise RuntimeError("Non-finite response likelihood")
            scores.append(score)
            del inputs, output, logits, targets
        weights = torch.tensor(scores).softmax(0).tolist()
        order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        return {"prediction": order[0], "margin": weights[order[0]] - weights[order[1]],
                "ranking_weights": weights, "mean_response_log_likelihood": scores,
                "calibrated_probability": False}


def write_reports(records, out):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    rows = pd.DataFrame(records)
    accepted = rows.prediction.ge(0)
    readable = rows["gt"].lt(4)
    passed = rows.prediction.between(0, 3)
    truth, pred = rows["gt"].astype(int), rows.prediction.astype(int)
    recalls = [(pred[truth.eq(c)] == c).mean() for c in range(5) if truth.eq(c).any()]
    metrics = {"support": len(rows), "accuracy_abstention_as_wrong": accuracy_score(truth, pred),
               "balanced_accuracy": sum(recalls) / len(recalls),
               "macro_f1": f1_score(truth, pred, labels=list(range(5)), average="macro", zero_division=0),
               "expert_referral_rate": float((~accepted).mean()),
               "accepted_accuracy": float(pred[accepted].eq(truth[accepted]).mean()) if accepted.any() else None,
               "readable_age_coverage": float(passed[readable].mean()) if readable.any() else None,
               "bad_pass_rate": float(passed[~readable].mean()) if (~readable).any() else None,
               "mean_tool_calls": float(rows.calls.mean()), "mean_seconds": float(rows.seconds.mean()),
               "note": "Margins are uncalibrated. Accepted accuracy must be interpreted with referral/coverage."}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    names = ["0", "1", "2", "3 or older", "bad"]
    report = classification_report(truth, pred, labels=list(range(5)), target_names=names, digits=4, zero_division=0)
    (out / "pipeline_report.txt").write_text(report)
    cm = confusion_matrix(truth, pred, labels=[0, 1, 2, 3, 4, -1])[:5, :]
    pd.DataFrame(cm, index=names, columns=names + ["expert_referral"]).to_csv(out / "pipeline_confusion.csv")
    columns = ["scale_id", "fish_key", "path", "prediction", "status", "reason", "calls", "seconds"]
    # Expert queue intentionally hides GT to support independent review.
    rows.loc[~accepted, columns].to_csv(out / "expert_review.csv", index=False)
    rows.drop(columns=["traces", "review_actions"]).to_csv(out / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--quality-adapter", required=True, type=Path)
    parser.add_argument("--age-adapter", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--quality-margin", type=float, default=0.05)
    parser.add_argument("--age-margin", type=float, default=0.05)
    parser.add_argument("--review-pixels", type=int, default=1605632)
    parser.add_argument("--max-calls", type=int, default=6)
    parser.add_argument("--limit", type=int, default=20, help="0: all; default exploratory pilot")
    parser.add_argument("--frozen-policy", type=Path, help="Validation policy_settings.json required for test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.max_calls < 3 or args.review_pixels < 56 * 56:
        raise ValueError("Invalid limit/budget/pixel settings")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in [args.quality_margin, args.age_margin]):
        raise ValueError("Ranking margins must be finite 0..1")
    label_hash = sha(args.labels)
    configs, adapter_hashes = {}, {}
    for task, folder in [("quality", args.quality_adapter), ("age", args.age_adapter)]:
        cfg = json.loads((folder.parent / "training_config.json").read_text())
        if cfg["task"] != task or cfg["labels_sha256"] != label_hash or cfg["model"] != MODEL or cfg["revision"] != REVISION:
            raise ValueError("Adapter provenance mismatch")
        configs[task] = cfg
        adapter_hashes[task] = sha(folder / "adapter_model.safetensors")
    policy = {"protocol": VERSION, "quality_margin": args.quality_margin, "age_margin": args.age_margin,
              "quality_pixels": configs["quality"]["max_pixels"], "age_pixels": configs["age"]["max_pixels"],
              "review_pixels": args.review_pixels, "max_calls": args.max_calls,
              "labels_sha256": label_hash, "adapter_hashes": adapter_hashes,
              "scoring": "mean_response_token_log_likelihood_softmax_not_calibrated"}
    if args.review_pixels <= max(policy["quality_pixels"], policy["age_pixels"]):
        raise ValueError("Review pixel budget must exceed both baseline budgets")
    if args.split == "test" and not args.frozen_policy:
        raise ValueError("Freeze validation policy before test; supply --frozen-policy")
    if args.frozen_policy and json.loads(args.frozen_policy.read_text()) != policy:
        raise ValueError("Frozen policy differs; do not retune on test")
    rows = evaluation_rows(pd.read_csv(args.labels), args.split)
    if args.limit:
        rows = rows.sample(frac=1, random_state=100).head(args.limit)
    if not rows.path.map(lambda p: Path(str(p)).is_file()).all():
        raise FileNotFoundError("Missing input images")
    print(f"split={args.split} images={len(rows)}; bounded workflow; uncalibrated margins", flush=True)
    if args.dry_run:
        print("Dry run passed; no model loaded")
        return
    if args.out.exists():
        raise FileExistsError("Choose a new output directory")
    tools = LoRATools(args.quality_adapter, args.age_adapter, configs)
    workflow = ReviewWorkflow(tools, policy)
    args.out.mkdir(parents=True)
    (args.out / "policy_settings.json").write_text(json.dumps(policy, indent=2))
    (args.out / "run_config.json").write_text(json.dumps({"split": args.split, "limit": args.limit,
                                                          "training_configs": configs}, indent=2))
    records = []
    for row in rows.to_dict("records"):
        start = time.monotonic()
        # Only the anonymous image path enters the workflow; GT is attached after completion.
        result = workflow.run(row["path"])
        record = {**result, "scale_id": row["scale_id"], "fish_key": row["fish_key"], "path": row["path"],
                  "gt": int(row["age4"]) if row["quality_gt"] == "readable" else 4,
                  "seconds": time.monotonic() - start, "image_sha256": sha(row["path"])}
        records.append(record)
        with (args.out / "predictions.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"{row['scale_id']}: pred={result['prediction']} status={result['status']} calls={result['calls']}", flush=True)
    write_reports(records, args.out)


if __name__ == "__main__":
    main()
