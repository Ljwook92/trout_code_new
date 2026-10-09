"""Bounded quality/age/review agents using trained SimCLR-feature Qwen tools.

No online weight updates, no GT in decisions, no invented visual explanations.
"""
import argparse
import base64
from collections import Counter
import html
import io
import json
import math
from pathlib import Path
import time

import pandas as pd
import torch
from PIL import Image, ImageOps

from evaluate_trout_feature_qwen import inspect_bundle, prompt_tokens, greedy_response
from evaluate_trout_vlm import decision, evaluation_rows
from train_trout_feature_qwen import (
    MODEL, REVISION, FeatureProjector, FrozenScaleEncoder, answer_tokens, feature_batch, sha,
)
from train_trout_vlm import transform_image

PROTOCOL = "bounded_simclr_feature_agents_v1"
NAMES = ["0", "1", "2", "3 or older", "bad"]


class ReviewAgent:
    """Choose additional full-frame inspections; agreement is not expert truth."""
    def resolve(self, traces, margin):
        reliable = [t for t in traces if t["prediction"] >= 0 and t.get("error") is None
                    and t.get("margin", 0) >= margin
                    and t["prediction"] == t.get("ranking_prediction")]
        if len({t["prediction"] for t in reliable}) > 1:
            return None, "strong_view_disagreement"
        if len(traces) >= 2 and all(t in reliable for t in traces[-2:]):
            return traces[-1]["prediction"], "two_consistent_views_uncalibrated_not_expert_confirmation"
        return None, "insufficient_consistent_evidence"


class StageAgent:
    def __init__(self, task, threshold, review):
        self.task, self.threshold, self.review = task, threshold, review

    def inspect(self, tools, image, policy, all_traces):
        traces = []
        views = ["original", "rotation", "contrast"]
        if policy["highres_size"]:
            views.append("full_frame_highres")
        for view in views:
            if len(all_traces) >= policy["max_calls"]:
                return None, "inspection_budget_exhausted"
            trace = tools.inspect(image, self.task, view, policy)
            trace.update({"task": self.task, "view": view})
            traces.append(trace)
            all_traces.append(trace)
            prediction, reason = self.review.resolve(traces, self.threshold)
            if prediction is not None:
                return prediction, reason
        return None, reason


class QualityAgent(StageAgent):
    def __init__(self, policy, review):
        super().__init__("quality", policy["quality_margin"], review)


class AgeAgent(StageAgent):
    def __init__(self, policy, review):
        super().__init__("age", policy["age_margin"], review)


def run_workflow(tools, image, policy):
    traces, review = [], ReviewAgent()
    quality, reason = QualityAgent(policy, review).inspect(tools, image, policy, traces)
    if quality is None:
        return {"prediction": -1, "status": "expert_review", "reason": "quality_" + reason,
                "quality_prediction": -1, "age_prediction": -1, "traces": traces}
    if quality == 1:
        return {"prediction": 4, "status": "bad", "reason": reason,
                "quality_prediction": 1, "age_prediction": -1, "traces": traces}
    age, reason = AgeAgent(policy, review).inspect(tools, image, policy, traces)
    return {"prediction": age if age is not None else -1,
            "status": "age_accepted" if age is not None else "expert_review",
            "reason": reason if age is not None else "age_" + reason,
            "quality_prediction": 0, "age_prediction": age if age is not None else -1,
            "traces": traces}


class FeatureTools:
    def __init__(self, runs, configs):
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel
        from torchvision import transforms as T
        if not torch.cuda.is_available():
            raise RuntimeError("An allocated CUDA GPU is required")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, dtype=self.dtype, attn_implementation="sdpa")
        self.model = PeftModel.from_pretrained(base, str(runs["quality"] / "best_bundle/qwen_adapter"),
                                             adapter_name="quality", is_trainable=False)
        self.model.load_adapter(str(runs["age"] / "best_bundle/qwen_adapter"),
                                adapter_name="age", is_trainable=False)
        self.model.to("cuda").eval()
        base.model.visual.to("cpu")
        self.model.requires_grad_(False)
        hidden = self.model.get_input_embeddings().weight.shape[1]
        self.encoders, self.projectors, self.tokenizers, self.prompts, self.candidates = {}, {}, {}, {}, {}
        for task, run in runs.items():
            cfg = configs[task]
            if cfg["hidden_size"] != hidden:
                raise ValueError("Qwen/projector dimensions differ")
            state = torch.load(run / "encoder.pt", map_location="cpu", weights_only=True)
            self.encoders[task] = FrozenScaleEncoder(state["backbone_state_dict"], cfg["grid"]).to("cuda").eval()
            projector = FeatureProjector(hidden, cfg["grid"])
            projector.load_state_dict(torch.load(run / "best_bundle/projector.pt", map_location="cpu", weights_only=True))
            self.projectors[task] = projector.to("cuda").eval().requires_grad_(False)
            tok = AutoTokenizer.from_pretrained(run / "best_bundle/qwen_adapter")
            self.tokenizers[task] = tok
            self.prompts[task] = prompt_tokens(tok, task)
            answers = ([json.dumps({"decision": label}) for label in ["readable", "bad"]]
                       if task == "quality" else [json.dumps({"prediction": label}) for label in range(4)])
            self.candidates[task] = [answer_tokens(tok, task, answer) for answer in answers]
        self.transforms = {size: T.Compose([T.Resize((size, size)), T.ToTensor(),
                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                           for size in [224, 448]}

    @torch.inference_mode()
    def inspect(self, image, task, view, policy):
        size = 224
        if view == "rotation":
            image = image.transpose(Image.Transpose.ROTATE_90)
        elif view == "contrast":
            image = transform_image(image, contrast=policy["contrast"])
        elif view == "full_frame_highres":
            size = policy["highres_size"]
        self.model.set_adapter(task)
        self.model.eval()
        # Preserve training's float32 frozen ResNet computation.
        features = self.encoders[task](self.transforms[size](image).unsqueeze(0).to("cuda"))
        projector, tokenizer = self.projectors[task], self.tokenizers[task]
        with torch.autocast("cuda", dtype=self.dtype):
            raw, error = greedy_response(self.model, projector, features, self.prompts[task], tokenizer)
            scores = [-float(self.model(**feature_batch(self.model, projector, features, tokens)).loss)
                      for tokens in self.candidates[task]]
        if not all(math.isfinite(s) for s in scores):
            raise RuntimeError("Non-finite candidate scores")
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        pred = -1
        if error is None:
            try:
                pred = decision(raw, task)
            except ValueError as exc:
                error = str(exc)
        return {"prediction": pred, "raw": raw, "error": error,
                "ranking_prediction": ranked[0], "scores": scores,
                "margin": scores[ranked[0]] - scores[ranked[1]],
                "input_size": size, "outside_training_resolution": size != 224,
                "candidate_forward_calls": len(scores)}


def metrics(records):
    from sklearn.metrics import f1_score
    labeled = [r for r in records if "gt" in r]
    result = {"support": len(records),
              "expert_referral_rate": sum(r["status"] == "expert_review" for r in records) / len(records),
              "mean_inspections": sum(len(r["traces"]) for r in records) / len(records),
              "mean_seconds": sum(r["seconds"] for r in records) / len(records),
              "referral_reasons": dict(Counter(r["reason"] for r in records if r["status"] == "expert_review")),
              "note": "Ranking margins are uncalibrated, not probabilities. Agreement does not prove correctness. "
                      "Inspection budget is not a forward-pass budget. No online learning or generated reasoning."}
    if labeled:
        truth = pd.Series([r["gt"] for r in labeled])
        pred = pd.Series([r["prediction"] for r in labeled])
        accepted = pred.ge(0)
        readable, bad = truth.lt(4), truth.eq(4)
        passed = pd.Series([r["quality_prediction"] == 0 for r in labeled])
        result.update({"labeled_support": len(labeled), "accuracy_abstention_as_wrong": float(pred.eq(truth).mean()),
            "balanced_accuracy": float(sum(pred[truth.eq(c)].eq(c).mean() for c in range(5) if truth.eq(c).any())
                                      / truth.nunique()),
            "macro_f1": float(f1_score(truth, pred, labels=list(range(5)), average="macro", zero_division=0)),
            "accepted_accuracy": float(pred[accepted].eq(truth[accepted]).mean()) if accepted.any() else None,
            "readable_gate_coverage": float(passed[readable].mean()) if readable.any() else None,
            "readable_age_coverage": float(pred[readable].between(0, 3).mean()) if readable.any() else None,
            "bad_pass_rate": float(passed[bad].mean()) if bad.any() else None})
    return result


def export_review(records, out):
    queued = [r for r in records if r["status"] == "expert_review"]
    columns = ["scale_id", "fish_key", "path", "image_sha256", "quality_gt", "expert_age", "reviewer", "reviewed_at"]
    rows, sections = [], []
    for r in queued:
        rows.append({**{key: r[key] for key in columns[:4]}, **{key: "" for key in columns[4:]}})
        with Image.open(r["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1200, 1200))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=90)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        # No GT is shown. Predictions are collapsed to reduce anchoring in expert review.
        info = {key: r[key] for key in ["reason", "quality_prediction", "age_prediction", "traces"]}
        sections.append(f'<section><h2>{html.escape(r["scale_id"])}</h2><img alt="Scale for expert review" '
                        f'src="data:image/jpeg;base64,{data}"><details><summary>Model inspection record</summary>'
                        f'<pre>{html.escape(json.dumps(info, indent=2))}</pre></details></section>')
    pd.DataFrame(rows, columns=columns).to_csv(out / "expert_annotations.csv", index=False)
    with (out / "expert_queue.jsonl").open("w") as stream:
        for r in queued:
            stream.write(json.dumps({key: value for key, value in r.items() if key != "gt"}) + "\n")
    # Full-resolution paths/hashes remain in CSV; HTML images are preview thumbnails.
    (out / "expert_review.html").write_text('<!doctype html><meta charset="utf-8"><title>Trout expert review</title>'
        '<style>body{font-family:Arial;max-width:1100px;margin:24px auto}img{max-width:100%;height:auto}'
        'section{border-bottom:1px solid #bbb;padding:20px 0}pre{white-space:pre-wrap}</style>'
        '<h1>Expert Review</h1><p>Preview images; use original paths for full-resolution inspection. '
        'Annotations belong in expert_annotations.csv. Existing GT is hidden.</p>' + ''.join(sections))


def apply_feedback(table, annotations):
    """Only human-confirmed TRAIN annotations may change a new training table."""
    table = table.copy()
    table_fields = {"scale_id", "fish_key", "path", "split", "label", "expert_age", "quality_gt",
                    "quality_source", "age_label", "age_label_source", "age4"}
    if (not table_fields <= set(table) or table.scale_id.isna().any() or table.scale_id.duplicated().any()
            or table.fish_key.isna().any() or table.groupby("fish_key")["split"].nunique().gt(1).any()):
        raise ValueError("Invalid baseline supervised table or fish partitions")
    required = {"scale_id", "fish_key", "path", "image_sha256", "quality_gt", "expert_age", "reviewer", "reviewed_at"}
    if not required <= set(annotations) or annotations.scale_id.isna().any() or annotations.scale_id.duplicated().any():
        raise ValueError("Invalid expert annotation template")
    incomplete = annotations.quality_gt.isna() & annotations[["expert_age", "reviewer", "reviewed_at"]].notna().any(axis=1)
    if incomplete.any():
        raise ValueError("A partial expert annotation requires quality_gt before import")
    completed = annotations.loc[annotations.quality_gt.notna()].copy()
    if completed.empty:
        raise ValueError("No completed expert quality annotations")
    known = table.set_index("scale_id")
    affected = set()
    fields = ["label", "expert_age", "quality_gt", "quality_source", "age_label", "age_label_source", "age4"]
    for a in completed.to_dict("records"):
        if a["scale_id"] not in known.index:
            raise ValueError("Unknown scale_id in feedback")
        old = known.loc[a["scale_id"]]
        if old["split"] != "train":
            raise ValueError("Validation/test/unassigned feedback cannot enter training")
        if (a["fish_key"] != old.fish_key or a["path"] != old.path
                or sha(old.path) != a["image_sha256"]):
            raise ValueError("Feedback image/fish identity changed")
        if any(pd.isna(a[key]) or not str(a[key]).strip() for key in ["reviewer", "reviewed_at"]):
            raise ValueError("Expert reviewer and reviewed_at are required")
        if a["quality_gt"] not in ["readable", "bad"]:
            raise ValueError("Quality must be human-confirmed readable or bad")
        age = a["expert_age"]
        if pd.notna(age):
            age = float(age)
            if not math.isfinite(age) or age not in range(6) or a["quality_gt"] != "readable":
                raise ValueError("Expert age must be 0..5 for readable, blank for bad")
        mask = table.scale_id.eq(a["scale_id"])
        for field in fields:
            backup = "pre_review_" + field
            if backup not in table:
                table[backup] = table[field]
        table.loc[mask, "quality_gt"] = a["quality_gt"]
        table.loc[mask, "quality_source"] = "expert_quality_csv"
        for key in ["reviewer", "reviewed_at"]:
            table.loc[mask, key] = str(a[key])
        table.loc[mask, "feedback_image_sha256"] = a["image_sha256"]
        if a["quality_gt"] == "bad":
            table.loc[mask, "label"] = 6
            table.loc[mask, ["expert_age", "age_label", "age4"]] = float("nan")
            table.loc[mask, "age_label_source"] = "unlabeled"
        elif pd.notna(age):
            table.loc[mask, ["label", "expert_age", "age_label"]] = int(age)
            table.loc[mask, "age4"] = min(int(age), 3)
            table.loc[mask, "age_label_source"] = "expert_direct"
        elif old.quality_gt != "readable":
            table.loc[mask, ["label", "expert_age", "age_label", "age4"]] = float("nan")
            table.loc[mask, "age_label_source"] = "unlabeled"
        affected.add(old.fish_key)
    inherited = table.fish_key.isin(affected) & table.age_label_source.eq("fish_propagated")
    table.loc[inherited, ["age_label", "age4"]] = float("nan")
    table.loc[inherited, "age_label_source"] = "unlabeled"
    conflicts = table.groupby("fish_key").expert_age.nunique().gt(1)
    table["fish_age_conflict"] = table.fish_key.map(conflicts).fillna(False)
    table["age_train_eligible"] = (table["split"].eq("train") & table.quality_gt.eq("readable")
                                   & table.age4.notna() & ~table.fish_age_conflict)
    table["quality_train_eligible"] = table["split"].eq("train") & table.quality_gt.isin(["readable", "bad"])
    table["age_evaluation_eligible"] = (table["split"].isin(["validation", "test"])
                                        & table.quality_gt.eq("readable") & table.age_label_source.eq("expert_direct"))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    feedback = sub.add_parser("apply-feedback", help="Import real human annotations into a NEW train table")
    feedback.add_argument("--labels", type=Path, required=True)
    feedback.add_argument("--annotations", type=Path, required=True)
    feedback.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--labels", type=Path, required=True, help="Original table for bundle provenance/audit only")
    run.add_argument("--quality-run", type=Path, required=True)
    run.add_argument("--age-run", type=Path, required=True)
    run.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    run.add_argument("--input-csv", type=Path, help="New images: scale_id, fish_key, path; no GT needed")
    run.add_argument("--frozen-policy", type=Path, help="Required for test, saved from validation")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--quality-margin", type=float, default=0.05)
    run.add_argument("--age-margin", type=float, default=0.05)
    run.add_argument("--max-calls", type=int, default=8, help="Max tool inspections, not transformer forwards")
    run.add_argument("--contrast", type=float, default=1.2)
    run.add_argument("--highres-size", type=int, choices=[0, 448], default=0, help="Optional out-of-training-resolution probe")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a new output path; existing artifacts are preserved")
    table = pd.read_csv(args.labels)
    if args.command == "apply-feedback":
        # Validate baseline identity/partitions before any human annotation is applied.
        evaluation_rows(table, "validation")
        result = apply_feedback(table, pd.read_csv(args.annotations))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print("Saved new expert-reviewed training table:", args.out)
        print("No weights updated. Retrain with this table into a NEW model directory.")
        return
    if (args.limit < 0 or not 2 <= args.max_calls <= 8
            or not all(math.isfinite(x) and x >= 0 for x in [args.quality_margin, args.age_margin])
            or not math.isfinite(args.contrast) or args.contrast <= 0):
        raise ValueError("Invalid review policy")
    rows = (pd.read_csv(args.input_csv) if args.input_csv else evaluation_rows(table, args.split))
    if (not {"scale_id", "fish_key", "path"} <= set(rows) or rows.empty
            or rows[["scale_id", "fish_key", "path"]].isna().any().any() or rows.scale_id.duplicated().any()):
        raise ValueError("Need unique image IDs, fish IDs and image paths")
    if not args.input_csv and ("quality_source" not in rows or not rows.quality_source.isin(
            ["expert_direct", "expert_quality_csv"]).all()):
        raise ValueError("Audit requires direct expert quality GT")
    if args.limit:
        rows = rows.sample(frac=1, random_state=100).head(args.limit).reset_index(drop=True)
    if not rows.path.map(lambda p: Path(str(p)).is_file()).all():
        raise FileNotFoundError("Missing inference images")
    configs, hashes = {}, {}
    runs = {"quality": args.quality_run, "age": args.age_run}
    for task, folder in runs.items():
        configs[task], hashes[task] = inspect_bundle(folder, task, table, sha(args.labels))
    policy = {"protocol": PROTOCOL, "quality_margin": args.quality_margin, "age_margin": args.age_margin,
              "max_calls": args.max_calls, "contrast": args.contrast, "highres_size": args.highres_size,
              "labels_sha256": sha(args.labels), "bundle_hashes": hashes,
              "workflow_code_sha256": sha(Path(__file__)),
              "scoring": "mean_response_token_log_likelihood_gap_uncalibrated",
              "decision_rule": "two_consecutive_reliable_views_without_strong_view_disagreement"}
    if args.split == "test" and not args.input_csv and not args.frozen_policy:
        raise ValueError("Test requires --frozen-policy from a completed validation run")
    if args.frozen_policy and json.loads(args.frozen_policy.read_text()) != policy:
        raise ValueError("Frozen policy or model hashes differ")
    print(f"images={len(rows)} max_inspections={args.max_calls}; no GT decisions or online updates", flush=True)
    if args.dry_run:
        print("Dry run passed; no Qwen loaded, predictions generated, or output written")
        return
    tools = FeatureTools(runs, configs)
    args.out.mkdir(parents=True)
    (args.out / "policy.json").write_text(json.dumps(policy, indent=2))
    (args.out / "run_config.json").write_text(json.dumps({"split": args.split if not args.input_csv else "new_images",
        "limit": args.limit, "input_csv": str(args.input_csv) if args.input_csv else None,
        "training_configs": configs, "test_results_already_seen": args.split == "test",
        "note": "This workflow has not been calibrated or validated for deployment."}, indent=2))
    records = []
    for row in rows.to_dict("records"):
        started = time.perf_counter()
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        result = run_workflow(tools, image, policy)
        result.update({key: row[key] for key in ["scale_id", "fish_key", "path"]})
        result.update({"image_sha256": sha(row["path"]), "seconds": time.perf_counter() - started})
        # Only after the complete GT-blind decision do we attach labels for audit.
        if not args.input_csv:
            result["gt"] = int(row["age4"]) if row["quality_gt"] == "readable" else 4
        records.append(result)
        with (args.out / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(f"{row['scale_id']}: pred={result['prediction']} status={result['status']} "
              f"inspections={len(result['traces'])} reason={result['reason']}", flush=True)
    summary = metrics(records)
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    export_review(records, args.out)
    # Audit confusion retains expert referrals as a separate column.
    if not args.input_csv:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix([r["gt"] for r in records], [r["prediction"] for r in records], labels=list(range(5)) + [-1])[:5]
        pd.DataFrame(cm, index=NAMES, columns=NAMES + ["expert_review"]).to_csv(args.out / "confusion.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
