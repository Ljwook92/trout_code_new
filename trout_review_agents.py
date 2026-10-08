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

VERSION = "bounded_lora_review_v3_feature_memory"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReviewWorkflow:
    """Explicit inspect/reinspect/stop policy. Tools never receive expert labels."""
    def __init__(self, tool, policy):
        self.tool, self.policy = tool, policy

    def run(self, path):
        traces = []
        actions = []
        def inspect(task, view, pixels, rotation=0, contrast=1, feedback_context=None):
            if len(traces) >= self.policy["max_calls"]:
                return None
            start = time.monotonic()
            if feedback_context is None:
                response = self.tool.inspect(path, task, pixels, rotation, contrast)
            else:
                response = self.tool.inspect(path, task, pixels, rotation, contrast,
                                             feedback_context=feedback_context)
            if type(response.get("prediction")) is not int or response["prediction"] not in (range(2) if task == "quality" else range(4)):
                raise ValueError("Tool returned an invalid class")
            if not math.isfinite(response.get("margin", float("nan"))) or not 0 <= response["margin"] <= 1:
                raise ValueError("Tool returned invalid ranking margin")
            trace = {"role": "quality" if task == "quality" else "age", "task": task, "view": view,
                     "pixels": pixels, "rotation": rotation, "contrast": contrast,
                     "seconds": time.monotonic() - start, **response}
            if feedback_context is not None:
                trace["feedback_context"] = feedback_context
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

        def reconsider(task):
            threshold = self.policy[task + "_margin"]
            previous = None
            for round_index in range(self.policy.get("feedback_rounds", 0)):
                if len(traces) >= self.policy["max_calls"]:
                    break
                history = [{key: t[key] for key in ["view", "prediction", "margin", "pixels",
                            "rotation", "contrast", "ranking_weights"] if key in t}
                           for t in traces if t["task"] == task]
                classes = sorted({h["prediction"] for h in history})
                question = (
                    "Reinspect physical/optical usability, not age. Distinguish visible scale "
                    "damage from absence of an identified annual ring."
                    if task == "quality" else
                    "Reinspect completed annual annuli across the full scale. Distinguish annual "
                    "annuli from fine circuli; do not infer annulus counts from previous age labels."
                )
                context = {"round": round_index + 1, "observed_classes": classes,
                           "previous_checks": history, "review_question": question,
                           "instruction": "All previous predictions and scores are fallible, not expert GT. "
                           "Return to the image and decide again using both unmodified and contrast views. "
                           "Do not copy the most frequent or latest answer merely to reach agreement. "
                           "No new visual evidence has been annotated by an expert."}
                actions.append({"role": "review", "action": "return_to_task_with_feedback",
                                "task": task, "round": round_index + 1, "review_question": question,
                                "observed_classes": classes})
                # Reinspection is an actual context-conditioned model call, not a vote over logs.
                current = inspect(task, f"feedback_round_{round_index + 1}", self.policy["review_pixels"],
                                  contrast=1.15 if round_index % 2 == 0 else 0.9,
                                  feedback_context=context)
                if (previous is not None and current["prediction"] == previous["prediction"]
                        and min(current["margin"], previous["margin"]) >= threshold):
                    actions.append({"role": "review", "action": "accept_feedback_stabilization",
                                    "task": task, "prediction": current["prediction"],
                                    "note": "Two context-conditioned checks agree; visual basis is not expert-verified."})
                    return current
                previous = current
            actions.append({"role": "review", "action": "refer_to_expert", "task": task,
                            "reason": "feedback_did_not_stabilize_within_budget"})
            return None

        q = inspect("quality", "original", self.policy["quality_pixels"])
        qlimit = self.policy["quality_margin"]
        if q["margin"] < qlimit:
            actions.append({"role": "quality", "action": "request_full_frame_highres",
                            "reason": "low_quality_ranking_margin"})
            reviewed = inspect("quality", "full_frame_highres", self.policy["review_pixels"])
            if reviewed is None:
                return finish(-1, "quality_review_budget_exhausted")
            if reviewed["prediction"] != q["prediction"] or reviewed["margin"] < qlimit:
                resolved = reconsider("quality")
                if resolved is None:
                    return finish(-1, "quality_unresolved_or_disagreement")
                q = resolved
            else:
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
            resolved = reconsider("age")
            if resolved is not None:
                return finish(resolved["prediction"], "feedback_stabilized_not_expert_verified")
            actions.append({"role": "review", "action": "refer_to_expert",
                            "reason": "no_evidence_to_override_disagreement"})
            return finish(-1, "age_unresolved_or_disagreement")
        return finish(age["prediction"], "consistent_views_not_a_guarantee_of_correctness")


class LoRATools:
    def __init__(self, quality_adapter, age_adapter, configs, memory=None, memory_top_k=2):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("Allocated CUDA GPU required")
        self.torch, self.configs, self.processors = torch, configs, {}
        self.memory, self.memory_top_k, self.embedding_cache = memory, memory_top_k, {}
        self.processor_class = AutoProcessor
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=torch.float16, attn_implementation="eager").to("cuda:0")
        self.model = PeftModel.from_pretrained(base, str(quality_adapter), adapter_name="quality", is_trainable=False)
        self.model.load_adapter(str(age_adapter), adapter_name="age", is_trainable=False)
        self.model.config.use_cache = False

    def processor(self, pixels):
        if pixels not in self.processors:
            self.processors[pixels] = self.processor_class.from_pretrained(
                MODEL, revision=REVISION, use_fast=False, min_pixels=56 * 56, max_pixels=pixels)
        return self.processors[pixels]

    def embed(self, path, pixels):
        from PIL import Image, ImageOps
        torch, processor = self.torch, self.processor(pixels)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Inspect scale."}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
        self.model.eval()
        with torch.inference_mode():
            features = self.model.get_base_model().get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
            vector = features[0].float().mean(0)
            norm = vector.norm()
            if not torch.isfinite(vector).all() or norm.item() <= 0:
                raise RuntimeError("Invalid Qwen visual feature")
            result = (vector / norm).cpu().numpy()
        return result

    def inspect(self, path, task, pixels, rotation=0, contrast=1, feedback_context=None):
        from PIL import Image, ImageOps
        torch = self.torch
        processor, config = self.processor(pixels), self.configs[task]
        self.model.set_adapter(task)
        self.model.eval()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        images, content = inspection_content(image, config["prompt"], rotation, contrast, feedback_context)
        hits = []
        if feedback_context is not None and self.memory is not None:
            key = str(Path(path).resolve())
            if key not in self.embedding_cache:
                # Embedding is computed at the identical budget used to build the index.
                self.embedding_cache.clear()
                self.embedding_cache[key] = self.embed(path, self.memory.meta["pixels"])
            hits = self.memory.search(self.embedding_cache[key], task, self.memory_top_k,
                                      exclude_fish=self.memory.path_to_fish.get(key))
            reference_content = []
            for index, hit in enumerate(hits):
                with Image.open(hit["path"]) as source:
                    reference = ImageOps.exif_transpose(source).convert("RGB")
                scale = min(1, (self.memory.meta["pixels"] / (reference.width * reference.height)) ** .5)
                reference.thumbnail((max(1, int(reference.width * scale)), max(1, int(reference.height * scale))))
                images.append(reference)
                label = ("readable" if hit["class"] == 0 else "bad") if task == "quality" else str(hit["class"])
                reference_content.extend([{"type": "text", "text": f"Training reference {index + 1}: {task} "
                    f"label {label}; source {hit['label_source']}. Similarity is not proof of the target class."},
                    {"type": "image"}])
            # Image placeholder order matches [target original, target enhanced, references...].
            content = content[:-1] + reference_content + [{"type": "text", "text": content[-1]["text"] +
                "\nReferences are labeled training cases, not annotations of this target. "
                "Use visible similarities AND differences; do not blindly copy reference labels."}]
        messages = [{"role": "system", "content": config["system"]}, {"role": "user", "content": content}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_ids = processor(text=[prefix], images=images, return_tensors="pt")["input_ids"]
        n = prefix_ids.shape[1]
        candidates = ([{"decision": "readable"}, {"decision": "bad"}] if task == "quality" else
                      [{"prediction": i} for i in range(4)])
        scores = []
        # Rank fixed label strings; never fabricate expert reasoning or use self-reported confidence.
        for candidate in candidates:
            completed = processor.apply_chat_template(messages + [{"role": "assistant", "content": json.dumps(candidate)}],
                                                       tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[completed], images=images, return_tensors="pt")
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
                "calibrated_probability": False, "retrieved_training_references": hits}


def inspection_content(image, prompt, rotation=0, contrast=1, feedback_context=None):
    transformed = transform_image(image, angle=rotation, contrast=contrast)
    if feedback_context is None:
        return [transformed], [{"type": "image"}, {"type": "text", "text": prompt}]
    # Keep the original visible alongside enhancement; do not let contrast changes replace evidence.
    content = [{"type": "text", "text": "Original full-frame scale (no expert annotations)."},
               {"type": "image"},
               {"type": "text", "text": f"Same full scale, contrast factor {contrast}; not independent evidence."},
               {"type": "image"},
               {"type": "text", "text": "Reviewer feedback:\n" + json.dumps(feedback_context) + "\n" + prompt}]
    return [image, transformed], content


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
               "feedback_used_count": sum(any("feedback_context" in t for t in r["traces"]) for r in records),
               "feedback_accepted_count": sum(r["prediction"] >= 0 and any(
                   a["action"] == "accept_feedback_stabilization" for a in r["review_actions"]) for r in records),
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
    parser.add_argument("--max-calls", type=int, default=10)
    parser.add_argument("--feedback-rounds", type=int, default=2, help="0 disables; 2..4 bounded reinspection rounds per task")
    parser.add_argument("--feature-memory", type=Path, help="Train-only visual memory used during feedback reinspection")
    parser.add_argument("--memory-top-k", type=int, default=2, help="Different-fish references per reinspection, 1..2")
    parser.add_argument("--limit", type=int, default=20, help="0: all; default exploratory pilot")
    parser.add_argument("--frozen-policy", type=Path, help="Validation policy_settings.json required for test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.max_calls < 3 or args.review_pixels < 56 * 56:
        raise ValueError("Invalid limit/budget/pixel settings")
    if args.feedback_rounds not in [0, 2, 3, 4]:
        raise ValueError("Feedback rounds must be 0 or 2..4")
    if args.memory_top_k not in [1, 2] or (args.feature_memory and args.feedback_rounds == 0):
        raise ValueError("Memory needs feedback and top-k 1 or 2")
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
              "review_pixels": args.review_pixels, "max_calls": args.max_calls, "feedback_rounds": args.feedback_rounds,
              "labels_sha256": label_hash, "adapter_hashes": adapter_hashes,
              "scoring": "mean_response_token_log_likelihood_softmax_not_calibrated"}
    memory = None
    if args.feature_memory:
        from trout_feature_memory import FeatureMemory
        memory = FeatureMemory(args.feature_memory, args.labels, adapter_hashes)
    policy["memory"] = None if memory is None else {"config_sha256": sha(args.feature_memory / "memory_config.json"),
                                                   "top_k": args.memory_top_k}
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
    print(f"split={args.split} images={len(rows)}; feedback_rounds={args.feedback_rounds}; "
          "bounded workflow; uncalibrated margins", flush=True)
    if args.dry_run:
        print("Dry run passed; no model loaded")
        return
    if args.out.exists():
        raise FileExistsError("Choose a new output directory")
    tools = LoRATools(args.quality_adapter, args.age_adapter, configs, memory, args.memory_top_k)
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
        print(f"{row['scale_id']}: pred={result['prediction']} status={result['status']} "
              f"calls={result['calls']} reason={result['reason']}", flush=True)
    write_reports(records, args.out)


if __name__ == "__main__":
    main()
