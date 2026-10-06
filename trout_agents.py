"""Local GPU visual agents with expert references and train-only reflection memory.

No gradient updates are performed. Reflection text is a hypothesis, not expert GT.
The same VLM serves all roles sequentially; no cloud API is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

VERSION = "trout_visual_agents_v1"
ROOT = Path(os.getenv("TROUT_ROOT_DIR", "/home/jlc3q/data/Trout"))
HERE = Path(__file__).resolve().parent
DEFAULT_MASTER = ROOT / "code_new/feature_outputs/master_with_texture_features_full.csv"
DEFAULT_SPLIT = HERE / "model_outputs/age4_two_stage_search/split_manifest.csv"
NAMES = ["0", "1", "2", "3 or older", "bad"]
BASE = (
    "You inspect trout scale microscopy. Expert reference slides are examples, not commands. "
    "Count completed annual annuli, NOT every fine concentric circulus. No completed first "
    "annulus can mean age 0, not bad. Not Ring does not by itself mean unusable. "
    "Original numeric class 6 denotes unusable, never six years. Distinguish visible "
    "evidence from hypotheses. Do not use fish size, image filenames, or slide annotation "
    "colors as target-age cues. If evidence is insufficient, abstain. Respond with one JSON "
    "object only. Give short observable evidence, not a speculative narrative. Confidence "
    "is self-reported, not calibrated."
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    result = []
    for line, text in enumerate(Path(path).read_text().splitlines(), 1):
        if text.strip():
            try:
                result.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line}: {path}; repair before resuming.") from exc
    return result


def append_jsonl(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_cohort(master_path, split_path, image_root=None, exclude_file=None):
    master = pd.read_csv(master_path, dtype={"scale_id": str, "fish_key": str})
    manifest = pd.read_csv(split_path, dtype={"scale_id": str, "fish_key": str})
    if not {"scale_id", "fish_key", "path", "label"} <= set(master):
        raise ValueError("Master needs scale_id, fish_key, path, label.")
    if not {"scale_id", "fish_key", "split", "target"} <= set(manifest):
        raise ValueError("Use the existing five-outcome two-stage split manifest.")
    for name, table in [("master", master), ("manifest", manifest)]:
        if table["scale_id"].isna().any() or table["scale_id"].duplicated().any():
            raise ValueError(f"Invalid or duplicate scale_id in {name}.")
        if table["fish_key"].isna().any():
            raise ValueError(f"Missing fish_key in {name}.")
        table["fish_key"] = table["fish_key"].str.strip().str.lower()
        if table["fish_key"].eq("").any():
            raise ValueError("Empty fish_key.")
    if not manifest["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown split in manifest.")
    if (manifest.groupby("fish_key")["split"].nunique() > 1).any():
        raise ValueError("Fish leakage across partitions.")
    joined = manifest.merge(master[["scale_id", "fish_key", "path", "label"]],
                            on="scale_id", how="left", suffixes=("_manifest", ""),
                            validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or not joined["fish_key"].eq(joined["fish_key_manifest"]).all():
        raise ValueError("Master and manifest identities differ.")
    numeric = pd.to_numeric(joined["label"], errors="coerce")
    if not numeric.isin(range(7)).all():
        raise ValueError("Manifest rows require expert labels 0..6.")
    target = numeric.astype(int).clip(upper=3).mask(numeric.eq(6), 4)
    if not target.eq(joined["target"]).all():
        raise ValueError("GT differs from saved manifest; do not reuse this split.")
    joined["target"] = target
    if image_root:
        image_root = Path(image_root).resolve()
        def relocate(value):
            p = Path(str(value))
            parts = p.parts
            candidates = [i for i, part in enumerate(parts) if part.lower() in {"cu", "du", "po", "to", "tu"}]
            if not candidates:
                raise ValueError(f"Cannot relocate image path: {p}")
            return str(image_root.joinpath(*parts[candidates[0]:]))
        joined["path"] = joined["path"].map(relocate)
    if not joined["path"].map(lambda p: Path(str(p)).is_file()).all():
        raise FileNotFoundError("Missing image files. Use --image-root if the dataset moved.")
    if exclude_file:
        excluded = pd.read_csv(exclude_file, dtype=str)
        if "fish_key" not in excluded:
            raise ValueError("Reference exclusion CSV needs fish_key; exclude whole reference fish.")
        joined = joined.loc[~joined["fish_key"].isin(excluded["fish_key"].str.strip().str.lower())]
    return joined.sort_values("scale_id").reset_index(drop=True)


def load_references(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if {r["slide"] for r in manifest["slides"]} != set(range(8, 14)):
        raise ValueError("Expert package must contain slides 8..13.")
    for row in manifest["slides"]:
        path = folder / row["image"]
        if sha(path) != row["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {path}")
    return manifest


def parse_response(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise ValueError("Model did not return a JSON object.")


def validate_response(role, value):
    if not isinstance(value.get("evidence"), str) or len(value["evidence"]) > 2500:
        raise ValueError("Missing/oversized observable evidence string.")
    confidence = value.get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be a finite number in [0,1].")
    if role == "quality":
        if value.get("decision") not in {"readable", "bad", "uncertain"}:
            raise ValueError("Invalid quality decision.")
    elif role.startswith("age_"):
        if value.get("match") not in {"yes", "no", "uncertain"}:
            raise ValueError("Invalid class match.")
    elif role in {"judge", "single"}:
        if type(value.get("prediction")) is not int or value["prediction"] not in range(-1, 5):
            raise ValueError("Prediction must be -1 (abstain), 0..3, or 4 (bad).")
        if role == "judge" and value["prediction"] == 4:
            raise ValueError("A readable-gate judge must predict age or abstain, not bad.")
    elif role == "reflection":
        if not isinstance(value.get("candidate_rule"), str) or len(value["candidate_rule"]) > 1500:
            raise ValueError("Reflection needs a short candidate_rule.")
    return value


class LocalVLM:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("No allocated CUDA GPU. Run on a scheduler GPU node, not the login node.")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16; use float16 (V100 default).")
        torch.manual_seed(args.seed)
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        self.torch = torch
        self.args = args
        self.processor = AutoProcessor.from_pretrained(
            args.model, revision=args.revision, min_pixels=256*28*28,
            max_pixels=args.max_pixels, use_fast=False, trust_remote_code=False)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, revision=args.revision, torch_dtype=dtype,
            attn_implementation="eager", trust_remote_code=False).to("cuda:0").eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        print("GPU:", torch.cuda.get_device_name(0), "model:", args.model, flush=True)

    def generate(self, prompt, images):
        content = []
        pixels = []
        for caption, path in images:
            content += [{"type": "text", "text": caption}, {"type": "image"}]
            with Image.open(path) as image:
                pixels.append(ImageOps.exif_transpose(image).convert("RGB"))
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": BASE}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=pixels, padding=True, return_tensors="pt").to("cuda:0")
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.args.max_new_tokens)
        answer = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                              skip_special_tokens=True)[0]
        del inputs, outputs
        return answer


class VisualAgents:
    def __init__(self, backend, references, folder, memory=None, mode="multi"):
        self.backend, self.references, self.folder = backend, references, Path(folder)
        self.memory, self.mode = memory or [], mode

    def ask(self, role, prompt, target, slides, extra=None):
        images = [(f"Expert reference slide {r['slide']}: {r['meaning']}", self.folder / r["image"])
                  for r in self.references["slides"] if r["slide"] in slides]
        images.append(("TARGET: unannotated scale to classify; GT is hidden.", target))
        relevant = [r["candidate_rule"] for r in self.memory if r.get("role") in {role, "reflection"}]
        prompt += "\nReference memory (may be fallible; verify against visible evidence): " + json.dumps(relevant)
        if extra:
            prompt += "\nAdditional data: " + json.dumps(extra)
        raw = self.backend.generate(prompt, images)
        try:
            return {"role": role, "response": validate_response(role, parse_response(raw)), "raw": raw}
        except ValueError as exc:
            # Retry format only; never supply a target label or invent a default prediction.
            raw_retry = self.backend.generate(prompt + "\nPrevious output failed schema validation: " + str(exc)
                                               + " Return only the requested JSON schema.", images)
            return {"role": role, "response": validate_response(role, parse_response(raw_retry)),
                    "raw": raw_retry, "first_raw": raw}

    def predict(self, path):
        if self.mode == "single":
            response = self.ask("single", 'Classify quality and age. JSON: {"prediction": -1 or 0 or 1 or 2 or 3 or 4, '
                                '"confidence": 0.0, "evidence": "short visible evidence"}. '
                                '3 means 3 or older; 4 means bad; -1 means uncertain.', path, list(range(8, 14)))
            return response["response"]["prediction"], [response]
        quality = self.ask("quality", 'Assess whether annuli can be interpreted. Use readable examples as well as '
                           'slide 13. A missing annual annulus is not automatically bad. JSON: '
                           '{"decision":"readable or bad or uncertain", "confidence":0.0, '
                           '"evidence":"visible quality evidence"}.', path, [8, 9, 13])
        traces = [quality]
        if quality["response"]["decision"] != "readable":
            return (4 if quality["response"]["decision"] == "bad" else -1), traces
        for age, slides in [(0, [8]), (1, [9]), (2, [10]), (3, [11, 12])]:
            traces.append(self.ask(f"age_{age}", f"Independently assess whether the target belongs to {NAMES[age]}. "
                                   'Do not advocate for your assigned class. Include counter-evidence. JSON: '
                                   '{"match":"yes or no or uncertain", "confidence":0.0, '
                                   '"evidence":"supporting and opposing visible evidence"}.', path, slides))
        evidence = [{"role": r["role"], **r["response"]} for r in traces]
        judge = self.ask("judge", 'Compare the independent evidence, then inspect the target yourself. '
                         'Do not use majority vote or assume different agents are independent models. '
                         'JSON: {"prediction":-1 or 0 or 1 or 2 or 3, "confidence":0.0, '
                         '"evidence":"short grounds for selection or abstention"}.', path, [8, 9, 10, 11, 12], evidence)
        traces.append(judge)
        return judge["response"]["prediction"], traces

    def reflect(self, path, gt, prediction, traces):
        return self.ask("reflection", 'This is TRAINING feedback, not evaluation. The expert target is now revealed. '
                        'A correct label does not establish the visual cause of an error. Review the target and '
                        'references. If the cause is not visible, state unknown; do not rationalize the GT. '
                        'JSON: {"confidence":0.0, "evidence":"visible observations and limitations", '
                        '"candidate_rule":"short, tentative rule requiring expert verification"}.',
                        path, list(range(8, 14)), {"expert_gt": NAMES[gt], "previous_prediction": prediction,
                                                 "previous_evidence": [r["response"] for r in traces]})


def load_memory(path, cohort, protocol, allow_unverified=False):
    if not path:
        return []
    train_fish = set(cohort.loc[cohort["split"].eq("train"), "fish_key"])
    if not Path(path).is_file():
        raise FileNotFoundError(f"Memory file not found: {path}")
    train_ids = cohort.loc[cohort["split"].eq("train")].set_index("scale_id")["fish_key"].to_dict()
    accepted = []
    for row in read_jsonl(path):
        if row.get("data_signature") != protocol or row.get("source_split") != "train":
            raise ValueError("Memory must originate from this cohort's training partition.")
        if row.get("fish_key") not in train_fish or train_ids.get(row.get("scale_id")) != row.get("fish_key"):
            raise ValueError("Memory includes held-out or unknown fish/scales.")
        if not isinstance(row.get("candidate_rule"), str):
            raise ValueError("Memory candidate_rule must be text.")
        if row.get("expert_verified") is True or allow_unverified:
            accepted.append(row)
    print("Memory:", len(accepted), "eligible entries", flush=True)
    return accepted


def run(args, backend=None):
    if args.feedback and args.split != "train":
        raise ValueError("GT reflection is allowed on train only, never validation or test.")
    if args.limit < 0 or args.memory_limit < 0 or args.max_new_tokens < 64 or args.max_pixels < 256*28*28:
        raise ValueError("Invalid limits or pixel/token budgets.")
    references = load_references(args.references)
    cohort = load_cohort(args.master, args.manifest, args.image_root, args.exclude_reference_fish)
    data_config = {"master_sha256": sha(args.master), "manifest_sha256": sha(args.manifest),
                   "excluded_reference_fish_sha256": sha(args.exclude_reference_fish) if args.exclude_reference_fish else None}
    data_signature = signature(data_config)
    memory = load_memory(args.memory, cohort, data_signature, args.allow_unverified_memory)
    memory = memory[-args.memory_limit:] if args.memory_limit else []
    frame = cohort.loc[cohort["split"].eq(args.split)].sample(frac=1, random_state=args.seed)
    if args.limit:
        frame = frame.iloc[:args.limit]
    if frame.empty:
        raise ValueError("No scales in selected partition.")
    config = {"version": VERSION, **data_config, "data_signature": data_signature,
              "references_sha256": sha(Path(args.references) / "manifest.json"),
              "model": args.model, "revision": args.revision, "dtype": args.dtype,
              "max_pixels": args.max_pixels, "max_new_tokens": args.max_new_tokens,
              "mode": args.mode, "split": args.split, "seed": args.seed, "feedback": args.feedback,
              "memory": memory, "image_root": str(args.image_root) if args.image_root else None,
              "reference_overlap_audited": bool(args.exclude_reference_fish)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Run settings changed; use a new --out directory.")
    else:
        atomic_json(config_path, config)
    if not args.exclude_reference_fish:
        print("WARNING: PPT reference fish overlap is unknown; results are exploratory.", flush=True)
    print(f"Partition={args.split}, scales={len(frame)}, mode={args.mode}, feedback={args.feedback}", flush=True)
    if args.dry_run:
        print("Dry run passed. No model loaded; no predictions or feedback generated.")
        return
    backend = backend or LocalVLM(args)
    model_info = {"requested_revision": args.revision, "resolved_revision": getattr(backend, "resolved_revision", None)}
    model_path = out / "model_info.json"
    if model_path.exists() and json.loads(model_path.read_text()) != model_info:
        raise ValueError("Model revision changed while resuming; pin --revision to a commit.")
    atomic_json(model_path, model_info)
    agents = VisualAgents(backend, references, args.references, memory, args.mode)
    prediction_path, memory_path = out / "predictions.jsonl", out / "reflection_memory.jsonl"
    records = read_jsonl(prediction_path)
    if len({r["scale_id"] for r in records}) != len(records):
        raise ValueError("Duplicate predictions; repair log before resuming.")
    done = {r["scale_id"]: r for r in records}
    if set(done) - set(frame["scale_id"]):
        raise ValueError("Saved run contains scales outside this limit/partition; do not shrink a resumed sample.")
    reflected = {r["scale_id"] for r in read_jsonl(memory_path)}
    for _, row in frame.iterrows():
        scale_id = row["scale_id"]
        image_hash = sha(row["path"])
        record = done.get(scale_id)
        if record is not None and record["image_sha256"] != image_hash:
            raise ValueError("Image changed since previous prediction; use a new run.")
        if record is None:
            started = time.perf_counter()
            try:
                prediction, traces = agents.predict(row["path"])
                error = None
            except (ValueError, OSError) as exc:
                prediction, traces, error = -1, [], f"{type(exc).__name__}: {exc}"
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError("GPU OOM. Reduce --max-pixels/token budget in a NEW output directory.") from exc
                raise
            # GT is attached only AFTER all blind inference calls have finished.
            record = {"scale_id": scale_id, "fish_key": row["fish_key"], "split": args.split,
                      "image_sha256": image_hash, "gt": int(row["target"]), "prediction": prediction,
                      "traces": traces, "error": error, "seconds": time.perf_counter()-started}
            append_jsonl(prediction_path, record)
            done[scale_id] = record
            print(f"{scale_id}: pred={prediction}, GT={record['gt']}, error={error}", flush=True)
        if args.feedback and record["prediction"] != record["gt"] and not record["error"] and scale_id not in reflected:
            reflection = agents.reflect(row["path"], record["gt"], record["prediction"], record["traces"])
            append_jsonl(memory_path, {"scale_id": scale_id, "fish_key": row["fish_key"],
                "source_split": "train", "data_signature": data_signature, "role": "reflection",
                "expert_verified": False, **reflection["response"], "raw": reflection["raw"]})
            reflected.add(scale_id)
    evaluate(out)


def evaluate(out):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
    out = Path(out)
    rows = read_jsonl(out / "predictions.jsonl")
    if not rows:
        raise ValueError("No predictions to evaluate.")
    truth = np.asarray([r["gt"] for r in rows])
    pred = np.asarray([r["prediction"] for r in rows])
    readable, bad, accepted = truth < 4, truth == 4, (pred >= 0) & (pred < 4)
    report = classification_report(truth, pred, labels=list(range(5)), target_names=NAMES,
                                   output_dict=True, zero_division=0)
    def rate(mask, values):
        return float(values[mask].mean()) if mask.any() else None
    metrics = {"support": len(rows), "accuracy": float(accuracy_score(truth, pred)),
               "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
               "macro_f1": report["macro avg"]["f1-score"],
               "abstention_rate": float((pred == -1).mean()),
               "error_count": sum(bool(r["error"]) for r in rows),
               "readable_coverage": rate(readable, accepted),
               "readable_rejected_as_bad": rate(readable, pred == 4),
               "bad_pass_rate": rate(bad, accepted),
               "readable_end_to_end_accuracy": rate(readable, pred == truth),
               "accepted_readable_age_accuracy": rate(readable & accepted, pred == truth),
               "note": "Abstentions/errors count as incorrect. Partial/limited runs are exploratory. "
                       "Same-image post-GT reflections are not new test predictions."}
    quality_pred = []
    for row in rows:
        gate = next((t["response"]["decision"] for t in row.get("traces", []) if t["role"] == "quality"), None)
        quality_pred.append({"readable": 0, "bad": 1, "uncertain": -1}.get(
            gate, 1 if row["prediction"] == 4 else 0 if row["prediction"] >= 0 else -1))
    quality_truth, quality_pred = (truth == 4).astype(int), np.asarray(quality_pred)
    metrics["quality_gate_accuracy"] = float(accuracy_score(quality_truth, quality_pred))
    quality_report = classification_report(quality_truth, quality_pred, labels=[0, 1],
        target_names=["readable", "bad"], output_dict=True, zero_division=0)
    pd.DataFrame(quality_report).T.to_csv(out / "quality_classification_report.csv")
    pd.DataFrame(confusion_matrix(quality_truth, quality_pred, labels=[0, 1, -1]),
        index=["readable", "bad", "abstain/error"], columns=["readable", "bad", "abstain/error"]
        ).to_csv(out / "quality_confusion_matrix.csv")
    atomic_json(out / "metrics.json", metrics)
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    labels = [0, 1, 2, 3, 4, -1]
    pd.DataFrame(confusion_matrix(truth, pred, labels=labels), index=NAMES + ["abstain/error"],
                 columns=NAMES + ["abstain/error"]).to_csv(out / "confusion_matrix.csv")
    pd.DataFrame([{k: r[k] for k in ["scale_id", "fish_key", "split", "gt", "prediction", "error", "seconds"]}
                  for r in rows]).to_csv(out / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Blind inference, optionally followed by train-only reflection.")
    run_p.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    run_p.add_argument("--manifest", type=Path, default=DEFAULT_SPLIT)
    run_p.add_argument("--references", type=Path, default=HERE / "agent_references")
    run_p.add_argument("--out", type=Path, required=True)
    run_p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    run_p.add_argument("--mode", choices=["multi", "single"], default="multi")
    run_p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    run_p.add_argument("--revision", default="66285546d2b821cf421d4f5eb2576359d3770cd3",
                       help="Pinned Qwen 3B revision; change when using a different model.")
    run_p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    run_p.add_argument("--max-pixels", type=int, default=512*28*28)
    run_p.add_argument("--max-new-tokens", type=int, default=384)
    run_p.add_argument("--seed", type=int, default=100)
    run_p.add_argument("--limit", type=int, default=20, help="0 runs the full partition; start with a pilot.")
    run_p.add_argument("--feedback", action="store_true")
    run_p.add_argument("--memory", type=Path)
    run_p.add_argument("--memory-limit", type=int, default=8)
    run_p.add_argument("--allow-unverified-memory", action="store_true", help="Experimental ablation; hypotheses, not GT.")
    run_p.add_argument("--image-root", type=Path)
    run_p.add_argument("--exclude-reference-fish", type=Path, help="CSV fish_key column listing all identified PPT reference fish.")
    run_p.add_argument("--dry-run", action="store_true")
    eval_p = sub.add_parser("evaluate", help="Recompute saved reports without loading a GPU model.")
    eval_p.add_argument("--out", type=Path, required=True)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        run(args) if args.command == "run" else evaluate(args.out)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
