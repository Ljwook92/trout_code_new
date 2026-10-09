"""Offline visual audit of decision changes and the actual retrieved train images."""
import argparse
import base64
import csv
import hashlib
import html
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row["scale_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate image identities: " + str(path))
    return {row["scale_id"]: row for row in rows}


def load_runs(baseline, memory_run, memory):
    old_policy = json.loads((baseline / "policy_settings.json").read_text())
    new_policy = json.loads((memory_run / "policy_settings.json").read_text())
    if old_policy.get("memory") or not new_policy.get("memory"):
        raise ValueError("Compare a no-memory baseline against a memory-enabled run")
    keys = ["quality_margin", "age_margin", "quality_pixels", "age_pixels", "review_pixels",
            "max_calls", "feedback_rounds", "labels_sha256", "adapter_hashes", "scoring"]
    for key in keys:
        if key not in old_policy or old_policy[key] != new_policy.get(key):
            raise ValueError("Non-memory policy setting changed: " + key)
    old_config = json.loads((baseline / "run_config.json").read_text())
    new_config = json.loads((memory_run / "run_config.json").read_text())
    if old_config["split"] != new_config["split"] or old_config["training_configs"] != new_config["training_configs"]:
        raise ValueError("Evaluation partition or training configuration differs")
    if new_config["split"] not in ["validation", "test"]:
        raise ValueError("Expected a held-out evaluation partition")
    if sha(memory / "memory_config.json") != new_policy["memory"]["config_sha256"]:
        raise ValueError("Memory configuration differs from the inference run")
    meta = json.loads((memory / "memory_config.json").read_text())
    if meta["labels_sha256"] != new_policy["labels_sha256"] or meta["adapter_hashes"] != new_policy["adapter_hashes"]:
        raise ValueError("Memory label/adapter provenance differs")
    if sha(memory / "entries.jsonl") != meta["artifact_hashes"]["entries.jsonl"]:
        raise ValueError("Memory entries checksum mismatch")
    entries = read_jsonl(memory / "entries.jsonl")
    if any(row["split"] != "train" for row in entries.values()):
        raise ValueError("Memory includes a held-out reference")
    old = read_jsonl(baseline / "predictions.jsonl")
    new = read_jsonl(memory_run / "predictions.jsonl")
    if old.keys() != new.keys():
        raise ValueError("Cohorts differ; compare identical image sets")
    heldout_fish = {row["fish_key"] for row in new.values()}
    if heldout_fish & {row["fish_key"] for row in entries.values()}:
        raise ValueError("Train-memory/evaluation fish overlap")
    for scale_id in old:
        for key in ["fish_key", "gt", "image_sha256"]:
            if old[scale_id][key] != new[scale_id][key]:
                raise ValueError("Evaluation identity/GT/image changed: " + scale_id)
    return old, new, entries


def outcome(row):
    if row["prediction"] == -1:
        return "referral"
    return "correct" if row["prediction"] == row["gt"] else "incorrect"


def references(row, entries):
    hits = []
    for call, trace in enumerate(row["traces"], 1):
        for rank, hit in enumerate(trace.get("retrieved_training_references", []), 1):
            source = entries.get(hit["scale_id"])
            task = trace["task"]
            if task not in ["quality", "age"] or source is None:
                raise ValueError("Unknown reference/task in inference trace")
            label_source = source["quality_source" if task == "quality" else "age_label_source"]
            if (source["fish_key"] != hit["fish_key"] or source["fish_key"] == row["fish_key"]
                    or source[task + "_class"] != hit["class"] or hit["label_source"] != label_source
                    or Path(source["path"]).resolve() != Path(hit["path"]).resolve()):
                raise ValueError("Reference identity, class or label source differs")
            hits.append({"scale_id": row["scale_id"], "task": task, "call": call,
                         "view": trace["view"], "rank": rank, "reference_scale_id": hit["scale_id"],
                         "reference_fish_key": hit["fish_key"], "reference_class": hit["class"],
                         "label_source": label_source, "cosine_similarity": hit["cosine_similarity"],
                         "model_correct_on_train": hit.get("model_correct"), "reference_path": source["path"]})
    return hits


def label(value, task="pipeline"):
    return ({0: "readable", 1: "bad", -1: "referral"} if task == "quality" else
            {0: "0+", 1: "1+", 2: "2+", 3: "3 or older", 4: "bad", -1: "referral"}).get(value, str(value))


def write_csv(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generate(baseline, memory_run, memory, out, max_side=2400, all_cases=False):
    if out.exists():
        raise FileExistsError("Choose a new audit output directory")
    if max_side < 256:
        raise ValueError("Preview size must be at least 256 pixels")
    old, new, entries = load_runs(baseline, memory_run, memory)
    changed = [key for key in new if (old[key]["prediction"], old[key]["status"]) !=
               (new[key]["prediction"], new[key]["status"])]
    selected = list(new) if all_cases else changed
    # Validate every logged retrieval, including unchanged decisions, before reporting.
    hits_by_id = {key: references(row, entries) for key, row in new.items()}
    previews = {}

    def preview(path, expected_hash):
        key = (str(path), expected_hash)
        if key not in previews:
            if sha(path) != expected_hash:
                raise ValueError("Image bytes changed since inference/memory creation: " + str(path))
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
                width, height = image.size
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                stream = io.BytesIO()
                image.save(stream, format="JPEG", quality=92)
            data = "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
            previews[key] = f'<img src="{data}" alt="Scale preview"><details><summary>Enlarge preview</summary><img style="height:auto" src="{data}" alt="Expanded scale preview"></details><p>Original: {width} x {height}; preview: {image.width} x {image.height}</p>'
        return previews[key]

    def trace_table(row):
        lines = []
        for trace in row["traces"]:
            values = [trace["task"], trace["view"], label(trace["prediction"], trace["task"]),
                      f'{trace["margin"]:.4f}', trace["pixels"], trace.get("rotation", 0), trace.get("contrast", 1)]
            lines.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in values) + "</tr>")
        return '<table><tr><th>Task</th><th>View</th><th>Prediction</th><th>Margin*</th><th>Pixels</th><th>Rotation</th><th>Contrast</th></tr>' + "".join(lines) + "</table>"

    sections, retrieval_rows, annotations = [], [], []
    for key in selected:
        before, after = old[key], new[key]
        hits = hits_by_id[key]
        target = preview(after["path"], after["image_sha256"])
        columns = ['<figure><h3>Target (original)</h3>' + target + "</figure>"]
        unique = {}
        for hit in hits:
            unique.setdefault((hit["task"], hit["reference_scale_id"]), []).append(hit)
        for (task, reference_id), uses in unique.items():
            source, hit = entries[reference_id], uses[0]
            caption = f'{task}: {label(hit["reference_class"], task)} | cosine={hit["cosine_similarity"]:.4f} | {reference_id}'
            calls = ", ".join(str(h["call"]) for h in uses)
            columns.append("<figure><h3>Train reference</h3>" + preview(source["path"], source["image_sha256"]) +
                           "<p>" + html.escape(caption) + "</p><p>" + html.escape(f'Calls: {calls}; label source: {hit["label_source"]}; model correct on train: {hit["model_correct_on_train"]}') + "</p></figure>")
            annotations.append({"scale_id": key, "task": task, "reference_scale_id": reference_id,
                                **{name: "" for name in ["color_similarity", "shape_similarity", "growth_pattern_similarity",
                                   "annual_band_evidence", "physical_quality_similarity", "reference_relevance", "reviewer", "notes"]}})
        retrieval_rows.extend(hits)
        header = f'{key}: {label(before["prediction"])} -> {label(after["prediction"])}; {outcome(before)} -> {outcome(after)}'
        sections.append('<section><h2>' + html.escape(header) + '</h2><p>Diagnostic GT: ' + html.escape(label(after["gt"])) +
                        '</p><div class="images">' + "".join(columns) + '</div><h3>No-memory trace</h3>' + trace_table(before) +
                        '<h3>Memory trace</h3>' + trace_table(after) + '<p>Baseline reason: ' + html.escape(before["reason"]) +
                        '<br>Memory reason: ' + html.escape(after["reason"]) + '</p>' +
                        ("" if hits else "<p>No references were used in this case.</p>") + '</section>')
    summary = {"support": len(new), "changed_cases": len(changed), "displayed_cases": len(selected),
               "baseline_outcomes": dict(Counter(outcome(r) for r in old.values())),
               "memory_outcomes": dict(Counter(outcome(r) for r in new.values())),
               "changed_transitions": dict(Counter(outcome(old[k]) + " -> " + outcome(new[k]) for k in changed)),
               "note": "Descriptive paired audit, not evidence of causal benefit or annulus-specific retrieval. Human annotations are not training labels."}
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trout Retrieval Audit</title><style>body{font:15px system-ui;margin:24px;color:#222}section{border-top:1px solid #aaa;padding:20px 0}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:16px}figure{margin:0}img{width:100%;height:420px;object-fit:contain;background:#eee}p{overflow-wrap:anywhere}table{border-collapse:collapse;display:block;overflow-x:auto}td,th{border:1px solid #ccc;padding:7px;text-align:left}pre{white-space:pre-wrap}h2{font-size:20px}</style><h1>Trout Retrieval Audit</h1><p>Post-hoc diagnostic only. GT is displayed for inspection, not supplied to inference. *Margins and cosine similarities are not calibrated probabilities.</p><p>Compare color/background and overall shape separately from center-to-edge growth-line crowding and candidate annual bands. Crowding alone is not proof of an annulus. For quality, compare physical/optical defects instead of age cues.</p><p>Previews are resized and JPEG encoded. Click to enlarge; subtle annual bands may require full-resolution originals. Retrieval captions describe logged references, not expert-verified visual evidence. Fill review_annotations_template.csv; no annotations are automatically learned.</p><pre>' + html.escape(json.dumps(summary, indent=2)) + '</pre>' + ("".join(sections) or "<p>No final prediction/status changed. Use --all-cases to inspect retrievals in unchanged cases.</p>") + '</html>'
    out.mkdir(parents=True)
    (out / "retrieval_audit.html").write_text(document)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "audit_config.json").write_text(json.dumps({"baseline": str(baseline), "memory_run": str(memory_run),
        "feature_memory": str(memory), "max_side": max_side, "all_cases": all_cases,
        "input_sha256": {str(path): sha(path) for path in [baseline / "predictions.jsonl", memory_run / "predictions.jsonl",
            baseline / "policy_settings.json", memory_run / "policy_settings.json", memory / "memory_config.json"]}}, indent=2))
    changes = [{"scale_id": k, "gt": new[k]["gt"], "baseline_prediction": old[k]["prediction"],
                "memory_prediction": new[k]["prediction"], "baseline_status": old[k]["status"],
                "memory_status": new[k]["status"], "transition": outcome(old[k]) + " -> " + outcome(new[k])} for k in changed]
    write_csv(out / "changed_cases.csv", changes, ["scale_id", "gt", "baseline_prediction", "memory_prediction", "baseline_status", "memory_status", "transition"])
    write_csv(out / "retrieved_examples.csv", retrieval_rows, ["scale_id", "task", "call", "view", "rank", "reference_scale_id", "reference_fish_key", "reference_class", "label_source", "cosine_similarity", "model_correct_on_train", "reference_path"])
    write_csv(out / "review_annotations_template.csv", annotations, ["scale_id", "task", "reference_scale_id", "color_similarity", "shape_similarity", "growth_pattern_similarity", "annual_band_evidence", "physical_quality_similarity", "reference_relevance", "reviewer", "notes"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--memory-run", required=True, type=Path)
    parser.add_argument("--feature-memory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-side", type=int, default=2400)
    parser.add_argument("--all-cases", action="store_true", help="Also display cases with unchanged final decisions")
    args = parser.parse_args()
    print(json.dumps(generate(args.baseline, args.memory_run, args.feature_memory, args.out,
                              args.max_side, args.all_cases), indent=2))
    print("Report:", args.out / "retrieval_audit.html")


if __name__ == "__main__":
    main()
