"""Train-only visual feature retrieval. Expert labels, never predictions, define classes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_trout_vlm import MODEL, REVISION


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_memory_rows(table):
    required = {"scale_id", "fish_key", "path", "split", "quality_gt", "quality_source",
                "age4", "age_label_source", "age_train_eligible"}
    if not required <= set(table):
        raise ValueError("Use the supervised label table")
    if table.scale_id.isna().any() or table.scale_id.duplicated().any() or table.fish_key.isna().any():
        raise ValueError("Invalid image/fish identities")
    if table.groupby("fish_key")["split"].nunique().gt(1).any():
        raise ValueError("Fish leakage in labels")
    rows = table[table["split"].eq("train") & table.quality_gt.isin(["readable", "bad"])].copy()
    if rows.empty:
        raise ValueError("No expert quality-labeled training images")
    if not rows.quality_source.isin(["expert_direct", "expert_quality_csv"]).all():
        raise ValueError("Memory quality must come from expert annotations")
    flags = rows.age_train_eligible.astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("Invalid age eligibility")
    eligible = flags.eq("true")
    if (eligible & (~rows.quality_gt.eq("readable") | ~rows.age4.isin(range(4)) |
                    ~rows.age_label_source.isin(["expert_direct", "fish_propagated"]))).any():
        raise ValueError("Invalid age memory labels")
    rows["quality_class"] = rows.quality_gt.eq("bad").astype(int)
    rows["age_class"] = rows.age4.where(eligible, -1).astype(int)
    return rows.sort_values("scale_id").reset_index(drop=True)


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all() or vectors.ndim != 2:
        raise ValueError("Invalid feature matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Zero visual feature vector")
    return vectors / norms


def class_prototypes(rows, features):
    prototypes = {}
    for task in ["quality", "age"]:
        column = task + "_class"
        for label in sorted(set(rows[column]) - {-1}):
            subset = rows[rows[column].eq(label)]
            # First average within each fish, then across fish; many scales cannot dominate.
            fish_vectors = [features[group.index].mean(0) for _, group in subset.groupby("fish_key")]
            prototype = normalize(np.asarray(fish_vectors)).mean(0)
            prototypes[f"{task}_{label}"] = normalize(prototype[None, :])[0].tolist()
    return prototypes


class FeatureMemory:
    def __init__(self, folder, labels_path, expected_adapter_hashes):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "memory_config.json").read_text())
        if (self.meta["labels_sha256"] != file_sha(labels_path) or self.meta["model"] != MODEL
                or self.meta["revision"] != REVISION or self.meta["adapter_hashes"] != expected_adapter_hashes):
            raise ValueError("Feature memory data/model/adapter provenance mismatch")
        for name in ["entries.jsonl", "features.npy"]:
            if file_sha(self.folder / name) != self.meta["artifact_hashes"][name]:
                raise ValueError("Memory artifact checksum mismatch")
        self.rows = pd.DataFrame([json.loads(line) for line in (self.folder / "entries.jsonl").read_text().splitlines()])
        self.features = normalize(np.load(self.folder / "features.npy", allow_pickle=False))
        if len(self.rows) != len(self.features) or self.rows.scale_id.duplicated().any():
            raise ValueError("Memory feature/identity mismatch")
        allowed = training_memory_rows(pd.read_csv(labels_path)).set_index("scale_id")
        self.path_to_fish = dict(zip(allowed.path.map(lambda p: str(Path(p).resolve())), allowed.fish_key))
        if not self.rows["split"].eq("train").all() or not self.rows.scale_id.isin(allowed.index).all():
            raise ValueError("Held-out or unknown images in feature memory")
        for row in self.rows.to_dict("records"):
            source = allowed.loc[row["scale_id"]]
            for key in ["fish_key", "quality_class", "age_class", "quality_source", "age_label_source"]:
                if source[key] != row[key]:
                    raise ValueError("Memory expert label/identity changed")
            if Path(source["path"]).resolve() != Path(row["path"]).resolve():
                raise ValueError("Memory image path differs from its expert-labeled source")

    def search(self, vector, task, top_k=2, exclude_fish=None):
        if task not in ["quality", "age"] or top_k < 1:
            raise ValueError("Invalid memory query")
        vector = normalize(np.asarray(vector)[None, :])[0]
        if vector.shape[0] != self.features.shape[1]:
            raise ValueError("Query feature dimension differs")
        scores = self.features @ vector
        eligible = self.rows[task + "_class"].ge(0).to_numpy(copy=True)
        if exclude_fish is not None:
            eligible &= self.rows.fish_key.ne(exclude_fish).to_numpy()
        indices = np.flatnonzero(eligible)
        indices = sorted(indices, key=lambda i: (-float(scores[i]),
                         not bool(self.rows.iloc[i].get(task + "_correct", False)), self.rows.iloc[i].scale_id))
        hits, seen = [], set()
        for index in indices:
            row = self.rows.iloc[index].to_dict()
            if row["fish_key"] in seen:
                continue
            if file_sha(row["path"]) != row["image_sha256"]:
                raise ValueError("Retrieved reference image changed")
            seen.add(row["fish_key"])
            hits.append({"path": row["path"], "scale_id": row["scale_id"], "fish_key": row["fish_key"],
                         "class": int(row[task + "_class"]), "label_source": row["quality_source"] if task == "quality"
                         else row["age_label_source"], "cosine_similarity": float(scores[index]),
                         "model_correct": row.get(task + "_correct"), "expert_visual_explanation_verified": False})
            if len(hits) >= top_k:
                break
        return hits


def main():
    import argparse
    from trout_review_agents import LoRATools
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--quality-adapter", required=True, type=Path)
    parser.add_argument("--age-adapter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pixels", type=int, default=200704)
    parser.add_argument("--limit", type=int, default=0, help="0: full train memory; pilot limits are exploratory")
    parser.add_argument("--audit-predictions", action="store_true", help="Mark correct/incorrect train predictions; retains both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.pixels < 56 * 56 or args.limit < 0:
        raise ValueError("Invalid pixel budget/limit")
    rows = training_memory_rows(pd.read_csv(args.labels))
    if args.limit:
        rows = rows.sample(frac=1, random_state=100).head(args.limit).reset_index(drop=True)
    if not rows.path.map(lambda p: Path(str(p)).is_file()).all():
        raise FileNotFoundError("Missing train images")
    label_hash, configs, hashes = file_sha(args.labels), {}, {}
    for task, folder in [("quality", args.quality_adapter), ("age", args.age_adapter)]:
        config = json.loads((folder.parent / "training_config.json").read_text())
        if (config["task"] != task or config["labels_sha256"] != label_hash
                or config["model"] != MODEL or config["revision"] != REVISION):
            raise ValueError("Training adapter provenance mismatch")
        configs[task] = config
        hashes[task] = file_sha(folder / "adapter_model.safetensors")
    print(f"Train memory images={len(rows)}, fish={rows.fish_key.nunique()}, audit={args.audit_predictions}", flush=True)
    if args.dry_run:
        print("Dry run passed; no GPU model loaded")
        return
    if args.out.exists():
        raise FileExistsError("Choose a new memory directory")
    tools = LoRATools(args.quality_adapter, args.age_adapter, configs)
    args.out.mkdir(parents=True)
    entries, vectors = [], []
    for row in rows.to_dict("records"):
        vector = tools.embed(row["path"], args.pixels)
        entry = {key: row[key] for key in ["scale_id", "fish_key", "path", "split", "quality_class", "age_class",
                                          "quality_source", "age_label_source"]}
        entry["image_sha256"] = file_sha(row["path"])
        if args.audit_predictions:
            for task in ["quality", "age"]:
                if row[task + "_class"] >= 0:
                    result = tools.inspect(row["path"], task, configs[task]["max_pixels"])
                    entry[task + "_prediction"] = result["prediction"]
                    entry[task + "_correct"] = result["prediction"] == row[task + "_class"]
        entries.append(entry)
        vectors.append(vector)
        with (args.out / "entries.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, allow_nan=False) + "\n")
        print("Stored:", row["scale_id"], flush=True)
    features = normalize(np.stack(vectors))
    np.save(args.out / "features.npy", features, allow_pickle=False)
    (args.out / "class_prototypes.json").write_text(json.dumps(class_prototypes(pd.DataFrame(entries), features)))
    summary = []
    for task in ["quality", "age"]:
        for label, group in pd.DataFrame(entries).groupby(task + "_class"):
            if label < 0:
                continue
            summary.append({"task": task, "class": label, "images": len(group), "fish": group.fish_key.nunique(),
                            "correct_predictions": int(group[task + "_correct"].sum()) if args.audit_predictions else None})
    pd.DataFrame(summary).to_csv(args.out / "class_summary.csv", index=False)
    meta = {"protocol": "qwen_visual_feature_memory_v1", "model": MODEL, "revision": REVISION,
            "labels_sha256": label_hash, "adapter_hashes": hashes, "pixels": args.pixels,
            "embedding": "mean_pooled_frozen_Qwen_visual_tokens_L2_normalized", "limit": args.limit,
            "audit_predictions": args.audit_predictions, "count": len(entries), "train_only": True,
            "artifact_hashes": {name: file_sha(args.out / name) for name in ["entries.jsonl", "features.npy"]}}
    (args.out / "memory_config.json").write_text(json.dumps(meta, indent=2))
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
