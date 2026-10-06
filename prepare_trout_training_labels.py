"""Prepare fish-propagated ages without propagating scale quality or changing splits."""
import argparse
from pathlib import Path

import pandas as pd


def prepare_labels(master, manifest, quality=None):
    master = master.copy()
    manifest = manifest.copy()
    for name, table in [("master", master), ("manifest", manifest)]:
        required = {"scale_id", "fish_key"} | ({"label", "path"} if name == "master" else {"split"})
        if not required <= set(table):
            raise ValueError(f"{name} requires {sorted(required)}")
        for col in ["scale_id", "fish_key"]:
            if table[col].isna().any():
                raise ValueError(f"Missing {col} in {name}")
            table[col] = table[col].astype(str).str.strip().str.lower()
            if table[col].eq("").any():
                raise ValueError(f"Empty {col} in {name}")
        if table.scale_id.duplicated().any():
            raise ValueError(f"Duplicate scale_id in {name}")
    if not manifest["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown manifest split")
    if manifest.groupby("fish_key")["split"].nunique().gt(1).any():
        raise ValueError("Fish leakage across splits")
    identities = manifest.merge(master[["scale_id", "fish_key"]], on="scale_id", how="left",
                                suffixes=("_saved", ""), validate="one_to_one")
    if not identities.fish_key.eq(identities.fish_key_saved).all():
        raise ValueError("Manifest/master identities differ")
    raw = master["label"]
    numeric = pd.to_numeric(raw, errors="coerce")
    if (raw.notna() & ~numeric.isin(range(7))).any():
        raise ValueError("Expert labels must be missing or integers 0..6")
    # Discard the source EDA 'split' (labeled/unlabeled), retaining the saved fish partition.
    master = master.drop(columns=["split"], errors="ignore")
    fish_splits = manifest[["fish_key", "split"]].drop_duplicates()
    rows = master.merge(fish_splits, on="fish_key", how="left", validate="many_to_one")
    rows["expert_age"] = pd.to_numeric(rows.label, errors="coerce").where(
        pd.to_numeric(rows.label, errors="coerce").lt(6)).astype("Int64")
    rows["quality_gt"] = "unknown"
    rows.loc[rows.expert_age.notna(), "quality_gt"] = "readable"
    rows.loc[pd.to_numeric(rows.label, errors="coerce").eq(6), "quality_gt"] = "bad"
    rows["quality_source"] = rows.quality_gt.map({"unknown": "unknown", "readable": "expert_direct", "bad": "expert_direct"})
    if quality is not None:
        quality = quality.copy()
        if not {"scale_id", "quality_gt"} <= set(quality):
            raise ValueError("Quality CSV requires scale_id, quality_gt")
        quality["scale_id"] = quality.scale_id.astype(str).str.strip().str.lower()
        if quality.scale_id.duplicated().any() or not quality.scale_id.isin(rows.scale_id).all():
            raise ValueError("Duplicate or unknown quality scale IDs")
        if not quality.quality_gt.isin(["readable", "bad"]).all():
            raise ValueError("Quality CSV must contain expert readable/bad labels, not predictions")
        supplied = rows.scale_id.map(quality.set_index("scale_id").quality_gt)
        if (supplied.notna() & rows.quality_gt.ne("unknown") & supplied.ne(rows.quality_gt)).any():
            raise ValueError("Quality GT conflicts with existing expert labels")
        rows.loc[supplied.notna(), "quality_gt"] = supplied[supplied.notna()]
        rows.loc[supplied.notna(), "quality_source"] = "expert_quality_csv"
    # Unassigned fish cannot become donors or recipients; never invent new partitions.
    assigned = rows[rows["split"].notna()]
    groups = assigned.groupby(["split", "fish_key"])["expert_age"]
    consensus = groups.agg(lambda s: s.dropna().unique()[0] if s.dropna().nunique() == 1 else pd.NA)
    counts = groups.nunique()
    keys = pd.MultiIndex.from_frame(rows[["split", "fish_key"]])
    rows["fish_age_conflict"] = counts.reindex(keys).fillna(0).to_numpy() > 1
    rows["age_label"] = rows.expert_age
    rows["age_label_source"] = "unlabeled"
    rows.loc[rows.expert_age.notna(), "age_label_source"] = "expert_direct"
    eligible = rows.expert_age.isna() & rows.quality_gt.eq("readable") & rows["split"].notna()
    inherited = pd.Series(consensus.reindex(keys).to_numpy(), index=rows.index, dtype="Int64")
    propagate = eligible & inherited.notna() & ~rows.fish_age_conflict
    rows.loc[propagate, "age_label"] = inherited[propagate]
    rows.loc[propagate, "age_label_source"] = "fish_propagated"
    rows["age4"] = rows.age_label.clip(upper=3).astype("Int64")
    rows["age_train_eligible"] = (rows["split"].eq("train") & rows.quality_gt.eq("readable")
                                  & rows.age_label.notna() & ~rows.fish_age_conflict)
    rows["age_evaluation_eligible"] = (rows["split"].isin(["validation", "test"])
                                       & rows.quality_gt.eq("readable")
                                       & rows.age_label_source.eq("expert_direct"))
    rows["quality_train_eligible"] = rows["split"].eq("train") & rows.quality_gt.ne("unknown")
    # Training inputs explicitly omit biological size measurements.
    return rows[["scale_id", "fish_key", "path", "split", "label", "expert_age", "quality_gt",
                 "quality_source", "fish_age_conflict", "age_label", "age_label_source", "age4",
                 "age_train_eligible", "age_evaluation_eligible", "quality_train_eligible"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--quality-csv", type=Path, help="Optional image-specific EXPERT quality annotations")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Output exists; choose a new filename to preserve previous labels")
    result = prepare_labels(pd.read_csv(args.master), pd.read_csv(args.manifest),
                            pd.read_csv(args.quality_csv) if args.quality_csv else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.groupby(["split", "quality_gt", "age_label_source"], dropna=False).size().to_string())
    print("Age training rows:", int(result.age_train_eligible.sum()))
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
