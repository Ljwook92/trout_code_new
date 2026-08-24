# Trout Code New

Initial EDA and table-building code for the corrected trout dataset.

Expected HPC layout:

```text
/home/jlc3q/data/Trout/
├── code_new/
├── cu/
├── du/
├── po/
├── to/
├── tu/
├── demo.xlsx
└── labeling.xlsx
```

Run notebooks in this order:

1. `trout_new_dataset_eda.ipynb`
   - Builds `eda_outputs/master_table_new.csv` by joining image paths, labels, fish length, and fish weight.
2. `trout_texture_feature_extraction.ipynb`
   - Reads `eda_outputs/master_table_new.csv`.
   - Extracts handcrafted texture features from scale images.
   - Saves `feature_outputs/texture_features_*.csv` and `feature_outputs/master_with_texture_features_*.csv`.
3. `trout_age4_model_comparison.ipynb`
   - Compares age4 models with fish-level train/test splitting.
   - Runs length/weight-only, texture-only, and texture+length/weight RandomForest baselines.
   - Includes optional CNN image-only and CNN+tabular fusion cells.

The notebook defaults to:

```python
ROOT_DIR = Path("/home/jlc3q/data/Trout")
CODE_DIR = ROOT_DIR / "code_new"
DEMO_XLSX = ROOT_DIR / "demo.xlsx"
LABEL_XLSX = ROOT_DIR / "labeling.xlsx"
```

Data files and generated outputs are intentionally not tracked by git.
