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
4. `trout_age4_image_only_models.ipynb`
   - Focused notebook for the main research objective.
   - Compares texture-only, CNN image-only, and CNN+texture fusion models.
   - Does not use fish length or weight.
5. `trout_age4_simclr_comparison.ipynb`
   - Trains a SimCLR ResNet18 backbone and age4 classifier.
   - Uses the same fish-level split policy.
   - Compares SimCLR against the saved image-only model results.
6. `trout_age7_simclr.ipynb`
   - Trains SimCLR and a 7-class classifier for labels 0 through 6.
   - Keeps label 6 as the bad/not-readable/regenerated class.
   - Uses scale images only and fish-level splitting.
7. `trout_age6_simclr_45combined.ipynb`
   - Trains SimCLR after combining original labels 4 and 5.
   - Uses six target classes: 0, 1, 2, 3, 4/5 combined, and 6/bad.
   - Uses scale images only and fish-level splitting.

The notebook defaults to:

```python
ROOT_DIR = Path("/home/jlc3q/data/Trout")
CODE_DIR = ROOT_DIR / "code_new"
DEMO_XLSX = ROOT_DIR / "demo.xlsx"
LABEL_XLSX = ROOT_DIR / "labeling.xlsx"
```

Data files and generated outputs are intentionally not tracked by git.
