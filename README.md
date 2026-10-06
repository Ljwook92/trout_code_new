# Trout Code New

Initial EDA and table-building code for the corrected trout dataset.

## Terminal-Based Visual Agents

`trout_agents.py` runs a local GPU VLM with expert slides 8..13, a readable/bad
gate, four independent age roles and a final adjudicator. It optionally collects
train-only GT reflection memory; this is not weight fine-tuning or RL. Existing
notebooks are unchanged. See [AGENTS_HPC.md](AGENTS_HPC.md) for environment setup,
pilot commands, memory review, fixed fish splits and held-out evaluation.

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

## SimCLR Backbone and Texture Comparison

`trout_age4_simclr_backbone_texture_comparison.ipynb` compares four configurations:
SimCLR ResNet18, ResNet18 + texture, ResNet50 + texture, and ResNet50.
Each configuration trains a readable/bad gate and a separate age classifier
for readable scales: 0, 1, 2, and 3 or older. Label 6 denotes bad, not an age.

This notebook defaults to `/home/jlc3q/data/Trout/trout_code_new`, matching the
actual HPC checkout. It automatically checks the full texture master under
`code_new` if the CSV is not in the checkout. Set `TROUT_TEXTURE_MASTER` or edit
`INPUT_CSV` for other locations. Run the cohort audit first,
then set `RUN_TRAINING=True` and run from the setup cell. Keep
`EVALUATE_TEST=False` during search; enable it only after selection is frozen.

All models share a saved fish-level train/validation/test manifest. SimCLR uses
readable and bad training scales only. Validation macro F1 selects each task's
hyperparameters and best-epoch checkpoint. Image-only and texture fusion search
the same pretrained candidates for each backbone.
Outputs, preprocessing objects, test predictions and checkpoints are saved in
`model_outputs/age4_two_stage_search`. Length and weight are excluded.
Use a new output directory when changing the cohort or split configuration.
The old comparison outputs are preserved. Changing search settings also requires
a new output directory. Adding bad changes the fish cohort;
retrain on the new split rather than importing old checkpoints.

### Validation-Only Hyperparameter Search

The default budget is two SimCLR candidates per backbone and three classifier
candidates per pretrained candidate/task: four pretraining runs and up to 48
classifier trials across the four model pairs. The previous baseline is included
when its values are in the search space. All configurations get the same candidate
lists; CUDA out-of-memory trials are recorded, not silently changed.

- `SIMCLR_SPACE`: learning rate, temperature, batch size and epoch budget.
- `CLASSIFIER_SPACE`: learning rate and batch size.
- `CLASSIFIER_MAX_EPOCHS=30`, `EARLY_STOP_PATIENCE=5`: validation macro-F1 early stopping.
- `N_PRETRAIN_TRIALS` / `N_CLASSIFIER_TRIALS`: random-search budgets.
- `SEARCH_MODE="grid"`: all finite combinations, potentially very expensive.
- `THRESHOLD_GRID`: validation-only bad-probability threshold sweep.
- `MAX_READABLE_REJECTION`: optional validation rejection-rate constraint.

Completed trials resume using matching `search_config.json` settings. Trial
histories, selected validation tables, threshold curves/plots and selected
thresholds are saved. Interrupted individual trials restart; completed trials are
reused. `NUM_WORKERS=0` and AMP are execution settings, not searched model parameters.
This is a limited search, not proof of a global optimum. More extensive searches
should use grouped nested validation; never retune based on the held-out test.

Three comparison tables are saved: `comparison_results.csv` (age with GT-readable
inputs), `quality_comparison_results.csv` (readable/bad), and
`pipeline_comparison_results.csv` (end-to-end, all labeled test images).
The pipeline table also reports readable rejection rate, bad pass rate, readable
coverage, and age accuracy among accepted readable images. Each matched model
pair uses its validation-selected threshold, maximizing five-class pipeline macro
F1 by default. `EVALUATE_TEST=True` evaluates the frozen selections, never each
search candidate. The quality test table uses the same selected gate threshold.
