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

Run `trout_new_dataset_eda.ipynb` first. It builds `eda_outputs/master_table_new.csv` by joining image paths, labels, fish length, and fish weight.

The notebook defaults to:

```python
ROOT_DIR = Path("/home/jlc3q/data/Trout")
CODE_DIR = ROOT_DIR / "code_new"
DEMO_XLSX = ROOT_DIR / "demo.xlsx"
LABEL_XLSX = ROOT_DIR / "labeling.xlsx"
```

Data files and generated outputs are intentionally not tracked by git.
