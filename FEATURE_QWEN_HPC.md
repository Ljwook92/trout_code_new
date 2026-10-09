# SimCLR features to Qwen, supervised by expert GT

This is a new input architecture, not the previous image-based LoRA/debate model.
Images enter a frozen, trained SimCLR ResNet18 only. Qwen's visual encoder is
bypassed. A trainable projector maps a global 512-dimensional vector and 49
spatial vectors into 50 continuous Qwen input tokens. `--grid 1` uses the pooled
vector alone. Spatial vectors are not expert annulus annotations.

Training updates the projector and language q/v LoRA. The only target is expert
GT JSON, with response-only cross entropy. No pseudo-labels, teacher predictions,
distillation loss, generated reasoning targets, fish filenames, length or weight
enter Qwen. Age uses only direct expert labels on readable scales, grouped into
0, 1, 2, 3 or older; quality uses image-specific readable/bad GT. Propagated ages
are excluded in this version. Test images are never loaded for training/evaluation.

## Before starting

Use an allocated H100 GPU and the existing Python environment from the earlier
LoRA runs (`requirements-vlm-training.txt`). Scratch is node-local and may need
fresh downloads when the compute node changes:

```bash
cd /home/jlc3q/data/Trout/trout_code_new
git pull --ff-only
export HF_HOME="/local/scratch/$USER/trout_hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HUB_CACHE"

ENCODER_RUN="model_outputs/age4_two_stage_search"
ls "$ENCODER_RUN"/simclr_resnet18*best.pt
ls "$ENCODER_RUN"/split_manifest.csv "$ENCODER_RUN"/search_config.json
```

The paths below match the notebook's two-stage search output names. If your run
is stored elsewhere, change `ENCODER_RUN` to that actual directory. Do not use an
old encoder trained with a different fish split. The script verifies checkpoint
search signature, manifest digest, image identities and fish partitions. A plain
legacy backbone checkpoint without provenance is deliberately rejected.

## Dry Run

```bash
python train_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --encoder-run "$ENCODER_RUN" \
  --encoder-checkpoint "$ENCODER_RUN/simclr_resnet18_quality_best.pt" \
  --task quality --out agent_outputs/quality_feature_qwen_v1 --dry-run

python train_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --encoder-run "$ENCODER_RUN" \
  --encoder-checkpoint "$ENCODER_RUN/simclr_resnet18_best.pt" \
  --task age --out agent_outputs/age_feature_qwen_v1 --dry-run
```

These are supervised fine-tuned SimCLR encoders from the notebook: the classifier
head is discarded, leaving its learned backbone. An original SimCLR pretraining
checkpoint with `backbone_state_dict` and matching search signature also works.
Separate task-specific encoders are useful here but teacher classification heads
are not needed. No prediction from those heads is used as a target.

## Train

```bash
python train_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --encoder-run "$ENCODER_RUN" \
  --encoder-checkpoint "$ENCODER_RUN/simclr_resnet18_quality_best.pt" \
  --task quality --out agent_outputs/quality_feature_qwen_v1

python train_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --encoder-run "$ENCODER_RUN" \
  --encoder-checkpoint "$ENCODER_RUN/simclr_resnet18_best.pt" \
  --task age --out agent_outputs/age_feature_qwen_v1
```

Defaults: 3 epochs, microbatch 1, accumulation 8, both learning rates 1e-4,
LoRA rank 8. These are starting settings, not claimed optimal values. The lowest
validation answer loss selects the bundle; no test tuning is performed. Optional
`--augment` applies train-only quarter rotations and contrast jitter, with no
brightness jitter or crop. It is off by default to keep this first run simple.
Images are resized to 224x224 with ImageNet normalization, matching the previous
SimCLR classifier. This does NOT solve resolution limitations automatically.

## Saved Bundle

- `training_config.json`: checkpoint, label and split provenance, preprocessing.
- `used_rows.csv`: exact direct-GT train/validation rows.
- `encoder.pt`: frozen ResNet18 weights, packaged independently of the old path.
- `history.csv`: per-epoch training and validation answer loss.
- `best_bundle/projector.pt`: feature mapping and learned spatial positions.
- `best_bundle/qwen_adapter/`: new Qwen language LoRA and tokenizer.
- `best_bundle/selection.json`: selected epoch and losses.

Inference must load the encoder, projector and new adapter together. Existing
`evaluate_trout_vlm.py`, image-based quality debate, review scripts and their old
LoRAs are NOT compatible with this bundle. This implements supervised training,
not expert-quality reasoning or a finished multi-agent decision policy.

## Validation Evaluation

The feature-specific evaluator loads the frozen bundles, checks label hashes and
saved train/validation partitions, and uses greedy JSON generation without GT
answers, pixels, augmentation or gradient updates. Encoder normalization and
float32 precision match training. Quality gate: 0 readable, 1 bad; pipeline class
4 is bad, not age 4. Malformed/truncated answers abstain (-1) and count as wrong.

First check files/weights without loading Qwen:

```bash
python evaluate_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --quality-run agent_outputs/quality_feature_qwen_v1 \
  --age-run agent_outputs/age_feature_qwen_v1 \
  --split validation --out agent_outputs/feature_qwen_validation_v1 --dry-run
```

Then run the full validation partition:

```bash
python evaluate_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --quality-run agent_outputs/quality_feature_qwen_v1 \
  --age-run agent_outputs/age_feature_qwen_v1 \
  --split validation --out agent_outputs/feature_qwen_validation_v1
```

`--limit 20` may be added for a quick, exploratory pilot; use a different output
directory. For final held-out test evaluation, `--split test --confirm-test` is
required. Do not tune settings after inspecting test results.

Outputs: `comparison.csv`, `predictions.jsonl`, `evaluation_config.json` (bundle
hashes), `feature_qwen_quality_report.txt`, `feature_qwen_age_gt_readable_report.txt`,
`feature_qwen_pipeline_report.txt`, corresponding `_confusion.csv` matrices (with
an abstain column), and `feature_qwen_coverage.json`. Age is inferred for every
image without consulting GT; age-only metrics include only expert-readable images.
The pipeline uses the predicted gate, so quality mistakes reduce end-to-end scores.
This is generation-based classification, not calibrated probability estimation.

## Balanced Training Follow-Up (Train/Validation Only)

Preserve the original v1 bundles and their already-inspected test results. This
follow-up uses the SAME expert GT, encoder, image size, splits, seed and architecture,
with two explicit changes: inverse-class-frequency train sampling and best-epoch
selection by generated validation macro F1. It is a combined experiment, not an
isolated estimate of the effect of sampling. No improvement is guaranteed.

```bash
ENCODER_RUN="/home/jlc3q/data/Trout/trout_code_new/model_outputs/age4_two_stage_search_v4"
ENCODER_CHECKPOINT="$ENCODER_RUN/search_trials/resnet18_pre000_simclr.pt"

python train_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --encoder-run "$ENCODER_RUN" --encoder-checkpoint "$ENCODER_CHECKPOINT" \
  --task age --class-balanced --selection-metric validation_macro_f1 \
  --out agent_outputs/age_feature_qwen_balanced_v2

python train_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --encoder-run "$ENCODER_RUN" --encoder-checkpoint "$ENCODER_CHECKPOINT" \
  --task quality --class-balanced --selection-metric validation_macro_f1 \
  --out agent_outputs/quality_feature_qwen_balanced_v2
```

Use the exact encoder used by v1 (check its `training_config.json`); the pre000
path above matches the previous v1 command. The script does not initialize from
the v1 LoRAs: it retrains the projector/LoRA from the same base and seed to keep
the comparison interpretable. Augmentation remains off for this experiment.

Each epoch draws the original number of training rows, WITH replacement, using
only training class counts. Classes have equal expected sampling probability,
not guaranteed equal sampled counts. This adds neither new GT nor new fish.
Validation/test rows are not resampled. Config records original class counts;
history records sampled counts and unique images each epoch.

When selecting macro F1, the script generates class predictions on all validation
rows each epoch and records accuracy, balanced accuracy, macro F1, per-class
recall and abstention rate. Quality additionally records bad pass rate, readable
coverage and false bad rejections. Tied macro F1 is broken by lower answer loss.
`validation_predictions_epoch_N.csv` retains predictions/raw answers for audit.
This costs extra inference time; defaults still preserve v1 loss-only selection.

After BOTH training jobs finish:

```bash
python evaluate_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --quality-run agent_outputs/quality_feature_qwen_balanced_v2 \
  --age-run agent_outputs/age_feature_qwen_balanced_v2 \
  --split validation --out agent_outputs/feature_qwen_balanced_validation_v2
```

Compare with v1 validation, particularly age 2/3+ recall, bad pass rate, readable
coverage, and overall macro F1. Keep the existing v1 test result as the reported
fixed-model result. Reusing that test for this new experiment is exploratory,
not a fresh independent final evaluation; do not use it to select improvements.
