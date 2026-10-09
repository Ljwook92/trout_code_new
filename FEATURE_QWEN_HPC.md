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

Inference must load the encoder, projector and new adapter together, using this
script's token layout and `feature_batch`. Existing `evaluate_trout_vlm.py`,
image-based quality debate, review scripts and their old LoRAs are NOT compatible
with this bundle. This script implements supervised training, not a claim of
expert-quality reasoning or a finished multi-agent decision policy. Classification
evaluation and agent integration follow after checking these training artifacts.
