# Expert-Guided Visual Agents on HPC

This is a local, GPU inference and reflection-memory experiment, not PPO/RL or
VLM fine-tuning. It does not update ResNet/SimCLR weights. No cloud API is used.
Do not interpret fluent explanations as proof of correct annulus detection.

## Architecture

One shared Qwen2.5-VL model runs separate roles sequentially:

1. Quality agent: readable, bad, or uncertain, using slides 8, 9 and 13. This
   assesses physical/optical suitability, NOT whether the age is already known.
   An uncertain gate receives one quality-only review; the reviewed decision
   can remain uncertain or bad. Readable does not mean a completed annulus was found.
2. Four independent class agents: 0, 1, 2, and 3 or older. Each checks evidence
   for and against its class without seeing the other class agents' answers.
3. Adjudicator: sees the target, reference slides and class evidence; predicts
   age or abstains. It is not a majority vote of independent models.
4. Optional reflection role: AFTER a blind training prediction is saved, sees
   its expert GT and records a tentative error explanation/rule.

Slide 12 is a 4+ example within the 3-or-older class. Slide 13's `Not Ring` does
not automatically mean bad; absence of an annulus can be normal for age 0.
The references retain expert arrows, boxes and image crops. They are committed
under `agent_references/`; no need to upload the full PPTX to HPC.

### Quality-Scope Correction (v2)

The initial pilot abstained on every image. Saved quality responses confused
missing/uncertain annuli with image unusability. Version 2 separates the quality
and age system prompts, requires an explicit quality reason type, and reviews
uncertain gates without supplying GT. Age-reflection memory cannot redefine the
quality gate. Quality metrics use the final reviewed gate, while both responses
remain in the trace. There is no automatic uncertain-to-readable conversion.
This fixes task instructions, not demonstrated VLM accuracy. Use a NEW output
directory; old predictions and reflection rules are preserved, not rewritten.
Start without memory or GT reflection to isolate the gate behavior:

```bash
python trout_agents.py run --split train --limit 5 --out agent_outputs/quality_scope_v2
```

If this pilot is usable, extend the same run with `--limit 20`. To collect GT
reflection afterwards, use a different output directory with `--feedback`.

## Environment

Use an allocated GPU compute node (not the login node). A V100 defaults to
float16 and eager attention; FlashAttention and bfloat16 are not required.
Only one GPU is used. Preserve the scheduler's CUDA_VISIBLE_DEVICES setting.

Use your existing working CUDA PyTorch environment, or create an isolated one:

```bash
conda create -n trout_agents python=3.11 -y
conda activate trout_agents
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-agent.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The first GPU run downloads the pinned Qwen 3B model from Hugging Face. Set
`HF_HOME` to writable storage with enough free space if the home quota is small.
Offline nodes need the model cached ahead of time using the same revision.
`--model /path/to/model` can use a local complete model/processor directory.
For another Hugging Face model revision, explicitly change `--revision`.

## Data and Split

Default master:
`/home/jlc3q/data/Trout/code_new/feature_outputs/master_with_texture_features_full.csv`.

Default manifest:
`model_outputs/age4_two_stage_search/split_manifest.csv` under this checkout.
No new split is generated. For a different output directory, pass its saved
manifest explicitly, for example:

```bash
--manifest model_outputs/age4_two_stage_search_v2/split_manifest.csv
```

Both files must agree on scale_id, fish_key and expert targets. The manifest is
the two-stage five-outcome manifest, NOT the earlier readable-only age4 split.
Source labels 0..5 become 0,1,2,3-or-older; 6 becomes bad. No length/weight is sent
to the model. Filenames and GT are not included in blind inference prompts.
If paths moved, `--image-root /home/jlc3q/data/Trout` relocates river subpaths.

## First Pilot

Pull changes first; do not discard local notebook changes if Git reports a conflict.
Start with a data-only dry run, then 20 validation scales:

```bash
python trout_agents.py run --split validation --limit 20 --out agent_outputs/multi_baseline --dry-run
python trout_agents.py run --split validation --limit 20 --out agent_outputs/multi_baseline
```

The same command resumes saved predictions without generating them again.
`--limit 0` runs the entire partition; increasing a limit extends the same seeded
sample. Other configuration changes require a new output directory.
These agents use several VLM calls per image; they are not expected to be faster
than one ResNet forward pass. Start small and inspect traces before scaling.

## Reflection and Re-Prediction

Collect GT feedback on TRAINING fish only:

```bash
python trout_agents.py run --split train --feedback --limit 20 --out agent_outputs/train_feedback
```

Incorrect predictions receive reflection AFTER the original prediction is saved.
Generated rules in `reflection_memory.jsonl` are `expert_verified=false` by default.
After actual expert review, copy the reviewed entries into a separate memory file,
correct the rule where needed, and set `expert_verified=true` for approved entries.
Then evaluate the memory-assisted agent on validation in a NEW directory:

```bash
python trout_agents.py run --split validation --limit 20 --memory agent_outputs/reviewed_memory.jsonl --out agent_outputs/multi_reviewed_memory
```

To explicitly test automatic self-reflection WITHOUT expert verification:

```bash
python trout_agents.py run --split validation --limit 20 --memory agent_outputs/train_feedback/reflection_memory.jsonl --allow-unverified-memory --out agent_outputs/multi_unverified_memory
```

That is an experimental ablation, not expert-certified knowledge. GT tells us the
answer was wrong, not its visual cause. Memory is frozen during each run; it is
not changed online while validation/test images are processed. Only rules sourced
from this exact cohort's training fish can be loaded. By default the last eight
eligible entries are used; choose any memory budget using validation only.
If the training pilot makes no errors, no reflection-memory file is produced.

## Comparisons and Final Test

Compare the single-agent baseline on the same seeded validation sample:

```bash
python trout_agents.py run --split validation --mode single --limit 20 --out agent_outputs/single_baseline
```

Compare multi-agent without memory, multi-agent with verified memory and single
agent against the existing SimCLR predictions using the same fish manifest and
scale_ids. Twenty rows are a smoke/pilot study, not a stable accuracy estimate.
Only after freezing all prompts/settings/memory, run the held-out test:

```bash
python trout_agents.py run --split test --limit 0 --memory agent_outputs/reviewed_memory.jsonl --out agent_outputs/frozen_test
```

Never use test or validation GT for reflection. Unknown/failed predictions count
as errors, not silently omitted rows. Report full-pipeline accuracy with readable
coverage and bad-pass rate, not accepted-image accuracy alone. Agent confidence
is self-reported and must not be treated as a calibrated probability.

PPT example fish may also appear in the dataset; their IDs are not provided in the
deck. Results remain exploratory until overlap is audited. If source fish are
identified, supply `--exclude-reference-fish reference_fish.csv` containing a
`fish_key` column. Exclude ALL scales of those fish consistently in baselines,
feedback and evaluation, and use that same exclusion file in every run.

## Outputs

- `predictions.jsonl`: immutable blind predictions with per-agent evidence.
- `predictions.csv`: scale IDs, expert GT, predictions, errors and runtime.
- `reflection_memory.jsonl`: tentative training error rules and provenance.
- `run_config.json`, `model_info.json`: settings, checksums and model revision.
- `metrics.json`, classification reports and confusion matrices: five final
  outcomes, readable/bad gate, and explicit abstention/error columns.

Recompute reports without a GPU:

```bash
python trout_agents.py evaluate --out agent_outputs/multi_baseline
python -m unittest discover -s tests -v
```

Changing referenced images, GT/splits or memory requires a new output directory.
Interrupted generation retries that image; completed predictions are reused.
Invalid JSON gets one formatting retry. GPU OOM stops the run; reduce pixel/token
budgets in a new directory rather than silently changing the experiment.

## Rebuilding Expert References

Optional: install LibreOffice and Poppler, then run:

```bash
python prepare_agent_references.py --pptx /path/to/Trout.pptx --out new_agent_references
```

Or supply `--pdf /path/to/Trout.pdf` exported with ALL slides, including hidden
ones. The original deck contains a hidden slide before slide 8; excluding it
shifts PDF page numbers. The preparation script validates the page count.

## Sources

- [Qwen2.5-VL model documentation](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
- [Pinned Qwen model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3)
- [Official PyTorch installation combinations](https://pytorch.org/get-started/previous-versions/)
- [Reflexion](https://arxiv.org/abs/2303.11366): related memory-based feedback idea;
  not evidence of improved trout-scale prediction.
# Fish-Level Age Propagation

## Bounded Quality, Age and Review Roles

`trout_review_agents.py` reuses the frozen quality and age adapters. This is an
explicit bounded tool workflow, not a claim of autonomous biological reasoning.
It does not train weights, learn from held-out GT, or fuse SimCLR into Qwen.

```bash
python trout_review_agents.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --quality-adapter agent_outputs/quality_lora_v1/best_adapter \
  --age-adapter agent_outputs/age_lora_h100_highres_v1/best_adapter \
  --split validation --limit 20 \
  --out agent_outputs/review_validation_pilot_v1 --dry-run
```

After this passes, remove `--dry-run` with the same arguments to load GPU tools.
For full validation use `--limit 0` and a new output directory. Keep HF cache
exports pointed to this compute node's `/local/scratch/$USER/trout_hf`.

Quality role ranks readable/bad. A small margin requests a whole-frame
high-resolution check. Disagreement or unresolved low margin refers to experts;
clear bad stops before age inference. Age role ranks 0/1/2/3-or-older, and review
checks the full image rotated 90 degrees. Unresolved age results request full-frame
higher resolution, then a mild contrast increase. Any remaining disagreement or
low margin is referred; majority vote never overrides disagreements. Agreement
does not prove correctness. There is NO crop: full scale edges are preserved.
High-resolution inspection is reprocessing the existing image, not recovery of
new detail or a new acquisition. Default review budget is 1,605,632 pixels.

Defaults `--quality-margin 0.05`, `--age-margin 0.05`, `--max-calls 6` are
EXPLORATORY. A margin is the difference between the two largest softmax weights
of mean candidate-response token log likelihoods. This is NOT a calibrated class
probability, and response length/format can affect it. Default thresholds have
not been validated. Rank inference differs from the earlier free JSON generation
evaluation, so compare on the same data/decoder before attributing gains to agents.
Each quality call performs two candidate forwards and each age call four;
max-calls bounds image inspection calls, not individual neural-network forwards.
GPU memory/runtime must be measured on HPC; local tests use fake tools only.

`predictions.jsonl` preserves per-tool evidence (class rankings, margins, view
settings, timing, review actions); `expert_review.csv` hides GT and lists image
paths/reasons for actual human review, not automatic notification. `metrics.json`
counts referrals as wrong in overall accuracy and separately reports accepted
accuracy, referral rate, coverage, bad pass rate and runtime. This workflow does
not run an oracle age model on GT-readable rejected images. It cannot claim
improvement merely from discarding difficult cases. Outputs are never overwritten;
interrupted runs retain partial logs but must restart in a new output directory.

Freeze thresholds and tool budgets using validation. Test requires
`--frozen-policy /path/to/validation_run/policy_settings.json` with matching data,
adapter hashes and parameters. Do not retune based on test outcomes. No expert GT
or fish ID enters inference; labels are attached only after each decision.
Optional SimCLR disagreement checking remains a follow-up requiring an audited
checkpoint with compatible fish splits; it is not silently enabled here.

## Train-Only Robustness Augmentation

Keep the running high-resolution baseline unchanged. After it completes, train
another age adapter with identical labels, split, resolution, LR, epochs and seed:

```bash
python train_trout_vlm.py --labels agent_outputs/supervised_labels_v1.csv \
  --task age --max-pixels 802816 --augment \
  --out agent_outputs/age_lora_h100_highres_aug_v1

python evaluate_trout_vlm.py --labels agent_outputs/supervised_labels_v1.csv \
  --quality-adapter agent_outputs/quality_lora_v1/best_adapter \
  --age-adapter agent_outputs/age_lora_h100_highres_aug_v1/best_adapter \
  --split validation --out agent_outputs/lora_validation_highres_aug_v1
```

Augmentation is OFF unless `--augment` is supplied, preserving old behavior.
Enabled augmentation independently samples a 0/90/180/270 degree turn, then a
small rotation within +/-15 degrees, brightness factor 0.85..1.15 and contrast
factor 0.85..1.15 per train image visit. Configurable via `--rotation-degrees`,
`--brightness-jitter`, `--contrast-jitter`. No crop, blur, hue changes or flips.
Rotation expands the frame and fills corners with median corner background color;
this avoids cutting scale edges but can introduce interpolation/background cues
and lower effective scale resolution under the fixed pixel budget. These choices
are starting assumptions, not validated optima. The transformed image is reused
for both chat-prefix and completion tokenization, preserving answer masking.
Validation/test are NEVER randomly augmented; quality/age labels are unchanged.
Configuration and seed are saved. Do not expect arbitrary hue robustness from
brightness/contrast alone. All original images and existing runs are preserved.

For a controlled validation stress test, add a FIXED perturbation to evaluation
and use a new output directory, e.g. `--rotation 90`, `--brightness 0.85`, or
`--contrast 0.85`. Evaluate each separately using both baseline and augmented
adapters on the same entire validation cohort. Compare clean and perturbed
age_gt_readable metrics to isolate age robustness. Pipeline metrics also include
the unchanged quality gate's sensitivity. Do not claim robustness from clean
accuracy alone or tune on test perturbations. Quality augmentation is a separate
experiment: use `--task quality --augment` with its baseline pixel budget, not a
simultaneous change to age and quality in the first paired comparison.

## Frozen Adapter Evaluation

```bash
python evaluate_trout_vlm.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --quality-adapter agent_outputs/quality_lora_v1/best_adapter \
  --age-adapter agent_outputs/age_lora_v1/best_adapter \
  --split validation --compare-base --limit 20 \
  --out agent_outputs/lora_validation_pilot_v1 --dry-run
```

Remove `--dry-run` to predict the same 20 images with base Qwen and LoRA.
For full validation remove `--limit 20` and use a NEW output directory.
Only after freezing all settings use `--split test` in another new directory.
Do not retune on test results. Keep scratch HF cache exports active in this shell.

One GPU base model loads both adapters, activating quality for usability and age
for age classification. Base comparison disables both adapters, using identical
training prompts and image budgets. There are no PPT reference images, multiple
age advocates, GT feedback or generated reasoning in this classification evaluation.
This isolates weight fine-tuning from the older prompt-based multi-agent workflow.

`comparison.csv` separates quality, oracle-GT-readable age, and predicted-gate
end-to-end outcomes. Age inference also runs for GT-readable rejected images only
to compute the oracle diagnostic; it never overrides the pipeline's quality gate.
Malformed JSON is counted as an abstention, not removed. Reports, confusion
matrices including abstentions, raw outputs, adapter/data hashes and coverage/bad
pass rates are saved. Limited samples are exploratory, not final estimates.
Validation is reused from checkpoint selection, not an unbiased final test.
Runs do not resume automatically: predictions.jsonl preserves partial records on
interruption; use a new output directory for a restart. Real CUDA evaluation
must still be verified on HPC. Do not run the old trout_agents.py expecting it
to automatically activate these adapters.

## Actual VLM Weight Training

`train_trout_vlm.py` performs response-only supervised LoRA fine-tuning of the
pinned Qwen2.5-VL-3B model. Train quality and age separately. Expert age GT does
not supply reasoning GT: targets contain only decision/prediction JSON, not
fabricated annulus explanations. PPT slides remain inference references, not
extra labeled training images. Language attention is adapted; the vision encoder
is frozen. This is supervised learning, not RL or whole-model fine-tuning.

```bash
export HF_HOME="/local/scratch/$USER/trout_hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HUB_CACHE"
python -m pip install --no-cache-dir -r requirements-vlm-training.txt

python train_trout_vlm.py --labels agent_outputs/supervised_labels_v1.csv \
  --task quality --out agent_outputs/quality_lora_v1 --dry-run
python train_trout_vlm.py --labels agent_outputs/supervised_labels_v1.csv \
  --task age --out agent_outputs/age_lora_v1 --dry-run

python train_trout_vlm.py --labels agent_outputs/supervised_labels_v1.csv \
  --task quality --out agent_outputs/quality_lora_v1
python train_trout_vlm.py --labels agent_outputs/supervised_labels_v1.csv \
  --task age --out agent_outputs/age_lora_v1
```

Use an allocated GPU node (V100: fp16). Scratch cache is node-local and is not a
persistent backup. Outputs remain in the repository's data filesystem. Never
overwrite a prior run: use v2 etc. Default micro-batch 1, accumulation 8, rank 8,
3 epochs and 1e-4 LR are a starting configuration, not searched optima. Pixel
budget is 200704; downsampling may lose fine annulus information. Check GPU
memory; reduce `--max-pixels` in a NEW run on OOM. GPU training has not been
verified on the developer Mac. Run the dry-run first, then a real training run.

Checkpoint selection uses validation answer-token loss, NOT validation macro-F1.
No test images enter training or selection. A held-out generation evaluation is
still needed to establish classification accuracy. `best_adapter/` contains LoRA
weights, not the complete base model. Existing `trout_agents.py` does not yet
load these adapters: its old inference results will not change automatically.

Prepare supervised training labels without changing the existing fish split:

```bash
python prepare_trout_training_labels.py \
  --master /home/jlc3q/data/Trout/code_new/feature_outputs/master_with_texture_features_full.csv \
  --manifest model_outputs/age4_two_stage_search/split_manifest.csv \
  --out agent_outputs/supervised_labels_v1.csv
```

Use the actual saved manifest path if your run used a different output directory.
Optional `--quality-csv expert_quality.csv` accepts `scale_id,quality_gt`, where
quality_gt is an expert's image-specific `readable` or `bad` annotation, never a
model prediction. Original labels 0..5 imply directly labeled readable scales;
6 denotes bad. Missing labels imply unknown quality unless separately annotated.

Only separately confirmed readable scales can receive an age from the same fish
when all its direct expert ages agree. Quality is never propagated. Conflicting
fish are flagged and excluded from age training pending expert review. Unassigned
fish are retained for audit but excluded from supervised training. Validation/test
age evaluation uses direct expert labels only, not propagated labels. Every output
row records label provenance; length and weight are excluded. Existing outputs
are never overwritten.

This command prepares a new training table; it does NOT fine-tune Qwen, update
SimCLR, or change the existing agent inference cohort. Propagated scales increase
image count, not the number of independently labeled fish. Until unknown-quality
scales receive expert quality annotations, they do not expand age training.
