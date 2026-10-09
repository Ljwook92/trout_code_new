# Feature-Based Quality / Age / Review Workflow

This connects the trained feature-conditioned Qwen models to bounded tool-using
controllers. Qwen never receives pixels, filenames, fish sizes, existing GT, or
other model claims as truth. It receives only projected SimCLR feature tokens
and its trained task prompt. It is NOT an independently trained debate/reasoning
model: quality/age agents use their learned classifiers; the review agent is an
explicit, auditable controller that chooses inspections, acceptance or referral.

## Flow

1. Quality Agent invokes the quality bundle on the full original frame.
2. Review Agent requests a full-frame 90-degree rotation to check consistency.
3. Two consecutive reliable readable decisions permit age analysis. Two reliable
   bad decisions stop it. Unresolved quality requests contrast, then optionally
   high-resolution inspection. Still unresolved -> expert; age is not attempted.
4. Age Agent uses the age bundle on original/rotated full frames. Review Agent
   requests further full-frame contrast/high-resolution checks if needed.
5. Strong cross-view class disagreement is not settled by majority vote. Persistent
   disagreement, malformed JSON or budget exhaustion -> expert referral (-1).
6. Experts annotate actual TRAIN images; the importer writes a NEW GT table.
   Separately retrain and validate new bundles. Nothing updates weights during
   prediction, and no prediction/reasoning is converted to expert GT automatically.

Reliable means valid greedy class, matching candidate ranking, and a sufficient
gap between the highest two mean response-token log-likelihood scores. Those
scores/gaps are NOT calibrated probabilities and include JSON formatting tokens.
Defaults are exploratory starting settings, not proven acceptance thresholds.
View agreement does not prove a correct age or verified visual explanation.

Original and contrast/rotation views are always full-frame, normalized and resized
to the training size 224x224. No cropping or brightness jitter. `--highres-size 448`
enables a full-frame 448x448 ResNet probe only when unresolved; it is explicitly
logged as outside training resolution and may worsen predictions. SimCLR features
remain the only Qwen image input. No annulus/damage explanation is fabricated.

## First Pilot

Use an allocated H100 and the existing training environment. Preserve old outputs.

```bash
cd /home/jlc3q/data/Trout/trout_code_new
git pull --ff-only
export HF_HOME="/local/scratch/$USER/trout_hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HUB_CACHE"

python trout_feature_agents.py run \
  --labels agent_outputs/supervised_labels_v1.csv \
  --quality-run agent_outputs/quality_feature_qwen_balanced_v2 \
  --age-run agent_outputs/age_feature_qwen_balanced_v2 \
  --split validation --limit 20 \
  --out agent_outputs/feature_agents_validation_pilot_v1 --dry-run
```

Run the same command without `--dry-run` to load Qwen and generate predictions.
The v1 feature model runs can be used instead, but changing bundles starts a new
policy experiment. Optional `--highres-size 448` also requires a new output folder.
Margins default to 0.05 each, contrast 1.2, max inspections 8 (quality and age
combined). Normally consistent readable+age requires 4 inspections; bad requires
2. Each inspection performs bounded greedy decoding plus 2/4 candidate forward
passes, so inspection count is NOT a transformer forward count or latency limit.

## Outputs and Evaluation

- `predictions.jsonl`: final decisions, view traces, raw answers, scores and post-decision audit GT.
- `policy.json`: exact tool settings, bundle hashes and workflow code hash.
- `metrics.json`: accuracy with referrals as wrong, accepted accuracy WITH referral
  rate/coverage, bad pass rate, call counts and time.
- `confusion.csv`: five classes plus the expert-review column.
- `expert_queue.jsonl`: referred images/model traces, WITHOUT audit GT.
- `expert_review.html`: portable image previews; existing GT hidden and model
  inspection details collapsed. Original full-resolution paths remain in CSV.
- `expert_annotations.csv`: blank human annotation template, never auto-filled GT.

First examine the pilot, then remove `--limit` and choose a new output directory
for all validation images. Select policy margins/settings on validation only.
Never judge accepted accuracy without its referral/coverage tradeoff. `--split test`
requires `--frozen-policy` pointing to the matching validation `policy.json`; model
hashes/settings/code must match exactly. Since this project's test was already
inspected, a new workflow test evaluation is exploratory, not fresh independent
evidence for model selection. Do not tune on it.

New image inference is supported with `--input-csv new_images.csv` containing
unique `scale_id,fish_key,path`; no GT is required. `--labels` is still the original
training table for bundle provenance, not a source of inference answers. Input CSV
may carry extra columns but they are not sent to the workflow. Predictions on new
images have no GT-based accuracy metric. Do not change `--labels` until models
have actually been retrained using the new label file.

## Expert Feedback and Separate Retraining

Run the workflow with `--split train` into a new directory to collect TRAIN review
cases, or supply an input CSV of unlabeled scales belonging to already-assigned
training fish. Validation/test review is diagnostic ONLY and cannot enter training.

In `expert_annotations.csv` experts fill:
- `quality_gt`: readable or bad, determined independently for each scale.
- `expert_age`: exact original expert age 0..5 for readable scales when known;
  blank for bad. 3..5 are mapped to the age-model class "3 or older". Do not invent
  an exact age from a model's 3+ output. Age may stay blank if unavailable.
- `reviewer`, `reviewed_at`: actual human attribution and review date/time.

Leave scale/fish/path/image hash unchanged. Reviewer text is required provenance,
not an authentication mechanism; real expert confirmation must happen outside
this script. Do not automatically fill the template from model predictions.

```bash
python trout_feature_agents.py apply-feedback \
  --labels agent_outputs/supervised_labels_v1.csv \
  --annotations agent_outputs/feature_agents_train_review_v1/expert_annotations.csv \
  --out agent_outputs/supervised_labels_expert_review_v2.csv
```

The importer rejects non-train annotations and modified images/fish/paths, records
pre-review labels plus reviewer/hash provenance, and preserves fish partitions.
Age conflicts are recomputed and excluded from age training. Inherited ages of
affected fish are cleared instead of silently keeping stale propagated GT; quality
is never propagated. Blank age on a newly-readable image cannot invent supervision.

Retrain with the existing `train_trout_feature_qwen.py` using the NEW labels file,
the SAME provenance-checked encoder/split, and NEW output directories. For example:

```bash
ENCODER_RUN="model_outputs/age4_two_stage_search_v4"
python train_trout_feature_qwen.py \
  --labels agent_outputs/supervised_labels_expert_review_v2.csv \
  --encoder-run "$ENCODER_RUN" \
  --encoder-checkpoint "$ENCODER_RUN/search_trials/resnet18_pre000_simclr.pt" \
  --task age --class-balanced --selection-metric validation_macro_f1 \
  --out agent_outputs/age_feature_qwen_expert_review_v3
```

Run separately for `--task quality` with its own new output directory. Validate
both new bundles using `evaluate_trout_feature_qwen.py` with the same NEW label
table. Only then explicitly select the new run paths for the agent workflow. There
is deliberately no automatic deployment/promotion or training on generated answers.
