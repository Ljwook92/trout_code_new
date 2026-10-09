# Two-agent quality gate

This is an additive quality-only experiment. Existing age, review, training and
memory code is unchanged. No API key or cloud inference is required. Use an
allocated GPU node and the existing `requirements-vlm-training.txt` environment.
It shares one pinned Qwen2.5-VL-3B base, with TWO separately trained LoRA adapters.
Agents are different roles/weights, not independent biological experts.

## Evidence-format repair (v2)

The first HPC pilot showed copied schema examples and unlabelled pixel-like
coordinates. The v2 evidence prompt uses field instructions without literal
observation sentences to copy. Defect locations are now described in words;
numeric boxes are NOT requested. Existing role adapters remain compatible:
their training system and decision prompt have not changed. No retraining is
needed just to test this inference repair. Use a NEW evaluation output directory,
such as `quality_debate_validation_pilot_v2`.

Copied observations such as ellipses or the old template phrases are rejected,
not counted as preservation evidence. Unknown coordinate units are never guessed
or rescaled: unsolicited invalid boxes are ignored with logged warnings. A bad
claim still requires a textual location or a valid normalized box. A normalized
box remains an unverified model claim, not expert localization. Original evidence
errors and raw model responses are preserved rather than replaced with a generic
binary-decision error. Budget, margins and abstention accounting are unchanged.

## 1. Train both quality role adapters

```bash
cd /home/jlc3q/data/Trout/trout_code_new
git pull --ff-only
export HF_HOME="/local/scratch/$USER/trout_hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HUB_CACHE"

python train_trout_quality_agents.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --out agent_outputs/quality_agents_v1 --dry-run

python train_trout_quality_agents.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --out agent_outputs/quality_agents_v1
```

Both agents see BOTH expert classes. Bad Agent audits defects and possible false
alarms; Readable Agent audits preservation and possible overlooked defects.
Neither is forced to advocate its name. Training is sequential to limit GPU
memory. Each role has its own initialized LoRA, seed and validation-selected
checkpoint. Outputs are `bad/best_adapter` and `readable/best_adapter` under the
new training folder. Existing `quality_lora_v1` is not overwritten or reused.

Training uses only image-specific expert quality GT, never fish-level propagated
quality. The response-only loss supervises `decision`, NOT auto-generated defect
explanations or bounding boxes. Vision encoder remains frozen. Validation chooses
the lowest answer loss; there is no test training. Defaults are exploratory,
not searched optimum settings. Augmentation is OFF; optional `--augment` adds
quarter turns, small expanded rotations and contrast jitter, with NO brightness
jitter or cropping. Validate the unaugmented experiment first.

## 2. Validation pilot: actual evidence and debate

```bash
python trout_quality_debate.py run \
  --labels agent_outputs/supervised_labels_v1.csv \
  --bad-adapter agent_outputs/quality_agents_v1/bad/best_adapter \
  --readable-adapter agent_outputs/quality_agents_v1/readable/best_adapter \
  --split validation --limit 20 \
  --out agent_outputs/quality_debate_validation_pilot_v1 --dry-run

python trout_quality_debate.py run \
  --labels agent_outputs/supervised_labels_v1.csv \
  --bad-adapter agent_outputs/quality_agents_v1/bad/best_adapter \
  --readable-adapter agent_outputs/quality_agents_v1/readable/best_adapter \
  --split validation --limit 20 \
  --out agent_outputs/quality_debate_validation_pilot_v1
```

Each agent generates center/material/visibility observations. Claimed defects
require a location in words, not ambiguous numeric coordinates. The
agent then ranks readable/bad responses conditioned on its own evidence. This
uses two teacher-forced candidate forwards, not calibrated class probabilities.
The evidence generation is greedy. Raw JSON failures are recorded and cannot
pass the gate. There are no silent formatting retries or guessed classifications.

First round: independent full-image assessments. Both roles receive the SAME
preceding history on later rounds, avoiding within-round anchoring. Subsequent
rounds use higher-resolution full frames, without crops or altered contrast.
Each includes actual peer observations and requests reinspection, not voting.

Initial agreement passes only with sufficient margins AND compatible structured
evidence: readable requires all criteria preserved; bad requires a localized
defect. After a disagreement/low margin, two consecutive confident agreeing
rounds are needed. Default maximum is 3 rounds, 6 assessments, up to 18 model
generation/forward calls. Exhaustion, uncertainty or persistent contradiction
goes to expert review. CUDA/runtime failures stop the run rather than masquerade
as biological uncertainty. Consensus itself is NOT expert confirmation.

An oval/asymmetric scale is NOT automatically bad. The agents distinguish shape
from disrupted central growth-line organization, tears/folds, optical issues and
regeneration. The expert image GT takes precedence over textual heuristics.
Do not redefine labels based on the agent's proposed defects.

Outputs: `predictions.jsonl` contains full observations, peer responses, ranking
scores and all rounds; `quality_report.txt`, `quality_confusion.csv`, `metrics.json`
contain GT comparisons. `agent_comparison.csv` shows each role's initial/final
decision metrics. Schema-error cases are counted separately in `metrics.json`.
`readable_queue.csv` is the only queue allowed into a
future age-agent stage. `expert_review.csv/jsonl` omit quality GT for independent
expert review. No age inference is performed by this script yet.

Evaluate all validation images by using `--limit 0` and a NEW output directory.
Track readable coverage, readable falsely rejected as bad, bad passed as readable,
referral rate and accepted accuracy TOGETHER. Abstentions count as wrong in total
accuracy/F1. Small pilots are debugging evidence, not a generalization estimate.
Test requires `--frozen-policy` pointing to the validation `policy_settings.json`;
settings, label hashes and adapters must match. Never use test feedback for training.

## 3. GT-based correction and continued learning

Run a TRAIN debate (same arguments as above, `--split train --limit 0`, with a new
output e.g. `agent_outputs/quality_debate_train_v1`). Predictions are completed
before GT is attached. Then:

```bash
python trout_quality_debate.py prepare-feedback \
  --labels agent_outputs/supervised_labels_v1.csv \
  --run agent_outputs/quality_debate_train_v1 \
  --out agent_outputs/quality_corrections_v1.jsonl

python train_trout_quality_agents.py \
  --labels agent_outputs/supervised_labels_v1.csv \
  --init-agents agent_outputs/quality_agents_v1 \
  --feedback agent_outputs/quality_corrections_v1.jsonl \
  --out agent_outputs/quality_agents_v2
```

Corrective examples retain prior fallible model observations as input, but the
target is the EXPERT readable/bad label. They cover each role's errors and all
referrals, with one corrective example per role/image. Original expert train
examples are retained to avoid training only on mistakes. This is actual LoRA
weight updating, not online RL or memory-only improvement. Corrections do not
claim that a self-generated explanation is correct. Initial weights and source
feedback hashes are recorded. Every iteration uses a new folder.

This first implementation does not train fabricated rationales. To supervise
specific defect positions/reasons, add expert-verified localized annotations in
a subsequent experiment. Scalar quality GT alone does not establish the model's
claimed cause. Inspect generated evidence before expanding a training loop.
Re-evaluate v2 on the SAME validation cohort; do not assume debate or continued
training necessarily improves accuracy or calibration.
