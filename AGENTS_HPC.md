# Expert-Guided Visual Agents on HPC

This is a local, GPU inference and reflection-memory experiment, not PPO/RL or
VLM fine-tuning. It does not update ResNet/SimCLR weights. No cloud API is used.
Do not interpret fluent explanations as proof of correct annulus detection.

## Architecture

One shared Qwen2.5-VL model runs separate roles sequentially:

1. Quality agent: readable, bad, or uncertain, using slides 8, 9 and 13.
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
