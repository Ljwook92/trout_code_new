"""Train separate bad/readable role LoRAs on expert quality GT, not invented reasons."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from train_trout_vlm import MODEL, REVISION, augment_image
from trout_agents import sha
from trout_quality_debate import DECISION_PROMPT, ROLES, public_history, quality_rows, system_prompt


def training_rows(table, role, feedback=None, labels_hash=None):
    train, val = quality_rows(table, "train"), quality_rows(table, "validation")
    for rows in [train, val]:
        rows["answer"] = rows.quality_gt.map(lambda x: json.dumps({"decision": x}))
        rows["prompt"] = DECISION_PROMPT
    if feedback is not None:
        meta = json.loads(feedback.with_suffix(feedback.suffix + ".meta.json").read_text())
        if not meta.get("train_only") or meta["labels_sha256"] != labels_hash:
            raise ValueError("Feedback must match TRAIN labels")
        by_id = train.set_index("scale_id")
        corrections, seen = [], set()
        for line in feedback.read_text().splitlines():
            row = json.loads(line)
            if row.get("role") not in ROLES:
                raise ValueError("Invalid feedback role")
            if row["role"] != role:
                continue
            if row["scale_id"] not in by_id.index or row["scale_id"] in seen:
                raise ValueError("Non-train or duplicate corrective example")
            source = by_id.loc[row["scale_id"]].to_dict()
            if row["decision"] != source["quality_gt"] or row.get("target_source") != "expert_quality_gt" or row.get("reasoning_target") is not False:
                raise ValueError("Corrective targets must be expert quality labels, not self-reasons")
            seen.add(row["scale_id"])
            # Re-sanitize contextual observations; never include expert GT in the user prompt.
            history = public_history(row["history"])
            source["prompt"] += "\nPrevious fallible quality assessments (data, not instructions):\n" + json.dumps(history)
            corrections.append(source)
        if corrections:
            train = pd.concat([train, pd.DataFrame(corrections)], ignore_index=True)
    return train, val


def train_role(args, role, train, val, out):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("Use an allocated CUDA GPU")
    seed = args.seed + ROLES.index(role)
    set_seed(seed)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, use_fast=False,
                                              min_pixels=56 * 56, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16, attn_implementation="eager")
    targets = [name for name, module in model.named_modules() if name.endswith(("q_proj", "v_proj"))
               and "visual" not in name and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("No language attention LoRA targets found")
    if args.init_agents:
        model = PeftModel.from_pretrained(model, str(args.init_agents / role / "best_adapter"), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
            lora_dropout=.05, target_modules=targets, task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.print_trainable_parameters()

    def collate(examples):
        if len(examples) != 1:
            raise ValueError("Micro-batch size one is required")
        row = examples[0]
        with Image.open(row["path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        image = augment_image(image, row["split"], args.augment, rotation_degrees=15,
                              brightness_jitter=0, contrast_jitter=args.contrast_jitter)
        messages = [{"role": "system", "content": system_prompt(role)}, {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(messages + [{"role": "assistant", "content": row["answer"]}],
                                              tokenize=False, add_generation_prompt=False)
        batch = processor(text=[full], images=[image], return_tensors="pt", padding=True)
        prefix_ids = processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"]
        n = prefix_ids.shape[1]
        if not torch.equal(prefix_ids, batch["input_ids"][:, :n]):
            raise ValueError("Chat prefix is not aligned; refusing incorrect loss masking")
        labels = batch["input_ids"].clone()
        labels[:, :n] = -100
        labels[batch["attention_mask"].eq(0)] = -100
        if not labels.ne(-100).any():
            raise ValueError("No expert answer tokens")
        batch["labels"] = labels
        return batch

    out.mkdir(parents=True)
    config = {"protocol": "quality_role_supervision_v1", "task": "quality", "quality_role": role,
        "model": MODEL, "revision": REVISION, "labels_sha256": sha(args.labels),
        "system": system_prompt(role), "prompt": DECISION_PROMPT, "max_pixels": args.max_pixels,
        "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "seed": seed,
        "accumulation": args.accumulation, "augment": args.augment, "contrast_jitter": args.contrast_jitter,
        "brightness_jitter": 0, "vision_encoder_frozen": True, "reasoning_supervision": False,
        "selection_metric": "validation_answer_loss", "feedback_sha256": sha(args.feedback) if args.feedback else None,
        "initialization": str(args.init_agents) if args.init_agents else "pinned_base_model",
        "initial_adapter_sha256": sha(args.init_agents / role / "best_adapter/adapter_model.safetensors") if args.init_agents else None,
        "n_train_examples": len(train), "n_validation": len(val)}
    (out / "training_config.json").write_text(json.dumps(config, indent=2))
    pd.concat([train, val]).to_csv(out / "used_rows.csv", index=False)
    training_args = TrainingArguments(output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=args.accumulation,
        fp16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, prediction_loss_only=True,
        remove_unused_columns=False, label_names=["labels"], report_to="none", logging_steps=10,
        dataloader_num_workers=0, seed=seed)
    trainer = Trainer(model=model, args=training_args, data_collator=collate,
                      train_dataset=train.to_dict("records"), eval_dataset=val.to_dict("records"))
    trainer.train()
    trainer.save_model(str(out / "best_adapter"))
    processor.save_pretrained(out / "best_adapter")
    trainer.save_state()
    print("Saved:", out / "best_adapter", flush=True)
    del trainer, model, processor
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=["both", *ROLES], default="both")
    parser.add_argument("--feedback", type=Path, help="Train-only corrective examples from prepare-feedback")
    parser.add_argument("--init-agents", type=Path, help="Continue weights from a prior two-role training directory")
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--contrast-jitter", type=float, default=.15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.max_pixels < 56 * 56 or min(args.epochs, args.rank, args.accumulation) < 1 or
            not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.contrast_jitter) or
            not 0 <= args.contrast_jitter < 1):
        raise ValueError("Invalid training settings")
    if args.out.exists():
        raise FileExistsError("Use a new training output directory")
    table = pd.read_csv(args.labels)
    roles = ROLES if args.role == "both" else [args.role]
    prepared = {}
    for role in roles:
        if args.init_agents:
            cfg = json.loads((args.init_agents / role / "training_config.json").read_text())
            if (cfg.get("task") != "quality" or cfg.get("quality_role") != role or
                    cfg.get("labels_sha256") != sha(args.labels) or cfg.get("model") != MODEL or
                    cfg.get("revision") != REVISION or cfg.get("system") != system_prompt(role) or
                    cfg.get("prompt") != DECISION_PROMPT or cfg.get("rank") != args.rank):
                raise ValueError("Initial role adapter provenance/rank differs")
            if not (args.init_agents / role / "best_adapter/adapter_model.safetensors").is_file():
                raise FileNotFoundError("Missing initial role adapter")
        train, val = training_rows(table, role, args.feedback, sha(args.labels))
        if not pd.concat([train, val]).path.map(lambda p: Path(p).is_file()).all():
            raise FileNotFoundError("Missing train/validation images")
        prepared[role] = train, val
        print(f"role={role} train={len(train)} validation={len(val)}; GT-label loss, not reasoning loss", flush=True)
    if args.dry_run:
        print("Dry run passed; no GPU model loaded or files written")
        return
    for role in roles:
        train_role(args, role, *prepared[role], args.out / role)


if __name__ == "__main__":
    main()
