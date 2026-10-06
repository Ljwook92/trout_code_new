"""Supervised, response-only Qwen LoRA training; no generated reasoning targets."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from trout_agents import AGE_SYSTEM, QUALITY_SYSTEM

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def select_rows(table, task):
    required = {"scale_id", "fish_key", "path", "split", "quality_gt", "age4",
                "age_label_source", "age_train_eligible", "age_evaluation_eligible"}
    if not required <= set(table):
        raise ValueError("Use the supervised table produced by prepare_trout_training_labels.py")
    if table.scale_id.isna().any() or table.scale_id.duplicated().any() or table.fish_key.isna().any():
        raise ValueError("Invalid image/fish identities")
    if table.groupby("fish_key")["split"].nunique().gt(1).any():
        raise ValueError("Fish leakage across splits")
    def flag(column):
        values = table[column].astype(str).str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"Invalid boolean column: {column}")
        return values.eq("true")
    if task == "age":
        train = table[table["split"].eq("train") & flag("age_train_eligible")].copy()
        val = table[table["split"].eq("validation") & flag("age_evaluation_eligible")].copy()
        for rows in [train, val]:
            if not rows.quality_gt.eq("readable").all() or not rows.age4.isin(range(4)).all():
                raise ValueError("Age supervision must be readable with age4 0..3")
            rows["answer"] = rows.age4.map(lambda x: json.dumps({"prediction": int(x)}))
        if not val.age_label_source.eq("expert_direct").all():
            raise ValueError("Validation must use direct expert age labels")
    else:
        train = table[table["split"].eq("train") & table.quality_gt.isin(["readable", "bad"])].copy()
        val = table[table["split"].eq("validation") & table.quality_gt.isin(["readable", "bad"])].copy()
        for rows in [train, val]:
            rows["answer"] = rows.quality_gt.map(lambda x: json.dumps({"decision": x}))
    if train.empty or val.empty:
        raise ValueError("Both training and direct-GT validation rows are required")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--task", choices=["quality", "age"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.accumulation < 1 or args.rank < 1 or args.max_pixels < 56 * 56:
        raise ValueError("Invalid training parameters")
    train, val = select_rows(pd.read_csv(args.labels), args.task)
    for rows in [train, val]:
        missing = rows.loc[~rows.path.map(lambda p: Path(str(p)).is_file()), "path"]
        if len(missing):
            raise FileNotFoundError(f"Missing image: {missing.iloc[0]}")
    print(f"task={args.task} train={len(train)} validation={len(val)}; test is not used", flush=True)
    print(train.answer.value_counts().to_string(), flush=True)
    if args.dry_run:
        print("Dry run passed; no model loaded or output written.")
        return
    if args.out.exists():
        raise FileExistsError("Use a new output directory; existing experiments are preserved")

    import torch
    from PIL import Image, ImageOps
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("An allocated CUDA GPU is required")
    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION, use_fast=False,
                                              min_pixels=56 * 56, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16, attn_implementation="eager")
    # Adapt language attention only; preserve the pretrained vision encoder.
    targets = [name for name, module in model.named_modules()
               if name.endswith(("q_proj", "v_proj")) and "visual" not in name
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError("No language attention LoRA targets found")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
                                            lora_dropout=0.05, target_modules=targets,
                                            task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.print_trainable_parameters()
    system = QUALITY_SYSTEM if args.task == "quality" else AGE_SYSTEM
    prompt = ('Judge only scale usability. Return JSON with decision: readable or bad.'
              if args.task == "quality" else
              'Predict scale age. Return JSON with prediction: 0, 1, 2, or 3. 3 means 3 or older.')

    def collate(examples):
        if len(examples) != 1:
            raise ValueError("This image collator uses micro-batch size one")
        row = examples[0]
        with Image.open(row["path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completed = processor.apply_chat_template(messages + [{"role": "assistant", "content": row["answer"]}],
                                                   tokenize=False, add_generation_prompt=False)
        batch = processor(text=[completed], images=[image], return_tensors="pt", padding=True)
        prefix_ids = processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"]
        n = prefix_ids.shape[1]
        if not torch.equal(batch["input_ids"][:, :n], prefix_ids):
            raise ValueError("Chat prefix is not token-aligned; refusing incorrect loss masking")
        labels = batch["input_ids"].clone()
        labels[:, :n] = -100
        labels[batch["attention_mask"].eq(0)] = -100
        if not labels.ne(-100).any():
            raise ValueError("No supervised answer tokens")
        batch["labels"] = labels
        return batch

    args.out.mkdir(parents=True)
    config = {**vars(args), "labels": str(args.labels), "out": str(args.out), "model": MODEL,
              "revision": REVISION, "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
              "system": system, "prompt": prompt, "selection_metric": "validation_answer_loss",
              "reasoning_supervision": False, "vision_encoder_frozen": True}
    (args.out / "training_config.json").write_text(json.dumps(config, indent=2))
    pd.concat([train, val]).to_csv(args.out / "used_rows.csv", index=False)
    training_args = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulation, fp16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, remove_unused_columns=False, label_names=["labels"],
        report_to="none", logging_steps=10, dataloader_num_workers=0, seed=args.seed)
    trainer = Trainer(model=model, args=training_args, data_collator=collate,
                      train_dataset=train.to_dict("records"), eval_dataset=val.to_dict("records"))
    trainer.train()
    trainer.save_model(str(args.out / "best_adapter"))
    processor.save_pretrained(args.out / "best_adapter")
    trainer.save_state()
    print("Saved best LoRA adapter:", args.out / "best_adapter")


if __name__ == "__main__":
    main()
