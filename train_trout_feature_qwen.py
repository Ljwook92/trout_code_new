"""Expert-GT training: frozen SimCLR ResNet18 -> projector -> Qwen language LoRA.

Qwen receives continuous feature embeddings, never pixel_values or teacher labels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from PIL import Image, ImageOps

from train_trout_vlm import MODEL, REVISION, augment_image, select_rows

PROTOCOL = "simclr_resnet18_feature_qwen_expert_gt_v1"
MARKER = "<<SCALE_FEATURES>>"
SYSTEM = (
    "Classify trout scales from learned numerical feature tokens. "
    "Return only the requested JSON class. Do not invent visual observations, "
    "annulus locations, damage descriptions, or explanations."
)
PROMPTS = {
    "quality": 'Predict usability. Return JSON with decision: readable or bad.',
    "age": 'Predict age. Return JSON with prediction: 0, 1, 2, or 3. 3 means 3 or older.',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_rows(table, task):
    train, val = select_rows(table, task)
    if task == "age":
        train = train.loc[train.age_label_source.eq("expert_direct")].copy()
    else:
        for rows in [train, val]:
            if "quality_source" not in rows or not rows.quality_source.isin(
                    ["expert_direct", "expert_quality_csv"]).all():
                raise ValueError("Quality labels must be image-specific expert GT")
    classes = set(range(4)) if task == "age" else {"readable", "bad"}
    for rows in [train, val]:
        observed = set(rows.age4) if task == "age" else set(rows.quality_gt)
        if observed != classes:
            raise ValueError("Every target class needs expert GT in train and validation")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def check_encoder_provenance(table, run, checkpoint):
    manifest = pd.read_csv(run / "split_manifest.csv")
    settings = json.loads((run / "search_config.json").read_text())
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    if checkpoint.get("search_signature") != signature:
        raise ValueError("Encoder checkpoint does not belong to --encoder-run")
    digest = hashlib.sha256(pd.util.hash_pandas_object(
        manifest.sort_values("scale_id"), index=False).values.tobytes()).hexdigest()
    if digest != settings.get("split_sha256"):
        raise ValueError("Encoder split manifest changed")
    if (manifest.scale_id.isna().any() or manifest.scale_id.duplicated().any()
            or manifest.fish_key.isna().any()
            or not manifest["split"].isin(["train", "validation", "test"]).all()
            or manifest.groupby("fish_key")["split"].nunique().gt(1).any()):
        raise ValueError("Invalid encoder fish split")
    fish_split = manifest.drop_duplicates("fish_key").set_index("fish_key")["split"]
    assigned = table.loc[table["split"].isin(["train", "validation", "test"])]
    if not assigned.fish_key.map(fish_split).eq(assigned["split"]).all():
        raise ValueError("Current fish partitions differ from the encoder training partitions")
    overlap = manifest.merge(table[["scale_id", "fish_key", "split"]], on="scale_id",
                             how="left", suffixes=("_encoder", ""), validate="one_to_one")
    if (not overlap.fish_key.eq(overlap.fish_key_encoder).all()
            or not overlap["split"].eq(overlap.split_encoder).all()):
        raise ValueError("Encoder/current image identities or partitions differ")
    if checkpoint.get("cohort_sha256", settings["cohort_sha256"]) != settings["cohort_sha256"]:
        raise ValueError("Encoder cohort changed")
    return {"search_signature": signature, "encoder_split_sha256": digest,
            "encoder_manifest_sha256": sha(run / "split_manifest.csv")}


def backbone_state(checkpoint):
    if checkpoint.get("backbone", "resnet18") != "resnet18" or checkpoint.get("feature_dim", 512) != 512:
        raise ValueError("This script requires the existing ResNet18 encoder")
    if "backbone_state_dict" in checkpoint:
        return checkpoint["backbone_state_dict"]
    if "model_state_dict" in checkpoint:
        state = {k[len("backbone."):]: v for k, v in checkpoint["model_state_dict"].items()
                 if k.startswith("backbone.")}
        if state:
            return state
    raise ValueError("Expected notebook backbone_state_dict or model_state_dict with backbone.*")


class FeatureProjector(nn.Module):
    def __init__(self, hidden_size, grid=7):
        super().__init__()
        self.tokens = 1 if grid == 1 else 1 + grid * grid
        self.net = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden_size),
                                 nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.positions = nn.Parameter(torch.zeros(1, self.tokens, hidden_size))
        nn.init.normal_(self.positions, std=0.01)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1:] != (self.tokens, 512):
            raise ValueError("Unexpected feature token shape")
        return self.net(features.float()) + self.positions


class FrozenScaleEncoder(nn.Module):
    def __init__(self, state, grid):
        super().__init__()
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        backbone.load_state_dict(state, strict=True)
        self.layers = nn.Sequential(*list(backbone.children())[:-2])
        self.grid = grid
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, images):
        self.eval()  # BatchNorm statistics must stay frozen during Qwen training.
        maps = self.layers(images)
        global_token = maps.mean(dim=(2, 3)).unsqueeze(1)
        if self.grid == 1:
            return global_token
        spatial = nn.functional.adaptive_avg_pool2d(maps, self.grid).flatten(2).transpose(1, 2)
        return torch.cat([global_token, spatial], dim=1)


def answer_tokens(tokenizer, task, answer):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": MARKER + "\n" + PROMPTS[task]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    completed = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False)
    before, suffix_prefix = prefix.split(MARKER)
    completed_before, suffix = completed.split(MARKER)
    if before != completed_before:
        raise ValueError("Chat template changed prefix")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix_ids, tail_ids = encode(suffix_prefix), encode(suffix)
    if tail_ids[:len(prefix_ids)] != prefix_ids or len(tail_ids) <= len(prefix_ids):
        raise ValueError("Response loss mask is not token aligned")
    return encode(before), tail_ids, len(prefix_ids)


def feature_batch(model, projector, features, tokens):
    before, tail, prefix_len = tokens
    device = features.device
    embed = model.get_input_embeddings()
    left = embed(torch.tensor([before], device=device))
    right = embed(torch.tensor([tail], device=device))
    feature_tokens = projector(features).to(left.dtype)
    inputs = torch.cat([left, feature_tokens, right], dim=1)
    labels = torch.full((1, inputs.shape[1]), -100, device=device, dtype=torch.long)
    offset = len(before) + projector.tokens
    labels[:, offset + prefix_len:] = torch.tensor([tail[prefix_len:]], device=device)
    # Explicit text-style RoPE bypasses Qwen's image-grid position construction.
    positions = torch.arange(inputs.shape[1], device=device).view(1, 1, -1).expand(3, 1, -1)
    return {"inputs_embeds": inputs, "labels": labels,
            "attention_mask": torch.ones_like(labels), "position_ids": positions,
            "use_cache": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-run", type=Path, required=True,
                        help="Notebook output directory containing split_manifest.csv and search_config.json")
    parser.add_argument("--task", choices=["quality", "age"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--projector-lr", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--grid", type=int, choices=[1, 7], default=7,
                        help="1: pooled vector only; 7: global vector plus 49 spatial vectors")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--augment", action="store_true", help="Train-only quarter rotation and contrast; no crop")
    parser.add_argument("--contrast-jitter", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.epochs < 1 or args.accumulation < 1 or args.rank < 1
            or not all(math.isfinite(x) and x > 0 for x in [args.lr, args.projector_lr])
            or not math.isfinite(args.contrast_jitter) or not 0 <= args.contrast_jitter < 1):
        raise ValueError("Invalid training parameters")
    table = pd.read_csv(args.labels)
    train, val = expert_rows(table, args.task)
    checkpoint = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
    provenance = check_encoder_provenance(table, args.encoder_run, checkpoint)
    state = backbone_state(checkpoint)
    encoder = FrozenScaleEncoder(state, args.grid)
    for rows in [train, val]:
        missing = rows.loc[~rows.path.map(lambda p: Path(str(p)).is_file()), "path"]
        if len(missing):
            raise FileNotFoundError(str(missing.iloc[0]))
    print(f"task={args.task} expert train={len(train)} validation={len(val)}; test unused", flush=True)
    print(train.answer.value_counts().to_string(), flush=True)
    if args.dry_run:
        print("Dry run passed: checkpoint/splits/GT/images checked; no Qwen loaded or output written")
        return
    if args.out.exists():
        raise FileExistsError("Choose a new output directory; existing runs are preserved")
    if not torch.cuda.is_available():
        raise RuntimeError("An allocated CUDA GPU is required")

    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration, set_seed
    from peft import LoraConfig, get_peft_model
    from torchvision import transforms as T
    from tqdm.auto import tqdm

    set_seed(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, attn_implementation="sdpa")
    hidden = base.get_input_embeddings().weight.shape[1]
    targets = [name for name, module in base.named_modules()
               if name.endswith(("q_proj", "v_proj")) and "visual" not in name
               and isinstance(module, nn.Linear)]
    if not targets:
        raise RuntimeError("No language LoRA targets found")
    model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=args.rank * 2,
                            lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda")
    # Keep unused Qwen vision weights off the GPU; they never receive images.
    base.model.visual.to("cpu")
    projector = FeatureProjector(hidden, args.grid).to("cuda")
    encoder.to("cuda")
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tokens = {answer: answer_tokens(tokenizer, args.task, answer)
              for answer in set(train.answer) | set(val.answer)}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
        {"params": projector.parameters(), "lr": args.projector_lr}], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    args.out.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"protocol": PROTOCOL, "model": MODEL, "revision": REVISION,
        "labels_sha256": sha(args.labels), "encoder_sha256": sha(args.encoder_checkpoint),
        **provenance, "hidden_size": hidden, "feature_tokens": projector.tokens,
        "supervision": "expert_GT_only_response_cross_entropy", "pseudo_labels": False,
        "distillation": False, "qwen_pixel_input": False, "encoder_frozen": True,
        "uses_length_weight": False, "image_size": 224, "crop": False,
        "selection_metric": "validation_answer_loss", "reasoning_supervision": False,
        "system": SYSTEM, "prompt": PROMPTS[args.task], "dtype": str(dtype)})
    (args.out / "training_config.json").write_text(json.dumps(config, indent=2))
    pd.concat([train, val]).to_csv(args.out / "used_rows.csv", index=False)
    # Package encoder weights so inference does not depend on a mutable external checkpoint.
    torch.save({"backbone_state_dict": state}, args.out / "encoder.pt")
    del checkpoint, state

    def loss_for(row, training):
        with Image.open(row.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        image = augment_image(image, "train" if training else "validation", args.augment,
                              rotation_degrees=0, brightness_jitter=0,
                              contrast_jitter=args.contrast_jitter)
        features = encoder(transform(image).unsqueeze(0).to("cuda"))
        with torch.autocast("cuda", dtype=dtype):
            loss = model(**feature_batch(model, projector, features, tokens[row.answer])).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for {row.scale_id}")
        return loss

    best_loss, history = float("inf"), []
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    for epoch in range(args.epochs):
        model.train()
        projector.train()
        encoder.eval()
        shuffled = train.sample(frac=1, random_state=args.seed + epoch).reset_index(drop=True)
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, row in enumerate(tqdm(shuffled.itertuples(index=False), total=len(train),
                                     desc=f"{args.task} {epoch + 1}/{args.epochs}")):
            # The final partial accumulation group gets its actual size, not the full configured size.
            group_start = (i // args.accumulation) * args.accumulation
            group_size = min(args.accumulation, len(train) - group_start)
            loss = loss_for(row, True)
            total += loss.item()
            scaler.scale(loss / group_size).backward()
            if (i + 1) % args.accumulation == 0 or i + 1 == len(train):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        projector.eval()
        with torch.no_grad():
            val_loss = sum(loss_for(row, False).item() for row in tqdm(
                val.itertuples(index=False), total=len(val), desc="validation")) / len(val)
        record = {"epoch": epoch + 1, "train_loss": total / len(train), "validation_loss": val_loss}
        history.append(record)
        print(json.dumps(record), flush=True)
        pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
        if val_loss < best_loss:
            best_loss = val_loss
            bundle = args.out / "best_bundle"
            model.save_pretrained(bundle / "qwen_adapter")
            tokenizer.save_pretrained(bundle / "qwen_adapter")
            torch.save(projector.state_dict(), bundle / "projector.pt")
            (bundle / "selection.json").write_text(json.dumps(record, indent=2))
    print("Saved best feature-conditioned model:", args.out / "best_bundle", flush=True)
    print("Load encoder.pt + projector.pt + qwen_adapter together; old image LoRAs are not used.")


if __name__ == "__main__":
    main()
