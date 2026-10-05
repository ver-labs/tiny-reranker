"""Start the scratch training loop from the saved initial state."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .collate import make_pair_loader
from .data import CleanPairDataset
from .model import ScratchConfig, ScratchReranker
from .scratch_tokenizer import load_scratch_tokenizer


ROOT = Path(__file__).resolve().parent.parent
INITIAL_CHECKPOINT = ROOT / "checkpoints" / "initial_seed42.pt"
INITIAL_MANIFEST = INITIAL_CHECKPOINT.with_suffix(".json")
TRAIN_BATCH_SIZE = 16


def load_training_model() -> tuple[ScratchReranker, torch.device]:
    """Restore initial random weights and enable training behavior."""
    manifest = json.loads(INITIAL_MANIFEST.read_text(encoding="utf-8"))
    digest = hashlib.sha256(INITIAL_CHECKPOINT.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise RuntimeError("Initial checkpoint does not match its manifest")

    config = ScratchConfig.from_file()
    saved = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=True)
    if saved["config"] != asdict(config):
        raise RuntimeError("Initial checkpoint config differs from current config")
    model = ScratchReranker(config)
    model.load_state_dict(saved["model_state_dict"], strict=True)

    requested_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(requested_device)
    device = next(model.parameters()).device
    model.train()
    return model, device


def make_training_loader(config: ScratchConfig):
    """Read only cleaned training pairs; keep loading in the main process."""
    dataset = CleanPairDataset("train")
    tokenizer = load_scratch_tokenizer()
    return make_pair_loader(
        dataset, tokenizer, batch_size=TRAIN_BATCH_SIZE,
        max_length=config.max_length, shuffle=False,
    )


def move_batch_to_device(batch: dict, device: torch.device):
    """Move model inputs and numeric labels; keep string IDs on the CPU."""
    inputs = batch["inputs"].to(device)
    labels = batch["labels"].to(device)
    return inputs, labels


def one_logit_per_label(raw_logits: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
    """Remove only a trailing score dimension, preserving batch size one."""
    if raw_logits.ndim == 2 and raw_logits.shape[1] == 1:
        logits = raw_logits.squeeze(-1)
    elif raw_logits.ndim == 1:
        logits = raw_logits
    else:
        raise ValueError(f"Expected [batch] or [batch, 1] logits, got {tuple(raw_logits.shape)}")
    if logits.shape != labels.shape:
        raise ValueError(f"Logit shape {tuple(logits.shape)} differs from labels {tuple(labels.shape)}")
    return logits


def main() -> None:
    model, device = load_training_model()
    assert model.training and all(module.training for module in model.modules())
    loader = make_training_loader(model.config)
    batch = next(iter(loader))
    inputs = batch["inputs"]
    assert loader.num_workers == 0
    assert inputs["input_ids"].shape[0] == TRAIN_BATCH_SIZE
    assert inputs["attention_mask"].shape == inputs["input_ids"].shape
    assert inputs["token_type_ids"].shape == inputs["input_ids"].shape
    assert inputs["input_ids"].dtype == torch.long
    assert batch["labels"].shape == (TRAIN_BATCH_SIZE,)
    assert batch["labels"].dtype == torch.float32
    assert len(batch["query_ids"]) == len(batch["document_ids"]) == len(batch["pair_ids"]) == TRAIN_BATCH_SIZE
    inputs, labels = move_batch_to_device(batch, device)
    assert all(tensor.device == device for tensor in inputs.values())
    assert labels.device == device
    assert next(model.parameters()).device == device
    model.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in model.parameters())
    initial_head = model.scoring_head.weight.detach().clone()
    logits = one_logit_per_label(model(**inputs), labels)
    assert logits.shape == (TRAIN_BATCH_SIZE,)
    assert bool(torch.isfinite(logits).all())
    assert logits.requires_grad
    one_item_inputs = {name: tensor[:1] for name, tensor in inputs.items()}
    # Keep training-mode gradient tracking; MPS cannot run attention dropout
    # through its no_grad scaled-dot-product-attention path.
    one_item_raw = model(**one_item_inputs)
    assert one_item_raw.shape == (1,)
    one_item_logits = one_logit_per_label(one_item_raw, labels[:1])
    assert one_item_logits.shape == (1,)
    loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
    assert loss.ndim == 0 and bool(torch.isfinite(loss)) and loss.requires_grad
    assert torch.equal(model.scoring_head.weight.detach(), initial_head)
    print(f"Device: {device}")
    print(f"Training mode enabled: {model.training}")
    print(f"Training pairs available: {len(loader.dataset):,}")
    print(f"Read one batch: {TRAIN_BATCH_SIZE} pairs, token shape {tuple(inputs['input_ids'].shape)}")
    print(f"Labels: {labels.cpu().tolist()}")
    print(f"IDs aligned: {len(batch['pair_ids'])} pair IDs; DataLoader workers: {loader.num_workers}")
    print(f"Model inputs and labels on: {device}; string IDs remain in Python")
    print("Old gradients cleared: all parameter gradients are None")
    print(f"Forward output: {tuple(logits.shape)} finite raw scores; first four: {logits.detach().cpu().tolist()[:4]}")
    print(f"One-item output keeps batch dimension: {tuple(one_item_logits.shape)}")
    print(f"Binary cross-entropy loss: {loss.item():.6f}")
    print("Weights unchanged; no backward pass or optimizer step run")


if __name__ == "__main__":
    main()
