"""Start the scratch training loop from the saved initial state."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .model import ScratchConfig, ScratchReranker


ROOT = Path(__file__).resolve().parent.parent
INITIAL_CHECKPOINT = ROOT / "checkpoints" / "initial_seed42.pt"
INITIAL_MANIFEST = INITIAL_CHECKPOINT.with_suffix(".json")


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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.train()
    return model, device


def main() -> None:
    model, device = load_training_model()
    assert model.training and all(module.training for module in model.modules())
    print(f"Device: {device}")
    print(f"Training mode enabled: {model.training}")
    print("Initial random weights restored; no batch or optimizer step run")


if __name__ == "__main__":
    main()
