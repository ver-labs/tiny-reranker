"""Save and verify the scratch model's untrained, seeded starting weights."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint_io import atomic_torch_save, atomic_write_text
from .model import DEFAULT_CONFIG, ScratchConfig, ScratchReranker


SEED = 42
CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / f"initial_seed{SEED}.pt"
MANIFEST = CHECKPOINT.with_suffix(".json")


def main() -> None:
    raw_config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config = ScratchConfig.from_file()
    torch.manual_seed(SEED)
    model = ScratchReranker(config)
    checkpoint = {
        "format_version": 1,
        "seed": SEED,
        "config": asdict(config),
        "tokenizer_id": raw_config["tokenizer_id"],
        "tokenizer_revision": raw_config["tokenizer_revision"],
        "model_state_dict": model.state_dict(),
    }

    CHECKPOINT.parent.mkdir(exist_ok=True)
    if not CHECKPOINT.exists():
        atomic_torch_save(checkpoint, CHECKPOINT)

    restored = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    for key in ("format_version", "seed", "config", "tokenizer_id", "tokenizer_revision"):
        if restored[key] != checkpoint[key]:
            raise RuntimeError(f"Saved checkpoint has different {key}")
    restored_model = ScratchReranker(config)
    restored_model.load_state_dict(restored["model_state_dict"], strict=True)
    if not all(
        torch.equal(value, restored_model.state_dict()[name])
        for name, value in model.state_dict().items()
    ):
        raise RuntimeError("Saved weights differ from the seeded initial model")

    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    manifest = {
        "checkpoint": CHECKPOINT.name,
        "sha256": digest,
        "seed": SEED,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "torch_version": str(torch.__version__),
        "config": raw_config,
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text(encoding="utf-8") != manifest_text:
        raise RuntimeError("Existing checkpoint manifest differs from the saved state")
    if not MANIFEST.exists():
        atomic_write_text(manifest_text, MANIFEST)

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"SHA-256: {digest}")
    print(f"Parameters: {manifest['parameter_count']:,}")
    print("Reloaded weights match the seeded initial model")


if __name__ == "__main__":
    main()
