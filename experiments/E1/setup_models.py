"""Prepare independent 2-layer and 4-layer scratch starting states for E1."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_torch_save, atomic_write_text
from scripts.model import ScratchConfig, ScratchReranker
from scripts.scratch_tokenizer import encode_pairs, load_scratch_tokenizer


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
SEED = 42
VARIANTS = {"two_layer": 2, "four_layer": 4}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    configs = {name: json.loads((EXPERIMENT / name / "config.json").read_text())
               for name in VARIANTS}
    for key in configs["two_layer"]:
        if key != "num_layers" and configs["two_layer"][key] != configs["four_layer"][key]:
            raise RuntimeError(f"E1 configurations differ beyond encoder depth: {key}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    original_four = torch.load(ROOT / "checkpoints" / "initial_seed42.pt",
                               map_location="cpu", weights_only=True)["model_state_dict"]

    for name, expected_layers in VARIANTS.items():
        folder = EXPERIMENT / name
        config_path = folder / "config.json"
        config = ScratchConfig.from_file(config_path)
        if config.num_layers != expected_layers:
            raise RuntimeError(f"Unexpected layer count for {name}")
        tokenizer = load_scratch_tokenizer(config_path)
        torch.manual_seed(SEED)
        model = ScratchReranker(config)
        state = {key: value.detach().cpu().clone()
                 for key, value in model.state_dict().items()}
        if name == "four_layer" and not all(
            torch.equal(value, original_four[key]) for key, value in state.items()
        ):
            raise RuntimeError("Four-layer initialization differs from saved seed-42 reference")
        checkpoint = folder / "initial.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["config"] != asdict(config) or saved["seed"] != SEED or not all(
                torch.equal(value, saved["model_state_dict"][key])
                for key, value in state.items()
            ):
                raise RuntimeError(f"Existing {name} initial state differs from seed-42 setup")
        else:
            atomic_torch_save({"config": asdict(config), "seed": SEED,
                               "model_state_dict": state}, checkpoint)
        inputs = encode_pairs(tokenizer, ["What is a reranker?"],
                              ["It scores query-document pairs."],
                              max_length=config.max_length).to(device)
        with torch.inference_mode():
            scores = model.to(device).eval()(**inputs)
        if scores.shape != (1,) or not bool(torch.isfinite(scores).all()):
            raise RuntimeError(f"{name} dummy forward pass failed")
        manifest = {
            "variant": name,
            "seed": SEED,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "config_sha256": sha256(config_path),
            "initial_checkpoint": checkpoint.name,
            "initial_checkpoint_sha256": sha256(checkpoint),
            "tokenizer_id": configs[name]["tokenizer_id"],
            "tokenizer_revision": configs[name]["tokenizer_revision"],
            "setup_backend": device.type,
            "trained": False,
        }
        manifest_path = folder / "initial.json"
        content = json.dumps(manifest, indent=2) + "\n"
        if manifest_path.exists():
            if manifest_path.read_text(encoding="utf-8") != content:
                raise RuntimeError(f"Existing {name} manifest differs from setup")
        else:
            atomic_write_text(content, manifest_path)
        print(f"{name}: {config.num_layers} layers, "
              f"{manifest['parameter_count']:,} parameters, "
              f"finite dummy score on {device}; no training", flush=True)


if __name__ == "__main__":
    main()
