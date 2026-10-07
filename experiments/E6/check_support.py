"""Smoke-test real scratch-reranker training operations on MPS by precision."""

import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import torch

from scripts.checkpoint_io import atomic_write_text
from scripts.collate import make_pair_loader
from scripts.data import iter_clean_pairs
from scripts.model import ScratchConfig, ScratchReranker
from scripts.scratch_tokenizer import load_scratch_tokenizer
from scripts.train import LEARNING_RATE, move_batch_to_device, one_logit_per_label


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL_FOLDER = ROOT / "experiments" / "E1" / "four_layer"
REPORT = HERE / "support.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_mode(name: str, dtype: torch.dtype | None, initial: dict,
             config: ScratchConfig, batch: dict) -> dict:
    """Try one real model forward, BCE backward, and AdamW update on MPS."""
    result = {"name": name, "autocast_dtype": str(dtype) if dtype else None,
              "supported_for_this_smoke_test": False}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            torch.manual_seed(123)
            device = torch.device("mps")
            model = ScratchReranker(config)
            model.load_state_dict(initial["model_state_dict"], strict=True)
            model = model.to(device).train()
            if any(parameter.device.type != "mps" for parameter in model.parameters()):
                raise RuntimeError("Model silently left MPS")
            inputs, labels = move_batch_to_device(batch, device)
            if labels.device.type != "mps" or \
                    any(tensor.device.type != "mps" for tensor in inputs.values()):
                raise RuntimeError("Input or label silently left MPS")
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            head_before = model.scoring_head.weight.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="mps", dtype=dtype, enabled=dtype is not None):
                logits = one_logit_per_label(model(**inputs), labels)
                loss = loss_fn(logits, labels)
            if logits.device.type != "mps" or loss.device.type != "mps":
                raise RuntimeError("Output or loss silently left MPS")
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite score or loss")
            scaler = None
            if dtype == torch.float16:
                scaler = torch.amp.GradScaler("mps")
                if not scaler.is_enabled():
                    raise RuntimeError("MPS FP16 GradScaler is disabled")
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
            else:
                loss.backward()
            gradients = [parameter.grad for parameter in model.parameters()
                         if parameter.requires_grad]
            if any(gradient is None for gradient in gradients):
                raise RuntimeError("A trainable parameter has no gradient")
            if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise RuntimeError("Nonfinite gradient")
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            torch.mps.synchronize()
            head_changed = not torch.equal(head_before, model.scoring_head.weight.detach())
            if not head_changed:
                raise RuntimeError("Optimizer did not change the scoring head")
            result.update({
                "supported_for_this_smoke_test": True,
                "model_parameter_dtype": str(next(model.parameters()).dtype),
                "logit_dtype": str(logits.dtype),
                "loss_dtype": str(loss.dtype),
                "gradient_dtype": str(gradients[0].dtype),
                "loss": float(loss.detach().item()),
                "finite_scores_loss_and_gradients": True,
                "head_changed_after_optimizer_step": True,
                "gradient_scaler": None if scaler is None else {
                    "class": type(scaler).__name__,
                    "enabled": scaler.is_enabled(),
                    "scale_before": scale_before,
                    "scale_after": scaler.get_scale(),
                },
            })
        except Exception as exc:
            result["failure"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
        result["warnings"] = [str(item.message)[:300] for item in caught]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Repeat and replace an existing local support report")
    args = parser.parse_args()
    if REPORT.exists() and not args.overwrite:
        raise FileExistsError(f"Support report already exists: {REPORT}")
    backend = {
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "device_selected": "mps" if torch.backends.mps.is_available() else None,
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
    }
    if not backend["mps_available"]:
        raise RuntimeError("MPS is unavailable; cannot report CPU checks as MPS support")
    if backend["pytorch_enable_mps_fallback"] == "1":
        raise RuntimeError("MPS CPU fallback is enabled; this check requires MPS-only execution")
    config_path = MODEL_FOLDER / "config.json"
    initial_path = MODEL_FOLDER / "initial.pt"
    initial_manifest = json.loads((MODEL_FOLDER / "initial.json").read_text())
    if sha256(config_path) != initial_manifest["config_sha256"] or \
            sha256(initial_path) != initial_manifest["initial_checkpoint_sha256"]:
        raise RuntimeError("E1 random initialization differs from its manifest")
    config = ScratchConfig.from_file(config_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial["config"] != asdict(config):
        raise RuntimeError("E1 initial checkpoint config mismatch")
    tokenizer = load_scratch_tokenizer(config_path)
    rows = list(islice(iter_clean_pairs("train"), 4))
    if len(rows) != 4:
        raise RuntimeError("Insufficient training examples for support check")
    batch = next(iter(make_pair_loader(rows, tokenizer, batch_size=4,
                                       max_length=config.max_length,
                                       padding="max_length", num_workers=0)))
    modes = [
        run_mode("fp32", None, initial, config, batch),
        run_mode("fp16_autocast", torch.float16, initial, config, batch),
        run_mode("bf16_autocast", torch.bfloat16, initial, config, batch),
    ]
    if not modes[0]["supported_for_this_smoke_test"]:
        raise RuntimeError(f"FP32 reference failed: {modes[0].get('failure')}")
    report = {
        "purpose": "E6 local precision support check; one optimizer step per mode, not an E6 training comparison",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "macos_version": platform.mac_ver()[0],
        "backend": backend,
        "model": "4-layer scratch reranker with saved random initial weights",
        "initial_checkpoint_sha256": sha256(initial_path),
        "config_sha256": sha256(config_path),
        "sample_pair_ids": batch["pair_ids"],
        "sample_batch_size": 4,
        "sample_padding": "fixed to 128 tokens",
        "operations": "token embeddings, position/segment embeddings, bidirectional encoder self-attention, layer normalization, GELU feed-forward, scoring head, BCEWithLogitsLoss, backward, AdamW update",
        "modes": modes,
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", REPORT,
                      replace_existing=args.overwrite)
    for mode in modes:
        status = "PASS" if mode["supported_for_this_smoke_test"] else "FAIL"
        print(f"{mode['name']}: {status}; output={mode.get('logit_dtype')}; "
              f"reason={mode.get('failure')}", flush=True)


if __name__ == "__main__":
    main()
