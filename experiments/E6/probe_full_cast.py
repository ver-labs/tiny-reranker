"""Probe low-precision model weights without autocast on the actual reranker."""

import hashlib
import json
import os
import platform
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
E1 = ROOT / "experiments" / "E1" / "four_layer"
REPORT = HERE / "full_cast_probe.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def try_mode(name: str, dtype: torch.dtype, initial: dict,
             config: ScratchConfig, batch: dict) -> dict:
    result = {"precision": name, "autocast_used": False,
              "model_weight_dtype": str(dtype), "one_update_completed": False}
    try:
        torch.manual_seed(123)
        model = ScratchReranker(config)
        model.load_state_dict(initial["model_state_dict"], strict=True)
        model = model.to(device="mps", dtype=dtype).train()
        inputs, labels = move_batch_to_device(batch, torch.device("mps"))
        if labels.device.type != "mps" or \
                any(tensor.device.type != "mps" for tensor in inputs.values()):
            raise RuntimeError("An input or label left MPS")
        if any(parameter.device.type != "mps" or parameter.dtype != dtype
               for parameter in model.parameters()):
            raise RuntimeError("A parameter has the wrong device or dtype")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.amp.GradScaler("mps") if dtype == torch.float16 else None
        if scaler is not None and not scaler.is_enabled():
            raise RuntimeError("MPS GradScaler is disabled")
        optimizer.zero_grad(set_to_none=True)
        logits = one_logit_per_label(model(**inputs), labels)
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite output or loss")
        result.update({"logit_dtype": str(logits.dtype), "loss_dtype": str(loss.dtype),
                       "loss": float(loss.detach().item()),
                       "gradient_scaler": None if scaler is None else "torch.amp.GradScaler('mps')"})
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters()):
            raise RuntimeError("Missing or nonfinite gradient")
        head_before = model.scoring_head.weight.detach().clone()
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        torch.mps.synchronize()
        if torch.equal(head_before, model.scoring_head.weight.detach()):
            raise RuntimeError("Optimizer did not change scoring-head weights")
        result["one_update_completed"] = True
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"Full-cast probe report already exists: {REPORT}")
    if not torch.backends.mps.is_available() or \
            os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("MPS without CPU fallback is required")
    config_path = E1 / "config.json"
    initial_path = E1 / "initial.pt"
    manifest = json.loads((E1 / "initial.json").read_text(encoding="utf-8"))
    if sha256(config_path) != manifest["config_sha256"] or \
            sha256(initial_path) != manifest["initial_checkpoint_sha256"]:
        raise RuntimeError("E1 initial state differs from its manifest")
    config = ScratchConfig.from_file(config_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial["config"] != asdict(config):
        raise RuntimeError("Initial config mismatch")
    tokenizer = load_scratch_tokenizer(config_path)
    rows = list(islice(iter_clean_pairs("train"), 4))
    batch = next(iter(make_pair_loader(rows, tokenizer, batch_size=4,
                                       max_length=config.max_length)))
    modes = [try_mode("fp16", torch.float16, initial, config, batch),
             try_mode("bf16", torch.bfloat16, initial, config, batch)]
    report = {
        "experiment": "E6 full model cast without autocast; separate control from autocast comparison",
        "torch_version": torch.__version__,
        "macos_version": platform.mac_ver()[0],
        "backend": "mps",
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "initial_checkpoint_sha256": sha256(initial_path),
        "sample_pair_ids": batch["pair_ids"],
        "sample_batch_size": 4,
        "modes": modes,
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", REPORT)
    for mode in modes:
        status = "PASS" if mode["one_update_completed"] else "FAIL"
        print(f"full-cast {mode['precision']}: {status}; {mode.get('failure')}", flush=True)


if __name__ == "__main__":
    main()
