"""Export and verify a self-contained scratch reranker inference bundle."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import BertTokenizer

from .model import ScratchConfig, ScratchReranker
from .scratch_tokenizer import encode_pairs, load_scratch_tokenizer
from .short_epoch import CHECKPOINT, REPORT, ROOT


BUNDLE = ROOT / "artifacts" / "short_epoch_1024"
WEIGHTS = "model.pt"
MODEL_CONFIG = "model_config.json"
MANIFEST = "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bundle(path: Path = BUNDLE, device: torch.device | None = None):
    """Load model and tokenizer from local bundle files only."""
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    for filename, expected_hash in manifest["file_sha256"].items():
        if sha256(path / filename) != expected_hash:
            raise RuntimeError(f"Inference bundle file changed: {filename}")
    config = ScratchConfig(**json.loads((path / MODEL_CONFIG).read_text(encoding="utf-8")))
    tokenizer = BertTokenizer.from_pretrained(str(path / "tokenizer"), local_files_only=True)
    if tokenizer.vocab_size != config.vocab_size or tokenizer.pad_token_id != config.pad_token_id:
        raise RuntimeError("Local tokenizer does not match the model configuration")
    model = ScratchReranker(config)
    model.load_state_dict(torch.load(path / WEIGHTS, map_location="cpu", weights_only=True), strict=True)
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(device).eval(), tokenizer, device


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if sha256(CHECKPOINT) != report["checkpoint_sha256"]:
        raise RuntimeError("Trained checkpoint differs from the saved run report")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if saved["optimizer_steps"] != report["optimizer_steps"]:
        raise RuntimeError("Checkpoint step count differs from the saved run report")
    config = ScratchConfig.from_file()
    tokenizer = load_scratch_tokenizer()
    BUNDLE.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = BUNDLE / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer.save_pretrained(str(tokenizer_dir))
    (BUNDLE / MODEL_CONFIG).write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    torch.save(saved["model_state_dict"], BUNDLE / WEIGHTS)

    files = [BUNDLE / MODEL_CONFIG, BUNDLE / WEIGHTS, *sorted(tokenizer_dir.iterdir())]
    manifest = {
        "model": "scratch_encoder_seed42_short_epoch_1024",
        "source_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256": report["checkpoint_sha256"],
        "optimizer_steps": report["optimizer_steps"],
        "tokenizer_source_id": json.loads((ROOT / "configs" / "scratch.json").read_text())["tokenizer_id"],
        "tokenizer_source_revision": json.loads((ROOT / "configs" / "scratch.json").read_text())["tokenizer_revision"],
        "input_format": "[CLS] query [SEP] document [SEP]",
        "max_length": config.max_length,
        "precision": "float32",
        "file_sha256": {str(file.relative_to(BUNDLE)): sha256(file) for file in files},
    }
    (BUNDLE / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    model, local_tokenizer, device = load_bundle()
    original = ScratchReranker(config)
    original.load_state_dict(saved["model_state_dict"], strict=True)
    original = original.to(device).eval()
    queries = ["What is a reranker?", "What is a reranker?"]
    documents = ["It scores query-document pairs.", "A recipe for soup."]
    original_inputs = encode_pairs(tokenizer, queries, documents, max_length=config.max_length)
    local_inputs = encode_pairs(local_tokenizer, queries, documents, max_length=config.max_length)
    if any(not torch.equal(original_inputs[key], local_inputs[key]) for key in original_inputs):
        raise RuntimeError("Exported tokenizer changes the tokenized inputs")
    with torch.inference_mode():
        expected = original(**original_inputs.to(device))
        actual = model(**local_inputs.to(device))
    if not torch.allclose(expected, actual, atol=1e-6, rtol=0):
        raise RuntimeError("Exported model changes inference scores")
    print(f"Inference bundle: {BUNDLE}")
    print(f"Device: {device}; local tokenizer and {len(model.state_dict())} model tensors reloaded")
    print(f"Input token IDs and scores match source checkpoint: {actual.cpu().tolist()}")
    print(f"Verified {len(manifest['file_sha256'])} bundle file hashes")


if __name__ == "__main__":
    main()
