"""Check whether the scratch reranker can memorize 32 fixed training pairs."""

import argparse
from itertools import islice

import torch

from .collate import PairCollator
from .data import iter_clean_pairs
from .scratch_tokenizer import load_scratch_tokenizer
from .train import LEARNING_RATE, load_training_model, move_batch_to_device, one_logit_per_label


SAMPLE_SIZE = 32
BATCH_SIZE = 16


def measure(model, batches, loss_fn):
    """Score all 32 examples with dropout off and gradients disabled."""
    was_training = model.training
    model.eval()
    try:
        total_loss = 0.0
        correct = 0
        with torch.inference_mode():
            for inputs, labels in batches:
                logits = one_logit_per_label(model(**inputs), labels)
                total_loss += float(loss_fn(logits, labels).item()) * len(labels)
                correct += int(((logits >= 0) == (labels == 1)).sum().item())
        return total_loss / SAMPLE_SIZE, correct / SAMPLE_SIZE
    finally:
        model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    torch.manual_seed(123)
    model, device = load_training_model()
    tokenizer = load_scratch_tokenizer()
    rows = list(islice(iter_clean_pairs("train"), SAMPLE_SIZE))
    if len(rows) != SAMPLE_SIZE or {row["label"] for row in rows} != {0, 1}:
        raise RuntimeError("Need 32 fixed training pairs containing both labels")
    collate = PairCollator(tokenizer, model.config.max_length)
    batches = [move_batch_to_device(collate(rows[start:start + BATCH_SIZE]), device)
               for start in range(0, SAMPLE_SIZE, BATCH_SIZE)]
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    initial_loss, initial_accuracy = measure(model, batches, loss_fn)
    print(f"Device: {device}; fixed training pairs: {SAMPLE_SIZE}; "
          f"batch size: {BATCH_SIZE}; max steps: {args.max_steps}", flush=True)
    print(f"step=0 sample_loss={initial_loss:.4f} sample_accuracy={initial_accuracy:.1%}", flush=True)
    final_loss, final_accuracy = initial_loss, initial_accuracy
    for step in range(1, args.max_steps + 1):
        model.train()
        inputs, labels = batches[(step - 1) % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        logits = one_logit_per_label(model(**inputs), labels)
        loss = loss_fn(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite training loss at step {step}")
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == args.max_steps:
            final_loss, final_accuracy = measure(model, batches, loss_fn)
            print(f"step={step} sample_loss={final_loss:.4f} "
                  f"sample_accuracy={final_accuracy:.1%}", flush=True)
            if final_loss < 0.1 and final_accuracy == 1.0:
                print("Tiny-sample fit confirmed; stopping early.", flush=True)
                break
    print(f"Final: loss {initial_loss:.4f} -> {final_loss:.4f}; "
          f"accuracy {initial_accuracy:.1%} -> {final_accuracy:.1%}", flush=True)
    print("Diagnostic only: no checkpoint saved and no validation or test data used.", flush=True)


if __name__ == "__main__":
    main()
