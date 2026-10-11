# E18 reproduction

Run from the tiny-reranker root. This is the user-authorized smaller first study in `plan.json`, not the original 147,456-request protocol. Assets require about 100 MB; results and compiler caches add space.

## Required local inputs

Use Python 3.12 and Rust/Cargo. Install `requirements.txt` for the Apple
workflow; the recorded complete environments are `apple_requirements.txt`
and `nvidia_requirements.txt` in this directory.

The historical training pipeline is archived under
[`archive/pre-inference-cleanup-2026-10-11`](https://github.com/ver-labs/tiny-reranker/tree/archive/pre-inference-cleanup-2026-10-11).
E18 now owns its ranking metrics and request schema; no training modules are
needed at runtime. Original data-preparation manifests are retained unchanged
in `provenance/`.

The candidate fixture `data/e16/development.jsonl` and original SciFact/NFCorpus
corpora/qrels under `data/beir/` must already be available with the hashes in
`data_manifest.json`. They are not included in a Git clone, and `prepare`
verifies these inputs rather than downloading or regenerating them. Full
saved-evidence verification also needs the separately retained score audits
and HTTP response bodies listed in `checksums.json`; these large files are
excluded from Git. Do not substitute newly retrieved candidates and present
them as the original study.

For the focused test suite:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cargo build --release --manifest-path experiments/E18/rust-server/Cargo.toml
.venv/bin/python -m pytest tests -q
```

```bash
.venv/bin/python -m experiments.E18.download_model
.venv/bin/python -m experiments.E18.prepare
.venv/bin/python -m pytest tests/test_e18_contract.py -q
cargo test --manifest-path experiments/E18/rust-server/Cargo.toml
cargo build --release --manifest-path experiments/E18/rust-server/Cargo.toml
.venv/bin/python -u -m experiments.E18.launch --device mps
.venv/bin/python -u -m experiments.E18.finish_device --device mps
.venv/bin/python -m experiments.E18.summarize
.venv/bin/python -m experiments.E18.verify
.venv/bin/python -m experiments.E18.plots
.venv/bin/python -m experiments.E18.report
```

GPU execution must be outside a filesystem sandbox that disables Metal. The launcher refuses CPU substitution, executes one benchmark at a time on each GPU, resumes completed artifacts, and starts a fresh child process for each path/session. Stock accuracy must precede candidate accuracy. Compilation failures are retained. Existing partial JSON-lines score artifacts can resume with identical hashes and record ordering. Completed timing cells are only reused when the complete path/session artifact exists; partial sessions are retained in exclusion records before rerunning.

For NVIDIA, copy E18 sources/plan/manifests to the existing approved cluster project, install the pinned dependencies already used in E17, download the immutable assets and verify hashes there, then:

```bash
/opt/slurm/current/bin/sbatch experiments/E18/kernel_preflight.sbatch
# Only after kernel correctness and compilation preflight complete:
/opt/slurm/current/bin/sbatch experiments/E18/nvidia.sbatch
```

CUDA build uses NVIDIA's checksum-verified project-local nvcc 12.6.85, architecture sm_89, FP32, FMA disabled, precise division/square root, and static CUDA runtime. The old cluster module was unavailable; compiler/runtime archives are pinned by the official redistribution manifest and archive SHA-256 values in cuda_compiler_manifest.json. Do not run directly on the login node's CPU or substitute another GPU.

Optional isolated diagnostic commands, run sequentially after the main launcher:

```bash
.venv/bin/python -m experiments.E18.profile --device mps --path stock
.venv/bin/python -m experiments.E18.profile --device mps --path mlx
.venv/bin/python -m experiments.E18.operation_benchmark --device mps --session 0
```

Repeat operation sessions with IDs 0–5. The timing scopes are explicit in every observation. Apple operation timing is synchronized amortized wall time, including dispatch; it is not claimed to be isolated GPU event latency. CUDA operation timing uses device events over 100 launches and includes GPU-visible launch gaps. Compiler first use has unknown cache state unless an isolated directory was explicitly assigned; compilation cost is excluded from warm timing.
