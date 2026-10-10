# E18 reproduction

Run from the tiny-reranker root. This is the user-authorized smaller first study in `plan.json`, not the original 147,456-request protocol. Assets require about 100 MB; results and compiler caches add space.

```bash
.venv/bin/python -m experiments.E18.download_model
.venv/bin/python -m experiments.E18.prepare
.venv/bin/python -m pytest tests/test_e18_contract.py -q
cargo test --manifest-path experiments/E18/rust-server/Cargo.toml
cargo build --release --manifest-path experiments/E18/rust-server/Cargo.toml
.venv/bin/python -u -m experiments.E18.launch --device mps
```

GPU execution must be outside a filesystem sandbox that disables Metal. The launcher refuses CPU substitution, executes one benchmark at a time on each GPU, resumes completed artifacts, and starts a fresh child process for each path/session. Stock accuracy must precede candidate accuracy. Compilation failures are retained. Existing partial JSON-lines score artifacts can resume with identical hashes and record ordering. Completed timing cells are only reused when the complete path/session artifact exists; partial sessions are retained in exclusion records before rerunning.

For NVIDIA, copy E18 sources/plan/manifests to the existing approved cluster project, install the pinned dependencies already used in E17, download the immutable assets and verify hashes there, then:

```bash
/opt/slurm/current/bin/sbatch experiments/E18/kernel_preflight.sbatch
# Only after kernel correctness and compilation preflight complete:
/opt/slurm/current/bin/sbatch experiments/E18/nvidia.sbatch
```

CUDA build uses NVHPC's nvcc, architecture sm_89, FP32, FMA disabled, precise division/square root, and static CUDA runtime. Module initialization is cluster-specific and recorded in build logs. Do not run directly on the login node's CPU or substitute another GPU.

Optional isolated diagnostic commands, run sequentially after the main launcher:

```bash
.venv/bin/python -m experiments.E18.profile --device mps --path stock
.venv/bin/python -m experiments.E18.profile --device mps --path mlx
.venv/bin/python -m experiments.E18.operation_benchmark --device mps --session 0
```

Repeat operation sessions with IDs 0–5. The timing scopes are explicit in every observation. Apple operation timing is synchronized amortized wall time, including dispatch; it is not claimed to be isolated GPU event latency. CUDA operation timing uses device events over 100 launches and includes GPU-visible launch gaps. Compiler first use has unknown cache state unless an isolated directory was explicitly assigned; compilation cost is excluded from warm timing.
