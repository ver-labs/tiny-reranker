"""Check the isolated Apple Triton runtime and fused kernel on the real GPU."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import traceback

import mlx.core as mx
import numpy as np

from experiments.E13.apple_runtime import residual_layer_norm


def main():
    report = {
        "status": "failed", "gpu_correctness_verified": False,
        "performance_measured": False, "route": "triton-msl documented MLX interface",
        "platform": platform.platform(), "python": platform.python_version(),
        "device": mx.device_info(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("triton", "triton-msl", "mlx", "torch", "numpy")},
        "checks": [], "limitations": "Kernel-only checks; epoch-3 model parity and latency not measured.",
    }
    try:
        compiler = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, text=True)
        report["metal_compiler"] = {"returncode": compiler.returncode, "stdout": compiler.stdout,
                                    "stderr": compiler.stderr}
        if compiler.returncode:
            report["status"] = "blocked_missing_metal_toolchain"
            raise RuntimeError("Metal compiler missing; install compatible Xcode and Metal Toolchain first")
        rng = np.random.default_rng(42)
        for warps in (1, 2, 4):
            for shape in ((1, 128), (5, 128), (2, 37, 128), (2, 128, 128), (128, 128, 128)):
                r, x = [rng.normal(size=shape).astype(np.float32) for _ in range(2)]
                w, b = [rng.normal(size=(128,)).astype(np.float32) for _ in range(2)]
                s, y = residual_layer_norm(*map(mx.array, (r, x, w, b)), num_warps=warps)
                mx.eval(s, y)
                mx.synchronize()
                summed = r + x
                f64 = summed.astype(np.float64)
                expected = (f64 - f64.mean(-1, keepdims=True)) / np.sqrt(f64.var(-1, keepdims=True) + 1e-5)
                expected = expected * w + b
                np.testing.assert_array_equal(np.array(s), summed)
                np.testing.assert_allclose(np.array(y), expected, atol=2e-5, rtol=2e-5)
                record = {"shape": shape, "num_warps": warps,
                          "max_abs_error_float64": float(np.max(np.abs(np.array(y) - expected)))}
                report["checks"].append(record)
                print(record, flush=True)
        report.update(status="apple_gpu_kernel_verified", gpu_correctness_verified=True)
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    output = Path(__file__).with_name("apple_setup_status.json")
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"{report['status']}: {output}", flush=True)
    return 0 if report["gpu_correctness_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
