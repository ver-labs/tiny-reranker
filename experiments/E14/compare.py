"""Verify provenance and compare same-GPU CUDA port/Triton plus saved M5 Metal."""
import json
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    report = json.loads((HERE / 'benchmark.json').read_text())
    device = json.loads((HERE / 'device_benchmark.json').read_text())
    metal = json.loads((ROOT / 'experiments/E11/kernel_benchmark.json').read_text())
    for key in ('checkpoint_sha256', 'candidate_sha256', 'sample_sha256'):
        if report[key] != metal[key]: raise ValueError(f'Mismatched Metal fixture: {key}')
    if device['checkpoint_sha256'] != report['checkpoint_sha256']: raise ValueError('Device checkpoint mismatch')
    for path, expected in report['code_sha256'].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected: raise ValueError(f'Changed measured code: {path}')
    rows = []
    for item in report['model']:
        size = item['pair_batch_size']
        op = next(x for x in report['operation'] if x['batch_size'] == size and x['num_warps'] == 4)
        graph = next(x for x in device['rows'] if x['batch_size'] == size)
        m = next(x for x in metal['model_benchmark'] if x['batch_size'] == size)
        mop = next(x for x in metal['operation_microbench'] if x['batch_size'] == size)
        if op['shape'] != graph['shape'] or op['shape'] != [size, mop['sequence_length'], 128]:
            raise ValueError('Mismatched operation shape')
        rows.append({'batch_size': size, 'shape': op['shape'],
            'l40s_operation_wall_ms': {mode: op[f'{mode}_ms']['median'] for mode in ('reference', 'triton', 'cuda')},
            'l40s_operation_graph_ms': {mode: value['median'] for mode, value in graph['operation_ms'].items()},
            'l40s_forward_ms': {mode: value['median'] for mode, value in item['median_batch_ms']['forward'].items()},
            'l40s_complete_scoring_ms': {mode: value['median'] for mode, value in item['median_batch_ms']['scoring'].items()},
            'm5_metal_operation_wall_ms': mop['fused']['median'],
            'm5_metal_complete_scoring_ms': m['scoring']['fused']['median_batch_ms']['median']})
    combined = {'environment': report['environment'], 'metal_device': metal['device'],
        'checkpoint_sha256': report['checkpoint_sha256'], 'sample_parity': report['sample_parity'],
        'rows': rows, 'limitations': [
            'CUDA is a handwritten CUDA C++ port of the Metal algorithm, not Metal running on NVIDIA.',
            'CUDA/Triton are paired on L40S; original Metal was measured on M5, so hardware differs.',
            'Graph timing amortizes 100 operations and excludes host dispatch; it is not an isolated kernel profiler trace.',
            'Epoch 3; fixed 128-query validation sample, not a full test-quality evaluation.',
            '20 rotated wall repeats; 30 rotated graph/operation repeats; warm inference only.']}
    (HERE / 'comparison.json').write_text(json.dumps(combined, indent=2) + '\n')
    x = [r['batch_size'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout='constrained')
    for mode, label, color in [('cuda', 'CUDA port of Metal algorithm', '#d66b24'), ('triton_4', 'Triton (4 warps)', '#207bb0'), ('triton_1', 'Triton (1 warp)', '#427451'), ('reference', 'PyTorch reference', '#747474')]:
        axes[0].plot(x, [r['l40s_operation_graph_ms'][mode] * 1000 for r in rows], 'o-', label=label, color=color)
    for mode, label, color in [('cuda', 'CUDA port of Metal algorithm', '#d66b24'), ('triton', 'Triton (4 warps)', '#207bb0'), ('reference', 'PyTorch reference', '#747474')]:
        axes[1].plot(x, [r['l40s_forward_ms'][mode] for r in rows], 'o-', label=label, color=color)
    axes[0].set_title('Residual + LayerNorm: device graph replay'); axes[0].set_ylabel('Microseconds per operation')
    axes[1].set_title('Epoch 3: prepared-input forward'); axes[1].set_ylabel('Milliseconds per batch')
    for ax in axes:
        ax.set_xscale('log', base=2); ax.set_xticks(x, [str(v) for v in x]); ax.set_xlabel('Pair batch size')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Same NVIDIA L40S · FP32 · warm inference', fontsize=14)
    fig.savefig(HERE / 'comparison.png', dpi=170)
    for row in rows: print(json.dumps(row))


if __name__ == '__main__': main()
