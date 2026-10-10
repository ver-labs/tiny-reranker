"""Assemble measured Metal/Triton latency results; absent GPU data stays null."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def compare(metal, triton=None):
    if triton is not None:
        if triton.get('status') != 'complete': raise ValueError('Triton report must be a completed GPU benchmark')
        for key in ('checkpoint_sha256', 'candidate_sha256', 'sample_sha256'):
            if metal[key] != triton[key]: raise ValueError(f'Mismatched comparison provenance: {key}')
    rows = []
    for model in metal['model_benchmark']:
        size = model['batch_size']
        op = next(x for x in metal['operation_microbench'] if x['batch_size'] == size)
        row = {'batch_size': size, 'padded_operation_tokens': op['sequence_length'],
               'metal': {'operation_reference_ms': op['standard']['median'], 'operation_custom_ms': op['fused']['median'],
                         'operation_speedup': op['speedup']}, 'triton': None}
        for scope in ('forward', 'scoring'):
            row['metal'][scope] = {mode: {'median_batch_ms': model[scope][mode]['median_batch_ms']['median'],
                                        'pairs_per_second': model[scope][mode]['pairs_per_second']['median']}
                                    for mode in ('standard', 'fused')}
        if triton is not None:
            cuda_model = next(x for x in triton['model'] if x['pair_batch_size'] == size)
            cuda_op = next(x for x in triton['operation'] if x['batch_size'] == size and x['num_warps'] == triton['num_warps'])
            if cuda_op['shape'] != [size, op['sequence_length'], 128]: raise ValueError('Mismatched operation fixture shape')
            row['triton'] = {'operation_reference_ms': cuda_op['reference_ms']['median'],
                             'operation_custom_ms': cuda_op['triton_ms']['median'],
                             'operation_speedup': cuda_op['reference_ms']['median'] / cuda_op['triton_ms']['median'],
                             'num_warps': triton['num_warps']}
            for scope in ('forward', 'scoring'):
                row['triton'][scope] = {mode: {'median_batch_ms': cuda_model['median_batch_ms'][scope][mode]['median'],
                                              'pairs_per_second': cuda_model['pairs_per_second'][scope][mode]['median']}
                                        for mode in ('original', 'reference', 'triton', 'optimized')
                                        if mode in cuda_model['median_batch_ms'][scope]}
        rows.append(row)
    return {'status': 'complete' if triton is not None else 'pending_nvidia_measurements',
            'checkpoint_sha256': metal['checkpoint_sha256'], 'dataset_split': 'validation',
            'metal_device': metal['device'], 'triton_environment': triton['environment'] if triton else None,
            'rows': rows, 'limitations': [
                'Metal and Triton execute on different GPU hardware; absolute latency cannot isolate kernel-language effects.',
                'Compare custom/reference speedups on each device; optimized CUDA model is a separate practical baseline with reported score differences.',
                'Wall timings include dispatch and synchronization; model outputs are CPU readable.',
                'Metal model repeats use two reversed batch-order rounds; CUDA rotates its paths in one batch-order sweep.',
                'Missing NVIDIA measurements are null, never estimated from Metal data.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triton-report', type=Path, default=HERE / 'benchmark.json')
    args = parser.parse_args()
    metal = json.loads((ROOT / 'experiments/E11/kernel_benchmark.json').read_text())
    triton = json.loads(args.triton_report.read_text()) if args.triton_report.exists() else None
    result = compare(metal, triton)
    from scripts.checkpoint_io import atomic_write_text
    path = HERE / 'metal_triton_comparison.json'
    atomic_write_text(json.dumps(result, indent=2) + '\n', path, replace_existing=True)
    print(result['status'])
    print('Batch | Metal operation ms | Metal complete scoring ms | Triton operation ms | Triton complete scoring ms')
    for row in result['rows']:
        t = row['triton']
        print(f'{row["batch_size"]:5} | {row["metal"]["operation_custom_ms"]:.3f} | '
              f'{row["metal"]["scoring"]["fused"]["median_batch_ms"]:.3f} | '
              + (f'{t["operation_custom_ms"]:.3f} | {t["scoring"]["triton"]["median_batch_ms"]:.3f}' if t else 'pending | pending'))
    print(path)


if __name__ == '__main__': main()
