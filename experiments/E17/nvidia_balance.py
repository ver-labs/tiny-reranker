"""Missing fifth startup rotation; frozen measurement function unchanged."""
import argparse
import json
from unittest.mock import patch
from .common import HERE, sha, verify_plan
from . import direct_benchmark


def main():
    verify_plan()
    supplement = json.loads((HERE/'nvidia_balance_plan.json').read_text())
    assert supplement['benchmark_plan_sha256'] == sha(HERE/'plan.json')
    assert json.loads((HERE/'cuda_compile.json').read_text())['status'] == 'compatible'
    # Only adapt argument validation: the existing parser permits sessions 0..3.
    # Timing, warming, inference and saved-observation logic remain byte-identical.
    with patch.object(argparse.ArgumentParser, 'parse_args', return_value=argparse.Namespace(device='cuda', session=4)):
        direct_benchmark.main()


if __name__ == '__main__':
    main()
