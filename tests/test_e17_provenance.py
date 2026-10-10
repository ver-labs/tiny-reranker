"""Reject changed custom binaries and prohibit CPU substitution for NVIDIA."""
import json
import sys
import pytest
from experiments.E17 import nvidia_provenance as guard


@pytest.fixture
def assets(tmp_path,monkeypatch):
    (tmp_path/'plan.json').write_text('{}')
    (tmp_path/'kernel.cu').write_text('source')
    (tmp_path/'kernel.so').write_bytes(b'binary')
    plan={'benchmark_plan_sha256':guard.sha(tmp_path/'plan.json'),
          'guard_source_sha256':guard.sha(guard.__file__),
          'files':{p.name:guard.sha(p) for p in [tmp_path/'kernel.cu',tmp_path/'kernel.so']}}
    (tmp_path/'nvidia_provenance_plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(guard,'ROOT',tmp_path);monkeypatch.setattr(guard,'HERE',tmp_path)
    monkeypatch.setattr(guard,'verify_plan',lambda: {})
    monkeypatch.setattr(sys,'argv',['guard'])
    return tmp_path


def test_asset_check_never_claims_gpu_validation(assets):
    guard.main()
    r=json.loads((assets/'nvidia_provenance.json').read_text())
    assert r['status']=='binary_and_source_hashes_verified'
    assert 'pending' in r['gpu_execution'] and 'gpu' not in r


def test_changed_custom_binary_rejected(assets):
    (assets/'kernel.so').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='NVIDIA asset changed'):guard.main()
    assert not (assets/'nvidia_provenance.json').exists()


def test_nvidia_guard_refuses_nonlinux_cpu_substitution(assets,monkeypatch):
    monkeypatch.setattr(sys,'argv',['guard','--require-gpu'])
    monkeypatch.setattr(guard.platform,'system',lambda: 'Darwin')
    with pytest.raises(RuntimeError,match='no CPU substitution'):guard.main()
    assert not (assets/'nvidia_provenance.json').exists()
