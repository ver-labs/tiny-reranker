"""Verify full-run sampler and checkpoint continuation without a long GPU job."""
import random

import numpy as np
import pytest
import torch

from scripts.full_train import load_state, new_state, order_for_epoch, save_state, validate_sampler


def components():
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler('cpu')
    return model, optimizer, scaler


def update(model, optimizer, scaler):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model(torch.ones(2, 3)).float().square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def test_full_sampler_and_last_partial_batch():
    order = order_for_epoch(735276, 0)
    assert np.array_equal(order, order_for_epoch(735276, 0))
    assert not np.array_equal(order, order_for_epoch(735276, 1))
    assert np.array_equal(np.sort(order), np.arange(735276))
    assert len(order[-12:]) == 12
    state = new_state()
    state.update(order=torch.from_numpy(order), next_sample_offset=16000)
    validate_sampler(state, len(order))
    assert len(state['order'][state['next_sample_offset']:]) == 719276
    state['next_sample_offset'] = len(order)
    validate_sampler(state, len(order))


def test_resume_restores_optimizer_scaler_rng_and_batch_position(tmp_path):
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    model, optimizer, scaler = components()
    update(model,optimizer,scaler)
    state = new_state()
    state.update(order=torch.from_numpy(order_for_epoch(32,0)),next_sample_offset=16,
                 optimizer_updates=1,attempted_batches=1)
    latest = tmp_path/'latest.pt'
    save_state(latest,model,optimizer,scaler,state,'provenance',torch.device('cpu'))
    expected_python, expected_numpy = random.random(), np.random.random()
    update(model,optimizer,scaler)
    expected_weights = {k:v.clone() for k,v in model.state_dict().items()}
    restored, restored_optimizer, restored_scaler = components()
    loaded = load_state(latest,restored,restored_optimizer,restored_scaler,'provenance',32,torch.device('cpu'))
    assert loaded['next_sample_offset']==16
    assert random.random()==expected_python and np.random.random()==expected_numpy
    update(restored,restored_optimizer,restored_scaler)
    assert all(torch.equal(v,expected_weights[k]) for k,v in restored.state_dict().items())
    assert restored_scaler.state_dict()==scaler.state_dict()
    assert {int(s['step'].item()) for s in restored_optimizer.state.values()}=={2}
    # Publishing a new generation leaves the earlier slot intact.
    loaded.update(next_sample_offset=32,optimizer_updates=2,attempted_batches=2)
    save_state(latest,restored,restored_optimizer,restored_scaler,loaded,'provenance',torch.device('cpu'))
    assert (tmp_path/'latest_0.pt').exists() and (tmp_path/'latest_1.pt').exists()
    with pytest.raises(RuntimeError,match='hash/provenance'):
        load_state(latest,restored,restored_optimizer,restored_scaler,'other',32,torch.device('cpu'))


def test_bad_resume_offsets_or_order_rejected():
    state = new_state()
    state.update(next_sample_offset=3,order=torch.arange(32))
    with pytest.raises(RuntimeError,match='boundary'):
        validate_sampler(state,32)
    state.update(next_sample_offset=16,order=torch.zeros(32,dtype=torch.int64))
    with pytest.raises(RuntimeError,match='permutation'):
        validate_sampler(state,32)


def test_profiling_windows_are_bounded_and_separate():
    from scripts.full_train import profiling_windows
    windows = profiling_windows(45955)
    measured = set()
    for start in windows.values():
        batches = set(range(start,start+60))
        assert batches.isdisjoint(measured)
        assert min(batches)>1 and max(batches)<45955
        measured.update(batches)
    assert len(measured)==180  # 150 synchronized batches and 30 CPU-traced batches.
