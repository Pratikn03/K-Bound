"""Native SAR config distinction: optimizer exclusion is not BN-stat freezing."""
import ast
import copy
import json
import hashlib
import os
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def functions():
    path = ROOT / 'docs/research/kbound/scripts/cifar_tent_mps_v2.py'
    tree = ast.parse(path.read_text())
    ns = {'torch': torch, 'nn': nn, 'copy': copy, 'os': os, 'json': json,
          'SAR_FREEZE_LAYER4': True}
    names = {'_bn_affine_params', '_clone_for_tta', 'run_imagenet_benchmark'}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                 and n.name in names], type_ignores=[]), str(path), 'exec'), ns)
    return ns


def test_excluded_final_block_uses_batch_stats_without_optimizer_parameters():
    base = nn.Module()
    base.bn = nn.BatchNorm2d(3)
    base.layer4 = nn.Sequential(nn.BatchNorm2d(3))
    base.eval()
    model, params, _ = functions()['_clone_for_tta'](base, freeze_layer4=True)
    bn = model.layer4[0]
    assert bn.track_running_stats is False
    assert bn.running_mean is None and bn.running_var is None
    assert not any(p is bn.weight or p is bn.bias for p in params)
    assert not bn.weight.requires_grad and not bn.bias.requires_grad
    x = torch.randn(4, 3, 2, 2)
    model.train(); before = bn(x)
    model.eval(); after = bn(x)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert base.layer4[0].track_running_stats is True


def test_default_still_optimizes_all_batch_norm_affines():
    base = nn.Module()
    base.layer4 = nn.Sequential(nn.BatchNorm2d(3))
    model, params, _ = functions()['_clone_for_tta'](base)
    assert any(p is model.layer4[0].weight for p in params)
    assert any(p is model.layer4[0].bias for p in params)


def test_full_resnet50_bn_optimizer_and_statistics_match_pinned_native_sar():
    """Configuration parity only: no pretrained weights or benchmark claims."""
    from torchvision.models import resnet50

    path = ROOT / "external/sar_official/sar.py"
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "0553a395ac2bc087049720f1f162789079fe5ee25aa6a1cf82b2ea6286d66eff"
    )
    tree = ast.parse(payload)
    native = {"nn": nn}
    exec(compile(ast.Module(body=[node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"configure_model", "collect_params"}],
        type_ignores=[]), str(path), "exec"), native)
    base = resnet50(weights=None).eval()
    reference = native["configure_model"](copy.deepcopy(base))
    reference_params, reference_names = native["collect_params"](reference)
    local, local_params, _ = functions()["_clone_for_tta"](base, freeze_layer4=True)
    ids = {id(param) for param in local_params}
    local_names = [name for name, param in local.named_parameters() if id(param) in ids]
    assert local_names == reference_names
    assert len(local_params) == len(reference_params) > 0
    for name, module in reference.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            actual = local.get_submodule(name)
            assert actual.training == module.training
            assert actual.track_running_stats == module.track_running_stats is False
            assert actual.running_mean is module.running_mean is None
            assert actual.running_var is module.running_var is None
    assert all(module.track_running_stats for module in base.modules()
               if isinstance(module, nn.BatchNorm2d))


def test_historical_resume_rejected_before_model_or_data_access(tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'rows': {'sar': []}, 'done': ['old-cell']}))
    original = checkpoint.read_bytes()
    ns = functions()
    ns['_ic_available'] = lambda *args: pytest.fail('data accessed before resume guard')
    with pytest.raises(SystemExit, match='fresh output'):
        ns['run_imagenet_benchmark']('unused', None, 'cpu', ['sar'], ['noise'],
                                    out_dir=str(tmp_path))
    assert checkpoint.read_bytes() == original


def test_results_without_resume_identity_are_preserved_and_rejected(tmp_path):
    result = tmp_path / 'decisive_tta_results.json'
    result.write_text('{}')
    ns = functions()
    ns['_ic_available'] = lambda *args: pytest.fail('data accessed before resume guard')
    with pytest.raises(SystemExit, match='fresh output'):
        ns['run_imagenet_benchmark']('unused', None, 'cpu', ['sar'], ['noise'],
                                    out_dir=str(tmp_path))
    assert result.read_text() == '{}'
