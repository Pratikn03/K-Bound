"""Synthetic gate and real NPZ loader validation; no model or dataset imports."""
import ast
import glob
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'experiments/kbound/poem_aetta'
spec = importlib.util.spec_from_file_location('aetta_pair_test_module', DIRECTORY / 'aetta_dropout.py')
AETTA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AETTA)


class EstimatePairValidation(unittest.TestCase):
    @staticmethod
    def alignment_route():
        """Execute actual input loaders/alignment, excluding scorer bootstrap/main."""
        reader_path = ROOT / 'experiments/kbound/wilds/multiseed_paired_ci.py'
        reader = next(n for n in ast.parse(reader_path.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'load_cell')
        reader_scope = {'os':os,'json':json}
        exec(compile(ast.Module(body=[reader],type_ignores=[]),str(reader_path),'exec'),reader_scope)
        path = DIRECTORY / 'score_official_headtohead.py'
        names = {'SchemaError','_npz_scalar_str','_index_sample_npzs','_load_source_entropy','_align_seed'}
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes} == names
        scope = {'np':np,'os':os,'glob':glob,'AETTA':AETTA,'A0_TOL':.005,
                 'MPC':SimpleNamespace(load_cell=reader_scope['load_cell'])}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        return scope

    @staticmethod
    def write_synthetic_inputs(directory, estimates):
        fields = dict(condition='synthetic',frozen_entropy=np.array([.2]),
                      adapted_entropy=np.array([.1]),a0=.6,a_adapted=.7,**estimates)
        np.savez(Path(directory)/'samples_fixture_tent_seed0__synthetic.npz',**fields)
        record = {'condition':'synthetic','a0':.6,'a_adapted':.7,'B':.1}
        (Path(directory)/'per_condition_fixture_tent_seed0.json').write_text(
            json.dumps({'records':[record]}))

    def test_alignment_propagates_invalid_pair_without_returning_streams(self):
        route = self.alignment_route()
        for key in ('aetta_acc_est','aetta_acc_est_frozen'):
            for value in (None,float('nan'),float('inf'),-float('inf'),-.01,1.01,True,[.8]):
                with self.subTest(key=key,value=value), tempfile.TemporaryDirectory() as directory:
                    estimates = {'aetta_acc_est':.8,'aetta_acc_est_frozen':.7}
                    if value is None:
                        del estimates[key]
                    else:
                        estimates[key] = value
                    self.write_synthetic_inputs(directory,estimates)
                    before = sorted(p.name for p in Path(directory).iterdir())
                    with self.assertRaises(route['SchemaError']):
                        route['_align_seed'](directory,'fixture','tent',0)
                    self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),before)

    def test_valid_alignment_to_gate_preserves_expected_decisions(self):
        route = self.alignment_route()
        for adapted,frozen,want in [(.8,.9,'FREEZE'),(.8,.7,'ADAPT'),(.19,.1,'FREEZE'),
                                    (.2,.2,'ADAPT'),(1.,1.,'ADAPT'),(0.,0.,'FREEZE')]:
            with self.subTest(adapted=adapted,frozen=frozen), tempfile.TemporaryDirectory() as directory:
                self.write_synthetic_inputs(directory,dict(aetta_acc_est=adapted,aetta_acc_est_frozen=frozen))
                records,streams,source = route['_align_seed'](directory,'fixture','tent',0)
                self.assertEqual(records[0]['condition'],streams[0]['condition'])
                self.assertIsNone(source)
                self.assertEqual(AETTA.aetta_dropout_decision(streams).tolist(),[want])

    def test_gate_rejects_missing_or_invalid_estimate(self):
        for key in ('aetta_acc_est', 'aetta_acc_est_frozen'):
            for value in (None, float('nan'), float('inf'), -float('inf'), -.01, 1.01,
                          True, '0.8', [0.8]):
                with self.subTest(key=key, value=value):
                    row = {'aetta_acc_est': .8, 'aetta_acc_est_frozen': .7}
                    if value is None:
                        del row[key]
                    else:
                        row[key] = value
                    with self.assertRaises(ValueError):
                        AETTA.aetta_dropout_decision([row])

    def test_valid_pair_keeps_gate_decisions(self):
        pairs = [(.8,.9), (.8,.7), (.19,.1), (.2,.2), (1.,1.), (0.,0.)]
        rows = [dict(aetta_acc_est=a, aetta_acc_est_frozen=f) for a,f in pairs]
        self.assertEqual(AETTA.aetta_dropout_decision(rows).tolist(),
                         ['FREEZE','ADAPT','FREEZE','ADAPT','ADAPT','FREEZE'])

    def test_real_npz_loader_rejects_missing_or_invalid_pair(self):
        path = DIRECTORY / 'score_official_headtohead.py'
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in ('SchemaError','_npz_scalar_str','_index_sample_npzs')]
        scope = {'np':np,'os':os,'glob':glob,'AETTA':AETTA}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
        for value in (None,float('nan'),float('inf'),-.01,1.01,.7):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                fields = dict(condition='synthetic',frozen_entropy=np.array([.2]),
                              adapted_entropy=np.array([.1]),aetta_acc_est=.8)
                if value is not None:
                    fields['aetta_acc_est_frozen'] = value
                np.savez(Path(directory)/'samples_fixture_tent_seed0__synthetic.npz',**fields)
                if value == .7:
                    result = scope['_index_sample_npzs'](directory,'fixture','tent',0)
                    self.assertEqual(result['synthetic']['aetta_acc_est_frozen'],.7)
                else:
                    with self.assertRaises(scope['SchemaError']):
                        scope['_index_sample_npzs'](directory,'fixture','tent',0)


if __name__ == '__main__':
    unittest.main()
