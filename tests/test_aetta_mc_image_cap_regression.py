"""Exercise the runner's actual batching AST without importing torch/models.

The observer replaces neural work after batch selection; N, iteration and slice
are unchanged executable nodes. A slice exceeding the requested budget fails.
"""
import ast
import copy
from pathlib import Path
import unittest


class AettaImageCapRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts/cifar_tent_mps_v2.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == '_aetta_accuracy_estimate')
        assign_n = next(n for n in function.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'N' for t in n.targets))
        loop = next(n for n in ast.walk(function) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == 'i')
        assign_batch = copy.deepcopy(loop.body[0])
        if not isinstance(assign_batch, ast.Assign) or assign_batch.targets[0].id != 'xb':
            raise AssertionError('Estimator batching changed; update the extraction boundary')
        observed_loop = ast.For(target=copy.deepcopy(loop.target), iter=copy.deepcopy(loop.iter),
                                body=[assign_batch, ast.parse('visited.extend(xb)').body[0]], orelse=[])
        fragment = ast.Module(body=[copy.deepcopy(assign_n), observed_loop], type_ignores=[])
        cls.batch_code = compile(ast.fix_missing_locations(fragment), str(path), 'exec')

    def test_positive_cap_never_processes_extra_samples(self):
        for cap, length in [(1,1000), (257,1000), (2049,3000), (2048,3000), (257,257), (2049,300)]:
            with self.subTest(cap=cap, length=length):
                namespace = {'x': list(range(length)), 'mc_images': cap, 'visited': []}
                exec(self.batch_code, namespace)
                self.assertEqual(namespace['visited'], list(range(min(cap, length))))

    def test_zero_retains_uncapped_behavior(self):
        namespace = {'x': list(range(1000)), 'mc_images': 0, 'visited': []}
        exec(self.batch_code, namespace)
        self.assertEqual(namespace['visited'], list(range(1000)))

    def test_empty_input_has_no_batches(self):
        namespace = {'x': [], 'mc_images': 257, 'visited': []}
        exec(self.batch_code, namespace)
        self.assertEqual(namespace['visited'], [])


if __name__ == '__main__':
    unittest.main()
