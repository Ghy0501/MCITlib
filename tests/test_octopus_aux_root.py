# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 MCITlib authors
"""CPU regressions for Octopus's optional, checkpoint-independent auxiliary root."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from mcitlib.config.schema import Command
from mcitlib.dataflow import find_gaps
from mcitlib.doctor import Report, check_inputs
from mcitlib.methods._octopus.plan import task_paths, train_stages
from mcitlib.methods._octopus.state import write_manifest
from mcitlib.paths import PathResolver
from mcitlib.validation_batch import worker_commands


class AuxiliaryRootTests(unittest.TestCase):
    def make_train(self):
        return Command(argv=['python', '--output_dir', '/weights/Stage2/Task3_llava_lora',
                             '--data_path', '/data/source.json', '--image_folder', '/images',
                             '--model_name_or_path', '/model', '--vision_tower', '/vision'],
                       env={}, kind='train', task_id=3, metadata={'order': 7})

    def config(self):
        return {'octopus_stage2_samples': 9600, 'octopus_stage2_epochs': 1,
                'octopus_stage2_lr': 0.0001, 'octopus_lambda_1': 0.02, 'octopus_lambda_2': 0.01}

    def test_absent_root_preserves_original_paths(self):
        self.assertEqual(task_paths('/weights/Stage2/Task3_llava_lora', 3), (
            '/weights/Stage1/Task3_llava_lora',
            ['/weights/Stage2/Task1_llava_lora', '/weights/Stage2/Task2_llava_lora'],
            '/weights/Stage2/task3_stage2_data.json', '/weights/Stage2/task3_gpwc.pt'))
        self.assertEqual(task_paths('/weights/Stage2/Task3_llava_lora', 3, None),
                         task_paths('/weights/Stage2/Task3_llava_lora', 3))

    def test_native_pipeline_uses_split_paths_and_same_budgets(self):
        baseline = train_stages(self.make_train(), self.config(), 3)
        split = train_stages(self.make_train(), dict(self.config(), octopus_aux_root='/project/aux'), 3)
        self.assertEqual(len(baseline), 5)
        self.assertEqual([c.metadata['octopus_stage'] for c in split],
                         ['stage1', 'sample', 'gpwc', 'stage2', 'cleanup'])
        bstage = {c.metadata['octopus_stage']: c for c in baseline}
        stage = {c.metadata['octopus_stage']: c for c in split}
        self.assertEqual(stage['stage1'].as_dict(), bstage['stage1'].as_dict())
        sample = '/project/aux/task3_stage2_data.json'
        gpwc = '/project/aux/task3_gpwc.pt'
        self.assertEqual(stage['sample'].argv[-1], sample)
        self.assertEqual(stage['sample'].argv[stage['sample'].argv.index('--count') + 1], '9600')
        self.assertEqual(stage['gpwc'].argv[stage['gpwc'].argv.index('--output') + 1], gpwc)
        for flag, value in (('--samples', '256'), ('--logical-batch', '16'), ('--microbatch', '1')):
            self.assertEqual(stage['gpwc'].argv[stage['gpwc'].argv.index(flag) + 1], value)
        self.assertEqual(stage['stage2'].argv[stage['stage2'].argv.index('--data_path') + 1], sample)
        self.assertEqual(stage['stage2'].argv[stage['stage2'].argv.index('--output_dir') + 1],
                         '/weights/Stage2/Task3_llava_lora')
        self.assertEqual(stage['stage2'].env['MCITLIB_OCTOPUS_GPWC'], gpwc)
        self.assertEqual(json.loads(stage['stage2'].env['MCITLIB_OCTOPUS_HISTORY']),
                         ['/weights/Stage2/Task1_llava_lora', '/weights/Stage2/Task2_llava_lora'])
        self.assertEqual(stage['cleanup'].argv[-1], gpwc + '.receipt.json')
        self.assertEqual(stage['cleanup'].argv[stage['cleanup'].argv.index('--input') + 1], gpwc)
        self.assertEqual(stage['stage2'].argv[stage['stage2'].argv.index('--learning_rate') + 1], '0.0001')

    def test_invalid_roots_rejected(self):
        for root in ('', ' ', '/project/../weights', 0):
            with self.subTest(root=root), self.assertRaises(ValueError):
                task_paths('/weights/Stage2/Task3_llava_lora', 3, root)

    def test_split_output_requires_producer_and_does_not_exempt_siblings(self):
        stages = train_stages(self.make_train(), dict(self.config(), octopus_aux_root='/project/aux'), 3)
        history = ('/weights/Stage2/Task1_llava_lora', '/weights/Stage2/Task2_llava_lora')
        self.assertEqual(find_gaps(stages, ('/model', '/vision', '/data'), history), [])
        missing_sample = [stage for stage in stages if stage.metadata.get('octopus_stage') != 'sample']
        gaps = find_gaps(missing_sample, ('/model', '/vision', '/data'), history)
        self.assertTrue(any(gap.path == '/project/aux/task3_stage2_data.json' for gap in gaps))
        report = Report()
        commands = [Command(argv=['run', '--data_path', '/project/aux/unrelated.json'])]
        check_inputs(report, commands, Path('/model'), Path('/data'),
                     (Path('/project/aux/task3_stage2_data.json'),))
        self.assertTrue(report.failed)

    def test_profile_explicit_opt_in_is_confined_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            profile = root / 'paths.yaml'
            profile.write_text('logs_root: logs\n')
            plain = PathResolver.from_file(profile, repo_root=root)
            self.assertIsNone(plain.octopus_aux_root)
            self.assertNotIn('octopus_aux_root', json.loads(plain.context('ucit')))
            profile.write_text('logs_root: logs\noctopus_aux_root: ${logs_root}/aux/ucit\n')
            resolved = PathResolver.from_file(profile, repo_root=root)
            self.assertEqual(resolved.octopus_aux_root, root / 'logs/aux/ucit')
            restored, benchmark = PathResolver.from_context(resolved.context('ucit'))
            self.assertEqual((restored.octopus_aux_root, benchmark), (resolved.octopus_aux_root, 'ucit'))
            self.assertEqual(plain.context('ucit'), resolved.context('ucit', include_octopus_aux=False))
            for invalid in ('', '../escape', '/tmp/elsewhere'):
                profile.write_text(f'logs_root: logs\noctopus_aux_root: "{invalid}"\n')
                with self.subTest(path=invalid), self.assertRaises(ValueError):
                    PathResolver.from_file(profile, repo_root=root)

    def test_octopus_option_does_not_change_keeplora_workers(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            profile = Path(td) / 'paths.yaml'
            profile.write_text('logs_root: logs\n')
            default = PathResolver.from_file(profile, repo_root=project)
            profile.write_text('logs_root: logs\noctopus_aux_root: ${logs_root}/octopus-aux\n')
            split = PathResolver.from_file(profile, repo_root=project)
            for benchmark in ('ucit', 'mllm_dcl'):
                with self.subTest(benchmark=benchmark):
                    plain = worker_commands(project, default, 'keeplora', benchmark, '0,1', 29800)
                    with_option = worker_commands(project, split, 'keeplora', benchmark, '0,1', 29800)
                    self.assertEqual([c.as_dict() for c in plain], [c.as_dict() for c in with_option])

    def test_first_task_still_copies_checkpoint(self):
        train = self.make_train()
        train.argv[train.argv.index('--output_dir') + 1] = '/weights/Stage2/Task1_llava_lora'
        default = train_stages(copy.deepcopy(train), self.config(), 1)
        split = train_stages(copy.deepcopy(train), dict(self.config(), octopus_aux_root='/project/aux'), 1)
        self.assertEqual([c.as_dict() for c in default], [c.as_dict() for c in split])

    def test_manifest_preserves_cross_root_relative_audit(self):
        # No symlinks: the checkpoint and auxiliary data are siblings.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint, auxiliary = root / 'weights/Task1_llava_lora', root / 'project/aux'
            checkpoint.mkdir(parents=True)
            auxiliary.mkdir(parents=True)
            (checkpoint / 'adapter_model.bin').write_bytes(b'adapter')
            audit = auxiliary / 'task3_gpwc.pt.audit.json'
            audit.write_text('{}')
            write_manifest(checkpoint, [], {'audit': str(audit), 'sha256': 'gpwc', 'audit_sha256': 'audit'})
            payload = json.loads((checkpoint / 'octopus_manifest.json').read_text())
            self.assertEqual((checkpoint / payload['gpwc']['audit']).resolve(), audit.resolve())


if __name__ == '__main__':
    unittest.main()
