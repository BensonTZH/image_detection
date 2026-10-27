"""Portable maintenance commands must preview changes and protect existing work."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
RELOCATE = ROOT / 'ml/scripts/relocate_training_paths.py'
TRAIN = ROOT / 'ml/scripts/train_prepared_classifier.py'
OLD = '/Users/bensontan/Desktop/image_detection'


class HandoffToolsTests(unittest.TestCase):
    def test_relocation_previews_then_backs_up_and_is_idempotent(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'web').mkdir()
            (root / 'web/package.json').write_text('{}')
            (root / 'dataset').mkdir()
            config = root / 'dataset/manifest.json'
            original = json.dumps({'source': OLD + '/dataset/photo.jpg'})
            config.write_text(original)
            image = root / 'dataset/photo.jpg'
            image.write_bytes(OLD.encode())
            command = [sys.executable, str(RELOCATE), '--root', str(root)]
            preview = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(preview.stdout)['files'], ['dataset/manifest.json'])
            self.assertEqual(config.read_text(), original)
            subprocess.run(command + ['--apply'], check=True, capture_output=True)
            self.assertEqual(json.loads(config.read_text())['source'], root.as_posix() + '/dataset/photo.jpg')
            self.assertEqual(config.with_name('manifest.json.original-paths').read_text(), original)
            self.assertEqual(image.read_bytes(), OLD.encode())
            again = subprocess.run(command + ['--apply'], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(again.stdout)['files'], [])

    def test_relocation_refuses_to_overwrite_a_backup(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'web').mkdir()
            (root / 'web/package.json').write_text('{}')
            (root / 'dataset').mkdir()
            config = root / 'dataset/data.yaml'
            original = 'path: ' + OLD + '/dataset\n'
            config.write_text(original)
            backup = config.with_name('data.yaml.original-paths')
            backup.write_text('previous backup')
            result = subprocess.run([sys.executable, str(RELOCATE), '--root', str(root), '--apply'], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(config.read_text(), original)
            self.assertEqual(backup.read_text(), 'previous backup')

    def test_classifier_plan_uses_current_weights_only_when_requested(self):
        with TemporaryDirectory() as folder:
            # Copy just the command into an empty repository-shaped directory.
            root = Path(folder).resolve()
            script = root / 'ml/scripts/train_prepared_classifier.py'
            script.parent.mkdir(parents=True)
            script.write_text(TRAIN.read_text())
            command = [sys.executable, str(script), 'high-angle', '--name', 'candidate', '--dry-run']
            original = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            current = json.loads(subprocess.run(command + ['--from-current'], check=True, capture_output=True, text=True).stdout)
            self.assertTrue(original['weights'].endswith('v6-roi/weights/best.pt'))
            self.assertTrue(current['weights'].endswith('v7-six/weights/best.pt'))
            self.assertFalse((root / 'runs').exists())
            (root / 'runs/classifier-candidates/candidate').mkdir(parents=True)
            refused = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('Refusing to overwrite', refused.stderr)

    def test_classifier_refuses_run_path_traversal(self):
        result = subprocess.run([sys.executable, str(TRAIN), 'acceptability', '--name', '../outside', '--dry-run'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('simple new run name', result.stderr)


if __name__ == '__main__':
    unittest.main()
