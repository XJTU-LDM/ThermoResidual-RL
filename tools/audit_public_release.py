"""Audit the allowlisted public tree without printing sensitive values."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def scan(root=ROOT, *, checkpoints=True):
    issues = []
    patterns = {
        'private_machine_path': re.compile(r'/(?:gpu_move|home|Users)/[^\s"\']+'),
        'credential_assignment': re.compile(r'(?im)^\s*(?:password|passwd|api_key|access_token|secret_key)\s*[:=]\s*["\']?[^\s"\']{6,}'),
        'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
        'github_token': re.compile(r'\bgh[pousr]_[A-Za-z0-9]{25,}\b'),
    }
    files = []
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if any(p in ('outputs', '.venv', '__pycache__', '.pytest_cache', '.git') for p in rel.parts):
            continue
        if not path.is_file():
            continue
        files.append(path)
        if any(part in ('experiments', 'results') for part in rel.parts):
            issues.append(dict(file=str(rel), reason='development_archive_directory'))
        if path.is_symlink():
            issues.append(dict(file=str(rel), reason='symlink_not_allowed'))
        if path.suffix.lower() in ('.xlsx', '.xls', '.pdf', '.pem', '.key') or path.name == 'data2.csv' or path.name.startswith('.env'):
            issues.append(dict(file=str(rel), reason='excluded_file_type_or_name'))
        if path.suffix.lower() == '.csv' and str(rel) not in (
                'data/synthetic/reference_trace.csv', 'data/synthetic/synthetic_boundary.csv'):
            issues.append(dict(file=str(rel), reason='unlisted_tabular_data'))
        if path.suffix == '.pth':
            continue
        if path.suffix.lower() == '.png':
            from PIL import Image
            if str(rel) != 'docs/figures/control_framework.png':
                issues.append(dict(file=str(rel), reason='unlisted_image'))
            with Image.open(path) as picture:
                picture.verify()
                if any(key not in ('dpi',) for key in picture.info):
                    issues.append(dict(file=str(rel), reason='unexpected_image_metadata'))
            continue
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            issues.append(dict(file=str(rel), reason='unexpected_binary'))
            continue
        for name, pattern in patterns.items():
            if pattern.search(content):
                issues.append(dict(file=str(rel), reason=name))
    checked_checkpoints = 0
    if checkpoints:
        import torch
        def inspect(value, rel):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == 'replay_count' and item != 0:
                        issues.append(dict(file=str(rel), reason='nonempty_replay'))
                    if key == 'replay' and isinstance(item, dict) and any(
                            isinstance(v, torch.Tensor) and v.numel() for v in item.values()):
                        issues.append(dict(file=str(rel), reason='nonempty_replay_tensor'))
                    inspect(item, rel)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    inspect(item, rel)
            elif isinstance(value, str):
                if any(pattern.search(value) for pattern in patterns.values()):
                    issues.append(dict(file=str(rel), reason='checkpoint_sensitive_metadata'))
        for path in files:
            if path.suffix == '.pth':
                state = torch.load(path, map_location='cpu', weights_only=True)
                inspect(state, path.relative_to(root))
                checked_checkpoints += 1
    return dict(files_checked=len(files), checkpoints_checked=checked_checkpoints,
                issues=issues, passed=not issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-only', action='store_true')
    parser.add_argument('--refresh-manifest', action='store_true', help='For an intentional new release only')
    args = parser.parse_args()
    result = scan(checkpoints=not args.text_only)
    if not result['passed']:
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    if args.refresh_manifest:
        manifest_path = ROOT / 'RELEASE_MANIFEST.json'
        files = {}
        for p in ROOT.rglob('*'):
            rel = p.relative_to(ROOT)
            if not p.is_file() or any(x in ('outputs', '.venv', '__pycache__', '.pytest_cache', '.git') for x in rel.parts):
                continue
            if p == manifest_path:
                continue
            files[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(dict(version='1.0.0', files=dict(sorted(files.items())),
                                                raw_field_data_included=False), indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
