"""Verify the recorded GitHub snapshot using only Python's standard library."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'packaging/sha256_manifest.json').read_text())
    errors = []
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file():
            errors.append(f'Missing: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f'Changed: {name}')
    if errors:
        print('\n'.join(errors))
        sys.exit(1)
    print(f'OK: {len(manifest)} recorded files match the packaged snapshot.')
    print('Generated runs/checkpoints/results are not part of this manifest.')


if __name__ == '__main__':
    main()
