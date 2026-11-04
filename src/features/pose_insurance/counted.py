"""Archive packed checkpoints and check the durable submission ledger."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from src.provenance import sha256_file


def entry(root, path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(root)), size_bytes=path.stat().st_size,
                sha256=sha256_file(path))


def checked(root, item):
    path = (root / item['path']).resolve()
    path.relative_to(root)
    if sha256_file(path) != item['sha256']:
        raise ValueError(f'Hash mismatch: {item["path"]}')
    if 'size_bytes' in item and path.stat().st_size != item['size_bytes']:
        raise ValueError(f'Size mismatch: {item["path"]}')
    return path


def checksum_entries(root, item):
    path = checked(root, item)
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix('*')
        if Path(name).name != name or len(digest) != 64 or name in result:
            raise ValueError(f'Invalid SHA256SUMS entry: {line}')
        result[name] = dict(path=str((path.parent / name).relative_to(root)), sha256=digest)
    if not result:
        raise ValueError('Empty checkpoint SHA256SUMS')
    return result


def check_ledger(root, ledger_path):
    """Print every file verdict and return a failure count without repairs."""
    ledger = json.loads(ledger_path.read_text())
    seen = {}

    def check(item):
        name = item['path']
        if name in seen:
            if seen[name][0] != item['sha256']:
                raise ValueError(f'Conflicting ledger hashes: {name}')
            return seen[name][1]
        path = (root / name).resolve()
        path.relative_to(root)
        if not path.is_file():
            status = 'MISSING'
        elif sha256_file(path) != item['sha256'] or (
                'size_bytes' in item and path.stat().st_size != item['size_bytes']):
            status = 'CHANGED'
        else:
            status = 'OK'
        print(f'{status} {name}')
        seen[name] = (item['sha256'], status == 'OK')
        return status == 'OK'

    bad = 0
    for row in ledger['entries']:
        for key in ('file', 'manifest', 'archived_manifest'):
            bad += not check(row[key])
        sums = row.get('model_hashes')
        if sums and check(sums):
            for item in checksum_entries(root, sums).values():
                bad += not check(item)
        elif sums:
            bad += 1
    print(f'SUMMARY {len(ledger["entries"])} ledger entries, {len(seen)} unique files, {bad} problems')
    return bad


def copy_unchanged(source, destination, expected):
    """Copy exclusively or accept exactly matching existing bytes."""
    if sha256_file(source) != expected:
        raise ValueError(f'Archive source changed: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected:
            raise FileExistsError(f'Refusing to overwrite differing file: {destination}')
    else:
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    if sha256_file(destination) != expected:
        raise ValueError(f'Archive copy mismatch: {destination}')


def parent_checkpoints(root, packed_hash):
    ledger = json.loads((root / 'docs/submissions/counted.json').read_text())
    row = next(row for row in ledger['entries'] if row['file']['sha256'] == packed_hash)
    return checksum_entries(root, row['model_hashes'])


def archive_pack(root, target, manifest_path, checkpoints):
    """Copy all counted policies, preserve the manifest, then append an unuploaded pack."""
    root = Path(root).resolve()
    target = Path(target).resolve()
    manifest_path = Path(manifest_path).resolve()
    target.relative_to(root)
    manifest_path.relative_to(root)
    pack = target.parent.name
    manifest = json.loads(manifest_path.read_text())
    checked(root, manifest['file'])
    if target != root / manifest['file']['path']:
        raise ValueError('Manifest names a different packed file')
    directory = root / 'models/policies' / pack
    lines = []
    for name, item in sorted(checkpoints.items()):
        if Path(name).name != name or not name.endswith('.safetensors'):
            raise ValueError('Expected a checkpoint basename')
        copy_unchanged(checked(root, item), directory / name, item['sha256'])
        lines.append(f'{item["sha256"]}  {name}\n')
    if not lines:
        raise ValueError('A policy pack needs checkpoint hashes')
    sums = directory / 'SHA256SUMS'
    content = ''.join(lines)
    if sums.exists() and sums.read_text() != content:
        raise FileExistsError(f'Refusing to overwrite differing file: {sums}')
    if not sums.exists():
        with sums.open('x') as stream:
            stream.write(content)
    archived = root / 'docs/submissions/manifests' / pack / manifest_path.name
    copy_unchanged(manifest_path, archived, sha256_file(manifest_path))
    row = dict(pack_folder=str(target.parent.relative_to(root)), file_name=target.name,
               status='packed, not uploaded', file=entry(root, target),
               manifest=entry(root, manifest_path), archived_manifest=entry(root, archived),
               code_commit_url=manifest['code_commit_url'], model_hashes=entry(root, sums), uploads=[])
    ledger_path = root / 'docs/submissions/counted.json'
    ledger = json.loads(ledger_path.read_text())
    existing = next((r for r in ledger['entries'] if r['file']['path'] == row['file']['path']), None)
    if existing is not None:
        if {k: v for k, v in existing.items() if k not in ('status', 'uploads')} != {
                k: v for k, v in row.items() if k not in ('status', 'uploads')}:
            raise ValueError('Existing ledger entry differs from packed archive')
        return existing
    ledger['entries'].append(row)
    with tempfile.NamedTemporaryFile(mode='w', dir=ledger_path.parent, prefix='.counted-',
                                     suffix='.json', delete=False) as stream:
        json.dump(ledger, stream, indent=2)
        stream.write('\n')
        pending = Path(stream.name)
    pending.replace(ledger_path)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', default='docs/submissions/counted.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    raise SystemExit(1 if check_ledger(root, root / args.ledger) else 0)


if __name__ == '__main__':
    main()
