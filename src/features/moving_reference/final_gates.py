"""Write and verify the one final M1 record without changing sealed references."""

import argparse
import hashlib
import json
from pathlib import Path

from ._final_gates import compose_final, KIND, RULE

ROOT = Path(__file__).resolve().parents[3]
SEALED = ROOT / 'data/outputs/moving_reference/ep39/sealed/sealed.json'
REFERENCES = {
    39: '44cd03d9da3a9b017610eab00439662eb8d7bc83ff00bd0a386cff618d30b3e1',
    38: 'c5da9fbe4802a62ccfeca8627f2b9aa69a2acf14eaeecbc565b3a728fb275f12',
}
EARLIER = ('gates.json', 'gates-amended.json', 'gates-amended-2.json', 'gates-amended-3.json')


def owned(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def entry(path: Path) -> dict:
    path = owned(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path))


def read(path: Path) -> dict:
    return json.loads(owned(path).read_text())


def checked(value: dict) -> Path:
    path = owned(value['path'])
    if digest(path) != value['sha256']:
        raise ValueError(f'Final M1 input hash mismatch: {path}')
    return path


def expected_record(episode: int, directory: Path, sealed_file: Path = SEALED) -> dict:
    """Bind the unchanged inputs and sealed decision to the pure final gate calculation."""
    if episode not in REFERENCES:
        raise ValueError('Final M1 records admit only episodes 39 and 38')
    directory, sealed_file = owned(directory), owned(sealed_file)
    sealed = read(sealed_file)
    source = read(directory / EARLIER[-1])
    final = compose_final(source, sealed)
    if source.get('episode') != episode or sealed.get('reference_sha256') != REFERENCES[39]:
        raise ValueError('Final M1 episode or sealed reference identity mismatch')
    reference = entry(directory / 'reference.npz')
    if reference['sha256'] != REFERENCES[episode] or source['reference_sha256'] != reference['sha256']:
        raise ValueError('Sealed M1 reference changed')
    for name, expected in source['preserved_inputs'].items():
        if digest(directory / name) != expected:
            raise ValueError(f'Preserved M1 input changed: {name}')
    earlier = [entry(directory / name) for name in EARLIER]
    original = read(directory / 'gates.json')
    if original.get('label_free') is not True:
        raise ValueError('Original M1 provenance is not label-free')
    final['finalization'] = dict(kind=KIND, rule=RULE, earlier_records=earlier,
                               sealed=entry(sealed_file), reference=reference,
                               label_free=original['label_free'], sealed_condition_verified=True)
    return final


def verify_final(file: Path, episode: int) -> dict:
    """Rebuild the final decision from its hashed inputs, refusing forged sealed conditions."""
    file = owned(file)
    data = read(file)
    provenance = data.get('finalization', {})
    if provenance.get('kind') != KIND:
        raise ValueError('Final M1 producer provenance is missing')
    sealed_file = checked(provenance['sealed'])
    # Check the decision before the canonical-path check so a forged failed seal is explicit.
    expected = expected_record(episode, file.parent, sealed_file)
    if sealed_file != SEALED:
        raise ValueError('Final M1 must use the actual canonical sealed decision')
    if data != expected:
        raise ValueError('Final M1 record differs from its recomputed sealed-conditional decision')
    if data.get('passed') is not True:
        raise ValueError('Final M1 requires every gate to pass')
    return data


def finalize(episode: int) -> Path:
    """Create one deterministic final gate record, or verify an identical existing record."""
    directory = ROOT / f'data/outputs/moving_reference/ep{episode}/candidate-c'
    data = expected_record(episode, directory)
    output = directory / 'gates-final.json'
    if output.exists():
        verify_final(output, episode)
        return output
    with output.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
    verify_final(output, episode)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--episode', type=int, choices=(39, 38))
    args = parser.parse_args()
    for episode in ((args.episode,) if args.episode is not None else (39, 38)):
        file = finalize(episode)
        print(f'Episode {episode}: PASS, {len(read(file)["gates"])} gates; {file}')


if __name__ == '__main__':
    main()
