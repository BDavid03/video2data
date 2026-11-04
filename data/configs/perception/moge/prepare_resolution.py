"""Read public static package metadata; never import or execute fetched code."""
import hashlib
import json
from pathlib import Path
import tomllib
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'data/configs/perception/moge'
SOURCES = {
    'moge': ('microsoft/MoGe', '07444410f1e33f402353b99d6ccd26bd31e469e8'),
    'utils3d': ('EasternJournalist/utils3d', '3fab839f0be9931dac7c8488eb0e1600c236e183'),
    'pipeline': ('EasternJournalist/pipeline', '866f059d2a05cde05e4a52211ec5051fd5f276d6'),
}
records = []
requirements = set()
for name, (repo, revision) in SOURCES.items():
    url = f'https://raw.githubusercontent.com/{repo}/{revision}/pyproject.toml'
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
    (OUT / f'{name}.pyproject.toml').write_bytes(raw)
    parsed = tomllib.loads(raw.decode())
    dependencies = parsed['project'].get('dependencies', [])
    records.append({'name': name, 'repository': f'https://github.com/{repo}',
                    'revision': revision, 'metadata_url': url,
                    'metadata_sha256': hashlib.sha256(raw).hexdigest(),
                    'version': parsed['project']['version'],
                    'requires_python': parsed['project'].get('requires-python'),
                    'dependencies': dependencies, 'build_system': parsed['build-system']})
    requirements.update(d for d in dependencies if 'git+' not in d)
requirements.update(['setuptools==80.10.2', 'wheel==0.47.0'])
(OUT / 'requirements.in').write_text(
    '# Static declared dependencies of the three source pins; source projects remain separate.\n'
    '# Torch and torchvision are intentionally excluded by compile command and reused read-only.\n'
    + '\n'.join(sorted(requirements)) + '\n', encoding='utf-8')
(ROOT / 'data/manifests/models/moge-source-pins.json').write_text(json.dumps({
    'status': 'static_metadata_only_no_source_execution', 'sources': records,
    'source_install_status': 'not_installed; future source installation needs approval and code review',
    'external_reuse': {'torch': '2.12.1+cu132', 'torchvision': '0.27.1+cu132',
                      'interpreter': 'Z:/hugging-face/locals/.venv/Scripts/python.exe',
                      'access': 'read_only; no modifications'},
}, indent=2) + '\n', encoding='utf-8')
print('Read three static pyproject files; wrote source pins and resolver input. No installation.')
