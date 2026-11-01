"""Explicit metadata inventories and immutable candidate freeze receipts; no producers."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from uuid import uuid4

from track3.paths import output_path, project_path
from track3.provenance import input_file

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).parent
CHANNELS = ("teacher_agreement", "independent_geometry", "reference_diagnostic",
            "achieved_behaviour", "runtime_resources")
CHOICES = ("models", "ensemble_weights", "thresholds", "geometry", "policy")
ROLES = ("depth", "camera", "identity", "object_pose", "surface", "hands",
         "contact", "scene", "retargeting", "policy", "achieved_rollout")
ROLE_KINDS = {
    "depth": {"depth.generate", "depth.ingest_challenger", "scratch_depth.predict",
              "learning_experiments.ensemble"},
    "camera": {"camera_motion.estimate"},
    "identity": {"object_masks.generate", "object_masks.correspondences"},
    "object_pose": {"object_pose.compose", "object_pose.refine"},
    "surface": {"object_mesh.reconstruct", "object_mesh.fuse"},
    "hands": {"hand_motion.lift", "hand_motion.refine"},
    "contact": {"hand_motion.contacts"},
    "scene": {"scene_assembly.assemble", "scene_assembly.refine"},
    "retargeting": {"retargeting.retarget"},
    "policy": {"policy_training.train"},
    "achieved_rollout": {"policy_rollout.rollout"},
}
TRAIN = (0, 11, 12, 18, 21, 23, 26, 31)
TUNING = (39, 42)
CONSUMED = (43, 44)
DISPOSITIONS = ("investigated", "implemented", "experimentally_assessed",
                "rejected", "deferred")
# Intentionally small allowlist: do not open datasets, predictions, checkpoints,
# labels, meshes or arbitrary JSON payloads disguised as evidence discovery.
METADATA_NAMES = {"run.json", "result.json", "selection.json", "frozen-selection.json",
                  "scorecard.json", "scorecards.json", "manifest.json", "reels.json",
                  "inspection.json", "reel-manifest.json", "geometry.json", "report.json"}


@dataclass(frozen=True)
class EvidenceRequest:
    records: list = field(default_factory=list)
    dispositions: list = field(default_factory=list)
    candidate: dict | None = None

    @classmethod
    def from_file(cls, path):
        return cls(**json.loads(project_path(ROOT, path).read_text(encoding="utf-8-sig")))

    def __post_init__(self):
        if type(self.records) is not list or type(self.dispositions) is not list:
            raise ValueError("records and dispositions must be explicit lists")
        if len(self.records) > 128 or len(self.dispositions) > 15:
            raise ValueError("At most 128 explicit records and 15 item dispositions")
        if self.candidate is not None and type(self.candidate) is not dict:
            raise ValueError("candidate must be a declared object or null")
        json.dumps(asdict(self), allow_nan=False)


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("An exact lowercase SHA-256 is required")
    return value


def _metadata(reference):
    """Read only one explicitly named, bounded metadata record, never discover files."""
    if type(reference) is not dict or set(reference) != {"path", "sha256"}:
        raise ValueError("Metadata reference must contain exactly path and sha256")
    path = project_path(ROOT, reference["path"])
    relative = path.relative_to(ROOT).as_posix()
    if (not relative.startswith(("outputs/", "runs/")) or path.name not in METADATA_NAMES):
        raise ValueError("Evidence must name explicit known metadata under outputs/ or runs/")
    # Read limit also bounds memory if a misnamed file is supplied.
    with path.open("rb") as stream:
        payload = stream.read(4 * 1024 * 1024 + 1)
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("Evidence metadata exceeds the 4 MiB limit")
    identity = {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}
    if identity["sha256"] != _digest(reference["sha256"]):
        raise ValueError(f"Metadata identity changed: {relative}")
    document = json.loads(payload.decode("utf-8-sig"))
    if type(document) is not dict:
        raise ValueError("Evidence metadata must be an object")
    return identity, document, len(payload)


def _normalized(reference):
    if type(reference) is not dict or set(reference) != {"path", "sha256"}:
        raise ValueError("Lineage reference requires exactly path and sha256")
    return {"path": project_path(ROOT, reference["path"]).relative_to(ROOT).as_posix(),
            "sha256": _digest(reference["sha256"])}


def _episode_list(values):
    if (type(values) is not list or not values
            or any(type(value) is not int or value < 0 or value == 41 for value in values)
            or len(values) != len(set(values))):
        raise ValueError("Declare unique nonnegative episodes; episode 41 remains excluded")
    return values


def _record_episodes(record):
    """Use recorded episode identities, never caller-assigned evidence labels alone."""
    episodes = set()
    containers = [record]
    if type(record.get("request")) is dict:
        containers.append(record["request"])
    for container in containers:
        for key in ("episode_index", "episode_indices", "train_episode_indices",
                    "tuning_episode_indices", "holdout_episode_indices"):
            if key not in container or container[key] is None:
                continue
            value = container[key]
            values = value if type(value) is list else [value]
            if any(type(item) is not int or item < 0 for item in values):
                raise ValueError("Recorded episode identity must contain nonnegative integers")
            episodes.update(values)
    return episodes


def _inventory(rows):
    identities, documents, records = {}, {}, []
    total_bytes = 0
    for row in rows:
        if type(row) is not dict or set(row) != {"record", "channel", "claim", "limitations", "episodes"}:
            raise ValueError("Evidence row needs record/channel/claim/limitations/episodes")
        if (row["channel"] not in CHANNELS
                or any(type(row[key]) is not str or not 1 <= len(row[key]) <= 4096
                       for key in ("claim", "limitations"))):
            raise ValueError("Name an evidence channel, supported claim and limitations")
        episodes = _episode_list(row["episodes"])
        identity, document, size = _metadata(row["record"])
        total_bytes += size
        if total_bytes > 64 * 1024 * 1024:
            raise ValueError("Combined evidence metadata exceeds the 64 MiB ceiling")
        recorded_episodes = _record_episodes(document)
        if 41 in recorded_episodes:
            raise ValueError("Episode 41 metadata is excluded")
        if recorded_episodes and not recorded_episodes <= set(episodes):
            raise ValueError("Evidence row omits recorded episode identities")
        key = identity["path"]
        if key in documents:
            raise ValueError("Inventory each metadata record once; link it by path elsewhere")
        identities[key], documents[key] = identity, document
        records.append({**row, "record": identity, "episodes": episodes,
                        "recorded_status": document.get("status", "unestablished"),
                        "holdout_consumed": bool(set(episodes) & set(CONSUMED)),
                        "binary_artifacts_opened": False})
    return identities, documents, records


def _dispositions(rows):
    result = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"item", "approaches", "blockers"}:
            raise ValueError("Disposition needs item, approaches and explicit blockers")
        item = row["item"]
        if type(item) is not int or item not in range(1, 16) or item in result:
            raise ValueError("Disposition items must be unique integers 1 through 15")
        if type(row["blockers"]) is not list or type(row["approaches"]) is not list or not row["approaches"]:
            raise ValueError("Each item needs approach dispositions and a blocker list")
        names = set()
        for approach in row["approaches"]:
            if (type(approach) is not dict or set(approach) != {"name", "status", "reason", "evidence"}
                    or type(approach["name"]) is not str or not approach["name"]
                    or approach["name"] in names or approach["status"] not in DISPOSITIONS
                    or type(approach["reason"]) is not str or not approach["reason"]
                    or type(approach["evidence"]) is not list):
                raise ValueError("Every unique named approach needs status, reason and evidence references")
            if approach["status"] in ("experimentally_assessed", "rejected") and not approach["evidence"]:
                raise ValueError("Assessed/rejected approaches need exact evidence references")
            names.add(approach["name"])
        result[item] = row
    return [result[item] for item in sorted(result)]


def _candidate(candidate, identities, documents, records):
    """Freeze only an admitted explicit chain; missing evidence remains a blocked report.

    Metadata admission, not numerical/checkpoint verification: producer-declared binary
    hashes are preserved without opening their payloads.
    """
    if candidate is None:
        return None, ["No production candidate was declared"]
    required = {"id", "choices", "roles", "tuning_evidence", "decision_rule", "exports"}
    if set(candidate) != required:
        raise ValueError(f"Candidate fields must be exactly {sorted(required)}")
    if type(candidate["id"]) is not str or re.fullmatch(r"[A-Za-z0-9._-]+", candidate["id"]) is None:
        raise ValueError("Candidate id must be a filesystem-safe name")
    if type(candidate["decision_rule"]) is not str or not candidate["decision_rule"]:
        raise ValueError("Declare the tuning-only decision rule before outcomes")
    choices, roles = candidate["choices"], candidate["roles"]
    if type(choices) is not dict or set(choices) != set(CHOICES):
        raise ValueError(f"Declare every frozen choice: {CHOICES}")
    if type(roles) is not dict or set(roles) != set(ROLES):
        raise ValueError(f"Declare exact producer references for all roles: {ROLES}")
    blockers, selected = [], {}
    rollout_root, role_episode = _role_blockers(roles, identities, documents, selected, blockers)
    _choice_blockers(choices, identities, documents, selected, blockers)
    blockers.extend(_lineage(selected, identities, documents, rollout_root))
    _tuning_blockers(candidate["tuning_evidence"], choices, identities, documents, records, blockers)
    _export_blockers(candidate["exports"], roles, role_episode, identities, documents, records, blockers)
    return candidate, blockers


def _role_blockers(roles, identities, documents, selected, blockers):
    """Admit each role producer into ``selected``; return the rollout root and the shared episode set."""
    rollout_root = None
    role_episode = set()
    role_split = set()
    for role, reference in roles.items():
        if reference is None:
            blockers.append(f"Missing admitted {role} producer")
            continue
        identity = _normalized(reference)
        if identities.get(identity["path"]) != identity:
            blockers.append(f"{role}: producer absent from explicit inventory")
            continue
        record = documents[identity["path"]]
        if (Path(identity["path"]).name != "run.json" or record.get("status") != "completed"
                or record.get("provenance") != "video_only" or record.get("kind") not in ROLE_KINDS[role]):
            blockers.append(f"{role}: needs a completed video-only producer run.json")
        episodes = _record_episodes(record)
        if not episodes or episodes & {41, *CONSUMED}:
            blockers.append(f"{role}: needs explicit eligible recorded episode identity")
        # One coherent episode chain per freeze; campaign-wide freezes keep one receipt per episode.
        if role != "policy":
            role_episode.update(episodes)
            split = record.get("request", {}).get("source_split", record.get("source_split"))
            if split not in ("public", "evaluation"):
                blockers.append(f"{role}: missing producer split identity")
            else:
                role_split.add(split)
        selected[identity["path"]] = identity
        if role == "achieved_rollout":
            rollout_root = identity["path"]
    if len(role_episode) != 1 or len(role_split) != 1:
        blockers.append("Role producers must describe the same single eligible episode/split")
    return rollout_root, role_episode


def _choice_blockers(choices, identities, documents, selected, blockers):
    """Admit each choice's settings producer and the owners of its bound artifacts into ``selected``."""
    for name, choice in choices.items():
        if (type(choice) is not dict or set(choice) != {"settings", "settings_record", "artifacts", "reason"}
                or type(choice["settings"]) is not dict or not choice["settings"]
                or type(choice["artifacts"]) is not list or not choice["reason"]):
            blockers.append(f"{name}: exact settings/artifact bindings/reason are missing")
            continue
        settings_identity = _normalized(choice["settings_record"])
        settings_path = settings_identity["path"]
        settings_document = documents.get(settings_path, {})
        if (identities.get(settings_path) != settings_identity
                or settings_document.get("status") != "completed"
                or settings_document.get("provenance") != "video_only"
                or Path(settings_path).name != "run.json"
                or settings_document.get("request") != choice["settings"]):
            blockers.append(f"{name}: settings must equal the exact completed settings producer request")
        else:
            selected[settings_path] = settings_identity
        if not choice["artifacts"]:
            blockers.append(f"{name}: requires exact producer-owned choice artifacts")
        for reference in choice["artifacts"]:
            artifact_identity = _normalized(reference)
            owner = _owner(artifact_identity, identities, documents)
            # Pretrained assets/source pins are raw inputs recorded by the settings producer.
            if (owner is None and settings_path in selected
                    and not artifact_identity["path"].startswith(("outputs/", "runs/"))
                    and any(_normalized(entry) == artifact_identity for entry in
                            [*settings_document.get("inputs", []), *settings_document.get("sources", [])])):
                owner = settings_path
            if owner is None:
                blockers.append(f"{name}: choice artifact is not bound by an admitted completed producer")
            else:
                selected[owner] = identities[owner]


def _tuning_blockers(tuning, choices, identities, documents, records, blockers):
    """Selection evidence must be completed independent geometry covering exactly the tuning episodes."""
    if type(tuning) is not list or not tuning:
        blockers.append("Missing predeclared tuning evidence")
        return
    supported = set()
    for reference in tuning:
        identity = _normalized(reference)
        row = next((row for row in records if row["record"] == identity), None)
        owner = _owner(identity, identities, documents)
        document = documents.get(identity["path"], {})
        if (row is None or row["channel"] != "independent_geometry"
                or not set(row["episodes"]) <= set(TUNING)
                or _record_episodes(document) != set(row["episodes"])
                or owner is None or documents[owner].get("kind") != "stereo_depth.report"
                or document.get("schema") != "independent-geometry-report-v1"
                or document.get("channel") != "independent_image_geometry"
                or document.get("holdback_partition") != "sha256-episode-frame-query-v1"
                or not _supported_geometry(document)):
            blockers.append("Selection requires completed non-label independent geometry on tuning only")
        else:
            supported.update(row["episodes"])
            model_bindings = choices["models"].get("artifacts", []) if type(choices["models"]) is dict else []
            blockers.extend(_lineage({owner: identities[owner]}, identities, documents, owner, model_bindings))
    if supported != set(TUNING):
        blockers.append("Independent tuning support must include both 39 and 42")


def _export_blockers(exports, roles, role_episode, identities, documents, records, blockers):
    """Each export must be owned by a completed producer of the selected episode descending from the scene."""
    if type(exports) is not dict or set(exports) != {"scorecards", "event_reels", "fixed_scale_comparison"}:
        raise ValueError("Declare scorecards, event_reels and fixed_scale_comparison metadata references")
    for name, reference in exports.items():
        identity = None if reference is None else _normalized(reference)
        if identity is None or identities.get(identity["path"]) != identity:
            blockers.append(f"Missing exact {name} metadata; no exported output is claimed")
            continue
        owner = _owner(identity, identities, documents)
        row = next(row for row in records if row["record"] == identity)
        if owner is None or set(row["episodes"]) != role_episode:
            blockers.append(f"{name}: needs a completed video-only export owner for the selected episode")
            continue
        required_exports = {}
        if roles["scene"] is not None:
            scene_identity = _normalized(roles["scene"])
            if scene_identity["path"] in identities:
                required_exports[scene_identity["path"]] = scene_identity
        if not required_exports:
            blockers.append(f"{name}: no admitted selected scene to bind exported evidence")
        blockers.extend(_lineage(required_exports, identities, documents, owner))


def _supported_geometry(document):
    """Require reported reserved support; an empty report is not tuning evidence."""
    if not document.get("fixed_metric_scales") or not document.get("protocol"):
        return False
    for row in document.get("frames", []):
        if row.get("status") != "completed":
            continue
        for stratum in row.get("metrics", {}).get("candidate", {}).values():
            summary = stratum.get("log_depth", {})
            if (type(summary.get("denominator")) is int and summary["denominator"] > 0
                    and summary.get("median_absolute") is not None):
                return True
    return False


def _owner(identity, identities, documents):
    """An artifact is admitted by a completed producer's recorded hash, without reading it."""
    for path, record in documents.items():
        if (Path(path).name != "run.json" or record.get("status") != "completed"
                or record.get("provenance") != "video_only"):
            continue
        if identities[path] == identity:
            return path
        for name, digest in _declared_artifacts(record):
            owner_directory = project_path(ROOT, Path(path).parent)
            artifact_path = project_path(ROOT, owner_directory / name)
            if not artifact_path.is_relative_to(owner_directory) or artifact_path == owner_directory:
                raise ValueError("Producer artifact ownership must stay inside its run directory")
            artifact = artifact_path.relative_to(ROOT).as_posix()
            if artifact == identity["path"] and digest == identity["sha256"]:
                return path
    return None


def _declared_artifacts(record):
    """Known producer-owned inventories only; do not guess paths or scan outputs."""
    yield from record.get("artifacts", {}).items()
    for row in record.get("frames", []):
        if row.get("status") == "completed" and "artifact" in row and "sha256" in row:
            yield row["artifact"], row["sha256"]
    for row in record.get("checkpoints", []):
        name = row.get("artifact", row.get("file"))
        if name is not None and "sha256" in row:
            yield name, row["sha256"]
    if (record.get("kind") == "depth.generate"
            and "preprocessing_artifact_sha256" in record):
        yield "preprocessing.npz", record["preprocessing_artifact_sha256"]


def _lineage(selected, identities, documents, rollout_root, required_bindings=()):
    """Traverse only admitted in-memory producer metadata, including exact parents."""
    blockers, visited, active = [], set(), set()

    def visit(path):
        if path in active:
            blockers.append(f"Cyclic producer metadata lineage at {path}")
            return
        if path in visited:
            return
        active.add(path)
        record = documents[path]
        if (type(record.get("sources")) is not list or not record["sources"]
                or type(record.get("inputs")) is not list):
            blockers.append(f"{path}: producer lacks exact sources/inputs")
        else:
            if not any(entry.get("path") == "configs/sources.lock.json" for entry in record["sources"]):
                blockers.append(f"{path}: missing exact source lock identity")
            for reference in [*record["sources"], *record["inputs"]]:
                identity = _normalized(reference)
                if identity["path"].startswith(("outputs/", "runs/")):
                    owner = _owner(identity, identities, documents)
                    if owner is None:
                        blockers.append(f"{path}: unadmitted derived lineage {identity['path']}")
                    elif owner != path:
                        visit(owner)
                    else:
                        blockers.append(f"{path}: producer consumes its own output")
        if _record_episodes(record) & {41, *CONSUMED}:
            blockers.append(f"{path}: excluded/consumed episode in candidate ancestry")
        active.remove(path)
        visited.add(path)

    if rollout_root is not None:
        visit(rollout_root)
    declared = []
    for path in visited:
        record = documents[path]
        declared.extend(_normalized(entry) for entry in [*record.get("inputs", []), *record.get("sources", [])])
    for reference in required_bindings:
        if _normalized(reference) not in declared:
            blockers.append("Independent tuning report ancestry does not bind the exact selected model assets")
    for path in selected:
        if path not in visited:
            blockers.append(f"{path}: selected producer is not an ancestor of achieved rollout")
    # Still collect independent failures for disconnected choices/roles.
    for path in selected:
        visit(path)
    return blockers


def campaign_evidence(operation: str, request: EvidenceRequest):
    if operation not in ("evidence", "freeze"):
        raise ValueError("Choose evidence or freeze")
    runtime_path = FEATURE / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8-sig"))
    if runtime["execution"].get(operation + "_authorized") is not True:
        raise PermissionError(f"Campaign {operation} remains disabled")
    # output_path gates canonical storage before reading any campaign evidence.
    destination_root = output_path(ROOT, f"outputs/campaign/{operation}")
    identities, documents, records = _inventory(request.records)
    dispositions = _dispositions(request.dispositions)
    blockers = [f"Missing disposition for item {item:02d}" for item in range(1, 16)
                if item not in {row["item"] for row in dispositions}]
    catalog_path = FEATURE / "evidence.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))["dispositions"]
    for expected in catalog:
        actual = next((row for row in dispositions if row["item"] == expected["item"]), None)
        names = {approach["name"] for approach in actual["approaches"]} if actual else set()
        blockers.extend(f"Item {expected['item']:02d}: missing approach {approach['name']}"
                        for approach in expected["approaches"] if approach["name"] not in names)
    for row in dispositions:
        blockers.extend(f"Item {row['item']:02d}: {value}" for value in row["blockers"])
        for approach in row["approaches"]:
            for reference in approach["evidence"]:
                identity = _normalized(reference)
                if identities.get(identity["path"]) != identity:
                    raise ValueError("Disposition evidence must occur in the exact explicit inventory")
    candidate, missing = _candidate(request.candidate, identities, documents, records)
    blockers.extend(missing)
    # Only freeze requests can seal choices, and no blocked request becomes frozen.
    frozen = operation == "freeze" and not blockers
    timestamp = datetime.now(timezone.utc).isoformat()
    source_paths = [Path(__file__), FEATURE / "__main__.py", FEATURE / "__init__.py",
                    ROOT / "track3/paths.py", runtime_path, catalog_path,
                    ROOT / "configs/sources.lock.json"]
    sources = [input_file(ROOT, path) for path in source_paths]
    manifest = {"schema_version": 1, "kind": f"campaign.{operation}",
                "status": "frozen_metadata" if frozen else "blocked",
                "created_at": timestamp, "request": asdict(request), "sources": sources,
                "inventory": records, "dispositions": dispositions,
                "candidate": candidate, "blockers": blockers,
                "partitions": {"train": TRAIN, "tuning": TUNING,
                               "consumed_holdout": CONSUMED, "excluded": [41]},
                "fresh_holdout_eligible": False, "fresh_holdout_protocol": None,
                "production_promoted": False, "pins_changed": False,
                "submission_uploaded": False, "producers_launched": False,
                "binary_artifacts_verified": False,
                "limitations": ["Metadata admission preserves producer-declared binary hashes; payloads were not opened",
                                "A metadata freeze is not numerical verification or production approval",
                                "Consumed 43/44 cannot select recipes, mixtures or a new final assessment",
                                "A fresh unbiased assessment needs an independently fixed protocol"]}
    payload = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    directory = destination_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                                    + "-" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    # Exclusive creation and content addressing: never replace an earlier receipt.
    path = directory / f"{digest}.json"
    with path.open("xb") as stream:
        stream.write(payload)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest,
            "status": manifest["status"], "blockers": blockers}
