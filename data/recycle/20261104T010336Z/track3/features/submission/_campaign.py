"""Submission's read-only integration with the campaign selection contract."""

from track3.features.campaign.selections import read_pin_history, role_pins, run_role, tracked_scene_pin
from track3.provenance import input_file
from ._common import ROOT, checked_entry, local_path, producer, read_json, record_input


class _Metadata(dict):
    """Load only run metadata needed to classify legacy pin roles."""

    def __init__(self):
        super().__init__()
        self.observed = {}
        self.problems = []

    def get(self, path, default=None):
        if path not in self:
            try:
                record = read_json(path)
                if (not isinstance(record, dict) or not isinstance(record.get("kind"), str) or
                        not isinstance(record.get("request"), dict) or not isinstance(record.get("inputs"), list) or
                        any(not isinstance(item, dict) or not isinstance(item.get("path"), str)
                            for item in record["inputs"])):
                    raise ValueError("Incomplete producer metadata for role classification")
                self.observed[path] = input_file(ROOT, local_path(path))
                self[path] = record
            except (OSError, ValueError, TypeError) as error:
                self[path] = None
                self.problems.append({"run": path, "reason": str(error)})
        return self[path] if self[path] is not None else default


class CampaignSelections:
    """Pin history is advisory; selected producer integrity remains mandatory."""

    def __init__(self, record):
        self.record = record
        history, self.log = read_pin_history(ROOT, "outputs/campaign/pins.jsonl")
        self.history = [item for item in history if item["split"] == "evaluation"]
        if self.log is not None:
            record["inputs"].append(self.log)
        record_input(input_file(ROOT, "track3/features/campaign/selections.py"), record, "sources")
        self.metadata = _Metadata()
        self.pins = role_pins(self.history, self.metadata)
        self.memo = {}

    def full_scene(self, episode):
        pin = tracked_scene_pin(self.history, "evaluation", episode, self.metadata.get)
        if pin is None:
            return None
        path = checked_entry({"path": pin["run"], "sha256": pin["sha256"]}, self.record)
        scene = producer(path, "scene_assembly.assemble", episode)
        if run_role(scene[1], self.metadata) != "tracked":
            raise ValueError("Pinned tracked-scene role differs from its producer request")
        return scene

    def stale_warnings(self, parent, trail=frozenset()):
        path, source = parent
        relative = path.relative_to(ROOT).as_posix()
        if relative in trail:
            raise ValueError("Selected submission lineage contains a dependency cycle")
        if relative in self.memo:
            return self.memo[relative]
        role = run_role(source, self.metadata)
        key = (source.get("split"), source.get("episode_index"), source["kind"], role)
        entry = input_file(ROOT, path)
        record_input(entry, self.record)
        findings = []
        current = self.pins.get(key)
        if current and (current["run"] != entry["path"] or current["sha256"] != entry["sha256"]):
            findings.append({"reason": "pin_changed_for_role", "role": role,
                             "input": entry, "current_pin": current})
        unknown = self.pins.get((*key[:3], "unknown"))
        if unknown:
            findings.append({"reason": "unclassifiable_campaign_pin", "input": entry, "pin": unknown})
        for item in source["inputs"]:
            if not item["path"].endswith("/run.json"):
                continue
            upstream_path = checked_entry(item, self.record)
            upstream = producer(upstream_path)
            inherited = self.stale_warnings(upstream, trail | {relative})
            if inherited:
                findings.append({"input": item, "chain": inherited})
        self.memo[relative] = findings
        return findings

    def snapshot(self):
        return {"schema_version": 1, "pin_log": self.log,
                "role_pins": [{"split": key[0], "episode_index": key[1], "artifact_name": key[2],
                               "role": key[3], "pin": pin} for key, pin in sorted(self.pins.items())],
                "metadata_inputs": list(self.metadata.observed.values()),
                "metadata_problems": self.metadata.problems,
                "scope": "Read-only campaign selection metadata; no candidate is selected by this snapshot"}
