"""Prospective tuning-only geometry selection; no models or historical outcomes.

Receipts are explicit summaries of independent geometry, not teacher scores. A
receipt's declaration is not proof of measurement validity; its producer report
and protocol remain required review evidence. No calibration or fits occur here.
"""

from dataclasses import dataclass
import itertools
import math
import statistics
import time

from track3.paths import project_path
from track3.provenance import sha256_file, write_json
from ._common import ROOT, authorize, operation_run, read_json

SCHEMA = "independent-geometry-selection-v1"
EPISODES = (39, 42)
THRESHOLDS = (0, 0.025, 0.05, 0.1, 0.2, None)  # None means infinity in JSON.
GUARDRAILS = ("reprojection_error_px", "temporal_residual_m", "rigidity_drift_m")
RULE = "item11-geometry-90pct-5pct-v1"


@dataclass(frozen=True)
class GeometrySelectionRequest:
    protocol: dict
    candidates: tuple
    incumbent: str = "top2_equal"

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))


@dataclass(frozen=True)
class GeometryFreezeRequest:
    comparison: dict

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))


def _digest(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Expected lowercase SHA-256 identity")
    return value


def _finite(value, *, signed=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (signed or value >= 0))


def _load(entry, inputs, *, parse=True):
    if type(entry) is not dict or set(entry) != {"path", "sha256"}:
        raise ValueError("Evidence requires exact path/SHA-256 pairs")
    path = project_path(ROOT, entry["path"])
    digest = _digest(entry["sha256"])
    if path in inputs and inputs[path] != digest:
        raise ValueError("Conflicting evidence identities")
    if sha256_file(path) != digest:
        raise ValueError("Evidence bytes differ from declared identity")
    inputs[path] = digest
    if not parse:
        return None
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Selection JSON exceeds 64 MiB input ceiling")
    return read_json(path)


def _unchanged(inputs):
    if any(sha256_file(path) != digest for path, digest in inputs.items()):
        raise ValueError("Selection evidence changed during operation")


def _key(row):
    if (type(row) is not dict or row.get("episode_index") not in EPISODES
            or type(row.get("episode_index")) is not int
            or any(type(row.get(name)) is not str or not 1 <= len(row[name]) <= 256
                   for name in ("object_id", "event_id", "event_stratum", "query_id"))):
        raise ValueError("Every query needs a tuning episode/object/event/stratum/id")
    return tuple(row[name] for name in
                 ("episode_index", "object_id", "event_id", "event_stratum", "query_id"))


def _protocol(entry, inputs):
    value = _load(entry, inputs)
    if (value.get("schema") != SCHEMA or value.get("episode_indices") != list(EPISODES)
            or value.get("partition") != "tuning" or value.get("decision_rule") != RULE
            or value.get("predeclared") is not True
            or value.get("reference_labels_used") is not False
            or value.get("scale_fitting") is not False):
        raise ValueError("Require predeclared independent tuning protocol, fixed metric scale")
    for field in ("data_identity", "source_identity", "event_index_identity"):
        _load(value[field], inputs, parse=False)
    strata = value.get("event_strata")
    if (type(strata) is not list or not 2 <= len(strata) <= 32 or not {"static", "dynamic"}.issubset(strata)
            or any(type(s) is not str or not 1 <= len(s) <= 256 for s in strata) or len(set(strata)) != len(strata)):
        raise ValueError("Declare unique static/dynamic event strata before outcomes")
    queries = value.get("queries")
    if type(queries) is not list or not 1 <= len(queries) <= 10000:
        raise ValueError("Require bounded full reserved query universe, including unavailable queries")
    keys = [_key(row) for row in queries]
    if len(set(keys)) != len(keys) or any(key[3] not in strata for key in keys):
        raise ValueError("Duplicate queries or undeclared event strata")
    return value, keys


def _candidate(entry, protocol_entry, keys, inputs):
    definition = _load(entry["definition"], inputs)
    candidate_id = definition.get("candidate_id")
    if type(candidate_id) is not str or not 1 <= len(candidate_id) <= 256:
        raise ValueError("Frozen candidate ID required")
    for field in ("data_identity", "source_identity", "configuration_identity", "geometry_identity",
                  "anchor_identity", "preprocessing_identity"):
        _load(definition[field], inputs, parse=False)
    members = definition.get("members")
    if type(members) is not list or not 1 <= len(members) <= 8:
        raise ValueError("Declare every direct member and exact checkpoint/weights")
    ids = []
    for member in members:
        member_id = member.get("member_id")
        if type(member_id) is not str or not 1 <= len(member_id) <= 256 or not _finite(member.get("weight")):
            raise ValueError("Member identity and finite nonnegative weight required")
        ids.append(member_id)
        _load(member["checkpoint"], inputs, parse=False)
    if len(set(ids)) != len(ids) or not any(m["weight"] > 0 for m in members):
        raise ValueError("Unique members and positive total weight required")
    if ("thresholds" not in definition or definition.get("frozen_before_receipt") is not True
            or definition.get("consumed_holdout_used_for_selection") is not False):
        raise ValueError("Candidate choices must be frozen before receipt, without consumed holdout")
    receipt = _load(entry["receipt"], inputs)
    if (receipt.get("schema") != SCHEMA or receipt.get("partition") != "tuning"
            or receipt.get("episode_indices") != list(EPISODES)
            or receipt.get("protocol_identity") != protocol_entry
            or receipt.get("candidate_identity") != entry["definition"]
            or receipt.get("reference_labels_used") is not False
            or receipt.get("scale_fitting") is not False
            or receipt.get("metric_scale") != "metres"
            or receipt.get("clocks_aligned") is not True
            or receipt.get("reserved_stereo_support") is not True):
        raise ValueError("Receipt must bind fixed independent geometry, clocks, candidate and protocol")
    _load(receipt["source_report"], inputs, parse=False)
    rows = receipt.get("rows")
    if type(rows) is not list or len(rows) != len(keys):
        raise ValueError("Receipt must include every reserved query, including unavailable ones")
    mapped = {}
    for row in rows:
        key = _key(row)
        if key in mapped or type(row.get("observed")) is not bool:
            raise ValueError("Duplicate query or missing explicit support")
        for field in ("depth_log_residual", "depth_error_m", *GUARDRAILS):
            if row.get(field) is not None and not _finite(row[field], signed=field == "depth_log_residual"):
                raise ValueError("Residuals must be finite, or explicit null when unavailable")
        if row["observed"] and (row.get("depth_log_residual") is None or row.get("depth_error_m") is None):
            raise ValueError("Observed stereo query requires log and raw metric errors")
        for field in ("catastrophic_identity_loss", "new_invalid_event"):
            if row.get(field) is not None and type(row[field]) is not bool:
                raise ValueError("Catastrophe/invalid flags require bool or explicit unavailable null")
        residuals = row.get("member_log_residuals")
        if type(residuals) is not dict or set(residuals) != set(ids):
            raise ValueError("Every direct member needs a residual or explicit null")
        if any(v is not None and not _finite(v, signed=True) for v in residuals.values()):
            raise ValueError("Invalid member residual")
        count = row.get("member_count")
        if type(count) is not int or not 0 <= count <= len(ids):
            raise ValueError("Explicit positive-weight supported direct-member count required")
        expected_count = sum(residuals[m["member_id"]] is not None and m["weight"] > 0 for m in members)
        if count != expected_count:
            raise ValueError("Direct-member count contradicts supported member residuals")
        std = row.get("log_depth_std")
        if (std is not None and not _finite(std)) or (count < 2 and std is not None):
            raise ValueError("Disagreement unavailable below two direct supported members")
        mapped[key] = row
    if set(mapped) != set(keys):
        raise ValueError("Candidate support universe differs from protocol")
    seconds = receipt.get("complete_operation_seconds")
    if not _finite(seconds):
        raise ValueError("Measured complete-operation time required for predeclared tie breaking")
    return {"id": candidate_id, "definition": definition, "identity": entry["definition"],
            "receipt": entry["receipt"], "rows": mapped, "seconds": seconds,
            "members": len(members)}


def _quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)


def _summary(rows, total):
    supported = [r for r in rows if r["observed"]]
    return {"reserved_queries": total, "supported_queries": len(supported),
            "missing_queries": total - len(supported),
            "absolute_coverage": len(supported) / total if total else None,
            "median_absolute_log_error": statistics.median(abs(r["depth_log_residual"]) for r in supported) if supported else None,
            "median_error_m": statistics.median(r["depth_error_m"] for r in supported) if supported else None}


def _groups(keys, event_strata=()):
    # Every declared object/event receives denominators even when support is zero.
    groups = {("episode", episode): [] for episode in EPISODES}
    groups.update({("stratum", episode, stratum): [] for episode in EPISODES for stratum in event_strata})
    for key in keys:
        for group in (("episode", key[0]), ("object", key[0], key[1]),
                      ("event", key[0], key[1], key[2]), ("stratum", key[0], key[3])):
            groups.setdefault(group, []).append(key)
    return groups


def _diagnostics(candidate, keys, protocol):
    rows = candidate["rows"]
    support, curves = [], []
    for group, subset in _groups(keys, protocol["event_strata"]).items():
        support.append({"group": list(group), **_summary([rows[k] for k in subset], len(subset))})
        for threshold in THRESHOLDS:
            admitted = [rows[k] for k in subset if rows[k]["observed"] and rows[k]["member_count"] >= 2
                        and rows[k].get("log_depth_std") is not None
                        and (threshold is None or rows[k]["log_depth_std"] <= threshold)]
            curves.append({"group": list(group), "maximum_log_depth_std": threshold,
                           **_summary(admitted, len(subset))})
    pairs = []
    ids = [m["member_id"] for m in candidate["definition"]["members"] if m["weight"] > 0]
    for group, subset in _groups(keys, protocol["event_strata"]).items():
        for left, right in itertools.combinations(ids, 2):
            paired = [(rows[k]["member_log_residuals"][left], rows[k]["member_log_residuals"][right])
                      for k in subset if rows[k]["observed"]
                      and rows[k]["member_log_residuals"][left] is not None
                      and rows[k]["member_log_residuals"][right] is not None]
            pairs.append({"group": list(group), "members": [left, right],
                          "reserved_queries": len(subset), "paired_queries": len(paired),
                          "missing_paired_queries": len(subset) - len(paired),
                          "opposite_sign_fraction": sum(a * b < 0 for a, b in paired) / len(paired) if paired else None,
                          "mean_absolute_residual_difference": statistics.mean(abs(a - b) for a, b in paired) if paired else None,
                          "oracle_minimum_absolute_residual": statistics.mean(min(abs(a), abs(b)) for a, b in paired) if paired else None})
    return {"support": support, "error_versus_coverage": curves, "complementarity": pairs,
            "member_support": [{"member_id": member, "reserved_queries": len(keys),
                                "available_residual_queries": sum(rows[k]["member_log_residuals"][member] is not None for k in keys),
                                "missing_or_invalid_queries": sum(rows[k]["member_log_residuals"][member] is None for k in keys)} for member in ids]}


def _compare(candidate, incumbent, protocol, keys):
    rows, base = candidate["rows"], incumbent["rows"]
    failures, episodes = [], []
    for episode in EPISODES:
        for stratum in (None, *protocol["event_strata"]):
            subset = [k for k in keys if k[0] == episode and (stratum is None or k[3] == stratum)]
            old = sum(base[k]["observed"] for k in subset)
            new = sum(rows[k]["observed"] and base[k]["observed"] for k in subset)
            if not old or new < 0.9 * old:
                failures.append(f"coverage unavailable/below 90%: episode={episode}, stratum={stratum}")
        paired = [k for k in keys if k[0] == episode and rows[k]["observed"] and base[k]["observed"]]
        primary = [abs(rows[k]["depth_log_residual"]) for k in paired]
        previous = [abs(base[k]["depth_log_residual"]) for k in paired]
        result = {"episode_index": episode, "matched_queries": len(paired),
                  "candidate": _summary([rows[k] for k in paired], sum(k[0] == episode for k in keys)),
                  "incumbent": _summary([base[k] for k in paired], sum(k[0] == episode for k in keys))}
        if not paired or statistics.median(primary) >= statistics.median(previous):
            failures.append(f"primary not lower on matched support: episode={episode}")
        result["guardrails"] = {}
        for metric in GUARDRAILS:
            # Require every matched query's channel; never select a favourable subintersection.
            complete = bool(paired) and all(rows[k].get(metric) is not None and base[k].get(metric) is not None for k in paired)
            current = _quantile([rows[k][metric] for k in paired], 0.9) if complete else None
            prior = _quantile([base[k][metric] for k in paired], 0.9) if complete else None
            result["guardrails"][metric] = {"candidate_p90": current, "incumbent_p90": prior,
                                           "matched_queries": len(paired) if complete else 0}
            if not complete or current > prior * 1.05:
                failures.append(f"guardrail missing/over 5%: episode={episode}, metric={metric}")
        episodes.append(result)
    if any(r.get(field) is not False for r in rows.values()
           for field in ("catastrophic_identity_loss", "new_invalid_event")):
        failures.append("catastrophic identity loss/new invalid event present or evidence unavailable")
    return {"candidate_id": candidate["id"], "eligible": not failures, "failures": failures,
            "episodes": episodes, "primary": statistics.mean(e["candidate"]["median_absolute_log_error"] for e in episodes)
            if all(e["candidate"]["median_absolute_log_error"] is not None for e in episodes) else None}


def compare_geometry(request: GeometrySelectionRequest):
    authorize("geometry_selection")
    if not 1 <= len(request.candidates) <= 8:
        raise ValueError("At most eight predeclared candidates")
    started, inputs = time.monotonic(), {}
    protocol, keys = _protocol(request.protocol, inputs)
    candidates = [_candidate(entry, request.protocol, keys, inputs) for entry in request.candidates]
    if any(c["definition"]["data_identity"] != protocol["data_identity"] for c in candidates):
        raise ValueError("All candidates must bind the protocol data identity")
    by_id = {c["id"]: c for c in candidates}
    if len(by_id) != len(candidates) or request.incumbent not in by_id:
        raise ValueError("Unique candidates and explicit incumbent required")
    if protocol.get("candidate_identities") != [entry["definition"] for entry in request.candidates]:
        raise ValueError("Exact ordered candidates must be predeclared in protocol")
    incumbent = by_id[request.incumbent]
    comparisons = [_compare(c, incumbent, protocol, keys) for c in candidates if c is not incumbent]
    # All comparison primary values use the SAME incumbent query set for ranking;
    # differing matched subsets cannot be ranked as if they were identical evidence.
    eligible = [r for r in comparisons if r["eligible"]]
    supports = [{k for k in keys if by_id[r["candidate_id"]]["rows"][k]["observed"] and incumbent["rows"][k]["observed"]} for r in eligible]
    if supports and any(s != supports[0] for s in supports[1:]):
        for r in eligible:
            r["eligible"] = False
            r["failures"].append("conflicting matched candidate support; retain incumbent")
        eligible = []
    selected = min(eligible, key=lambda r: (r["primary"], by_id[r["candidate_id"]]["seconds"],
                                          by_id[r["candidate_id"]]["members"], r["candidate_id"]))["candidate_id"] if eligible else request.incumbent
    diagnostic_rows = sum(len(_groups(keys, protocol["event_strata"]))
                          * (1 + len(THRESHOLDS) + c["members"] * (c["members"] - 1) // 2)
                          for c in candidates)
    if diagnostic_rows > 250000:
        raise ValueError("Diagnostic row ceiling exceeded; stop within 2 GiB artifact budget")
    diagnostics = {c["id"]: _diagnostics(c, keys, protocol) for c in candidates}
    if time.monotonic() - started > 1200:
        raise TimeoutError("20 minute comparison budget exhausted; no selection persisted")
    _unchanged(inputs)
    with operation_run("geometry_selection", request, inputs=list(inputs)) as (directory, record):
        result = {"schema": SCHEMA, "decision_rule": RULE, "protocol_identity": request.protocol,
                  "candidate_identities": [c["identity"] for c in candidates],
                  "candidate_receipts": [c["receipt"] for c in candidates],
                  "selected_candidate": selected, "selected_definition": by_id[selected]["identity"],
                  "incumbent": request.incumbent, "comparisons": comparisons, "diagnostics": diagnostics,
                  "partition": "tuning", "episode_indices": list(EPISODES),
                  "consumed_holdout_used_for_selection": False, "holdout_eligible": False,
                  "holdout_ineligibility": "43/44 already consumed; no independently fixed fresh protocol",
                  "calibrated_uncertainty": False, "fits_performed": False,
                  "evidence_identities": [{"path": p.relative_to(ROOT).as_posix(), "sha256": h} for p, h in inputs.items()]}
        write_json(directory / "comparison.json", result)
        record.update(selected_candidate=selected, holdout_eligible=False, model_execution=False,
                      geometry_evidence_receipts=True)
        _unchanged(inputs)
    return directory / "comparison.json"


def freeze_geometry(request: GeometryFreezeRequest):
    authorize("geometry_freeze")
    inputs = {}
    result = _load(request.comparison, inputs)
    if (result.get("schema") != SCHEMA or result.get("decision_rule") != RULE
            or result.get("partition") != "tuning" or result.get("episode_indices") != list(EPISODES)
            or result.get("consumed_holdout_used_for_selection") is not False
            or result.get("holdout_eligible") is not False):
        raise ValueError("Only ineligible-for-holdout tuning comparison can freeze here")
    owner = _load({"path": str(project_path(ROOT, request.comparison["path"]).parent / "run.json"),
                   "sha256": sha256_file(project_path(ROOT, request.comparison["path"]).parent / "run.json")}, inputs)
    if (owner.get("status") != "completed" or owner.get("kind") != "learning_experiments.geometry_selection"
            or owner.get("artifacts", {}).get("comparison.json") != request.comparison["sha256"]):
        raise ValueError("Comparison must be owned by a completed geometry_selection run")
    definition = _load(result["selected_definition"], inputs)
    if definition.get("candidate_id") != result.get("selected_candidate"):
        raise ValueError("Selected candidate differs from exact definition")
    for identity in result["evidence_identities"]:
        _load(identity, inputs, parse=False)
    _unchanged(inputs)
    with operation_run("geometry_freeze", request, inputs=list(inputs)) as (directory, record):
        frozen = {"schema": SCHEMA, "decision_rule": RULE, "comparison_identity": request.comparison,
                  "selected_candidate": result["selected_candidate"], "candidate_definition": definition,
                  "definition_identity": result["selected_definition"], "protocol_identity": result["protocol_identity"],
                  "evidence_identities": result["evidence_identities"], "holdout_eligible": False,
                  "holdout_ineligibility": result["holdout_ineligibility"],
                  "historical_campaign_winner_changed": False, "production_promotion": False,
                  "calibrated_uncertainty": False, "consumed_holdout_used_for_selection": False}
        write_json(directory / "frozen-geometry-selection.json", frozen)
        record.update(selected_candidate=result["selected_candidate"], holdout_eligible=False,
                      historical_campaign_winner_changed=False)
        _unchanged(inputs)
    return directory / "frozen-geometry-selection.json"
