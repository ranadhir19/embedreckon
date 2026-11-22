"""Input formats and loaders.

Vector export (JSONL, one record per line)::

    {"id": "doc-0001", "partition": "docs-a", "vector": [0.01, ...],
     "provenance": {"model": "pseudo-a", "model_version": "1.0",
                    "dimension": 64, "metric": "cosine"}}

``partition`` defaults to ``"default"``; ``provenance`` is optional.
A tombstone record marks a deleted id (used by migration
certification)::

    {"id": "doc-0999", "tombstone": true}

Provenance manifest (JSON)::

    {"format": "embedreckon-manifest/1",
     "index": {"model": ..., "model_version": ..., "dimension": ...,
               "metric": ...},
     "partitions": {"<name>": { ... per-partition claim overrides ... }}}

A partition's claim is the index claim with any per-partition fields
overriding it. A partition absent from the manifest still inherits
the index claim; a manifest with no claim at all for a partition
leaves it UNSTAMPED.

Paired embeddings for the compatibility probe (JSONL)::

    {"text_id": "t-0001", "a": [...], "b": [...],
     "model_a": "...", "model_b": "...", "family_a": "...",
     "family_b": "..."}

(model/family fields optional, taken from the first record that has
them.)

Golden set for certification (JSONL)::

    {"query_id": "q-00", "vector": [...], "reference_top_k": ["id", ...]}

The query vector is embedded by the NEW model; ``reference_top_k``
is the old index's top-k for the corresponding old-space query.

Loaders never silently drop input: malformed lines are collected as
parse issues with line numbers and surfaced in every report.
"""

import json


class Record:
    __slots__ = ("id", "partition", "vector", "provenance", "tombstone")

    def __init__(self, rid, partition, vector, provenance, tombstone=False):
        self.id = rid
        self.partition = partition
        self.vector = vector
        self.provenance = provenance
        self.tombstone = tombstone


def load_export(path):
    """Return (records, issues). Issues are dicts with line + error."""
    records = []
    issues = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append({"line": lineno, "error": f"invalid JSON: {exc}"})
                continue
            if not isinstance(obj, dict) or "id" not in obj:
                issues.append({"line": lineno, "error": "record has no 'id'"})
                continue
            if obj.get("tombstone"):
                records.append(Record(obj["id"], obj.get("partition", "default"),
                                      None, None, tombstone=True))
                continue
            vec = obj.get("vector")
            if (not isinstance(vec, list) or not vec
                    or not all(isinstance(x, (int, float)) for x in vec)):
                issues.append({"line": lineno,
                               "error": f"record {obj['id']!r}: missing or "
                                        "non-numeric 'vector'"})
                continue
            records.append(Record(obj["id"], obj.get("partition", "default"),
                                  [float(x) for x in vec],
                                  obj.get("provenance")))
    return records, issues


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    return manifest


def claim_for(manifest, partition):
    """Resolve the provenance claim for a partition, or None."""
    if manifest is None:
        return None
    claim = dict(manifest.get("index") or {})
    per = (manifest.get("partitions") or {}).get(partition)
    if per:
        claim.update(per)
    if not claim.get("model"):
        return None
    return claim


def load_pairs(path):
    """Return (pairs, issues); pairs are dicts with text_id/a/b + meta."""
    pairs = []
    issues = []
    meta = {}
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append({"line": lineno, "error": f"invalid JSON: {exc}"})
                continue
            a, b = obj.get("a"), obj.get("b")
            if (not isinstance(a, list) or not isinstance(b, list)
                    or not a or not b):
                issues.append({"line": lineno,
                               "error": "pair record needs 'a' and 'b' vectors"})
                continue
            pairs.append({"text_id": obj.get("text_id", f"line-{lineno}"),
                          "a": [float(x) for x in a],
                          "b": [float(x) for x in b]})
            for key in ("model_a", "model_b", "family_a", "family_b"):
                if key in obj and key not in meta:
                    meta[key] = obj[key]
    return pairs, issues, meta


def load_golden(path):
    queries = []
    issues = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append({"line": lineno, "error": f"invalid JSON: {exc}"})
                continue
            vec = obj.get("vector")
            ref = obj.get("reference_top_k")
            if not isinstance(vec, list) or not vec or not isinstance(ref, list):
                issues.append({"line": lineno,
                               "error": "golden record needs 'vector' and "
                                        "'reference_top_k'"})
                continue
            queries.append({"query_id": obj.get("query_id", f"line-{lineno}"),
                            "vector": [float(x) for x in vec],
                            "reference_top_k": [str(x) for x in ref]})
    return queries, issues


def partitions_of(records):
    """Group live (non-tombstone) records by partition, in file order."""
    grouped = {}
    for rec in records:
        if rec.tombstone:
            continue
        grouped.setdefault(rec.partition, []).append(rec)
    return grouped
