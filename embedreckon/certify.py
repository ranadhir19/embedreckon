"""Migration certification.

Given an old index export, a new index export, the new index's
provenance manifest and a golden query set, certify or refuse the
migration before cutover on three independent legs:

1. Completeness -- the new index must contain exactly the live
   document set: every live id from the old export present
   (missing ids refuse), no tombstoned (deleted) id present
   (resurrected ids refuse). Ids in the new index that the old
   never held are reported as additions, not failures.
2. Single-space -- the new index must pass the provenance audit:
   every partition CONSISTENT. A MIXED partition refuses; an
   UNVERIFIABLE partition also refuses -- a certificate is not
   issued on partial evidence. A partition the manifest does not
   claim at all also refuses this leg: certification binds to the
   manifest's declaration, not to self-declared record provenance.
3. Retrieval equivalence -- for each golden query (embedded by
   the new model), the new index's top-k is compared with the
   reference top-k recorded from the old index. The mean overlap
   must meet the declared threshold. This is the leg that catches
   the canonical failure -- a "completed" migration whose vectors
   are still in the old space -- when every provenance record
   claims otherwise.

The certificate names every check, threshold and measured value.
"""

from . import REFUSAL_BOUNDARY, stats
from .audit import audit_export
from .model import claim_for

DEFAULT_OVERLAP_THRESHOLD = 0.35


def _topk_ids(query_vec, records, k):
    scored = []
    nq = stats.norm(query_vec)
    for rec in records:
        nv = stats.norm(rec.vector)
        if nq == 0.0 or nv == 0.0:
            continue
        scored.append((sum(x * y for x, y in zip(query_vec, rec.vector))
                       / (nq * nv), rec.id))
    scored.sort(reverse=True)
    return [rid for _s, rid in scored[:k]]


def certify(old_records, new_records, manifest, golden,
            threshold=DEFAULT_OVERLAP_THRESHOLD, pairs=stats.SCREEN_PAIRS):
    legs = {}

    # Leg 1: completeness.
    live_old = {r.id for r in old_records if not r.tombstone}
    tombstoned = {r.id for r in old_records if r.tombstone}
    new_ids = {r.id for r in new_records if not r.tombstone}
    missing = sorted(live_old - new_ids)
    resurrected = sorted(new_ids & tombstoned)
    added = sorted(new_ids - live_old - tombstoned)
    legs["completeness"] = {
        "status": "PASS" if not missing and not resurrected else "FAIL",
        "live_ids_old": len(live_old),
        "ids_new": len(new_ids),
        "missing_ids": missing,
        "missing_count": len(missing),
        "resurrected_ids": resurrected,
        "resurrected_count": len(resurrected),
        "added_ids_count": len(added),
    }

    # Leg 2: single-space (provenance audit of the new index).
    # Certification binds to the *manifest's* declaration: a
    # partition the manifest does not claim cannot be certified on
    # the strength of self-declared record provenance alone, so it
    # makes this leg UNVERIFIABLE even if the audit itself stamps
    # the partition CONSISTENT against its own records.
    audit = audit_export(new_records, manifest, pairs=pairs)
    stamps = {name: p["stamp"] for name, p in audit["partitions"].items()}
    unclaimed = [name for name in audit["partitions"]
                 if claim_for(manifest, name) is None]
    if unclaimed:
        ss_status = "UNVERIFIABLE"
    elif all(s == "CONSISTENT" for s in stamps.values()) and stamps:
        ss_status = "PASS"
    elif any(s == "MIXED" for s in stamps.values()):
        ss_status = "FAIL"
    else:
        ss_status = "UNVERIFIABLE"
    legs["single_space"] = {
        "status": ss_status,
        "partition_stamps": stamps,
        "partitions_without_manifest_claim": unclaimed,
        "audit": audit,
    }

    # Leg 3: retrieval equivalence on the golden set.
    live_new = [r for r in new_records if not r.tombstone]
    overlaps = []
    per_query = []
    for query in golden:
        k = len(query["reference_top_k"])
        got = _topk_ids(query["vector"], live_new, k)
        overlap = (len(set(got) & set(query["reference_top_k"])) / k) if k else 0.0
        overlaps.append(overlap)
        per_query.append({"query_id": query["query_id"],
                          "overlap": round(overlap, 4)})
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    if not golden:
        eq_status = "UNVERIFIABLE"
    else:
        eq_status = "PASS" if mean_overlap >= threshold else "FAIL"
    legs["retrieval_equivalence"] = {
        "status": eq_status,
        "threshold": threshold,
        "mean_topk_overlap": round(mean_overlap, 4),
        "min_topk_overlap": round(min(overlaps), 4) if overlaps else None,
        "n_queries": len(golden),
        "per_query": per_query,
    }

    statuses = [legs[key]["status"] for key in
                ("completeness", "single_space", "retrieval_equivalence")]
    certified = all(s == "PASS" for s in statuses)
    refusal_reasons = []
    if legs["completeness"]["status"] != "PASS":
        refusal_reasons.append("COMPLETENESS_FAILED")
    if legs["single_space"]["status"] == "FAIL":
        refusal_reasons.append("SINGLE_SPACE_FAILED")
    elif legs["single_space"]["status"] != "PASS":
        refusal_reasons.append("SINGLE_SPACE_UNVERIFIABLE")
    if legs["retrieval_equivalence"]["status"] == "FAIL":
        refusal_reasons.append("RETRIEVAL_REGRESSION")
    elif legs["retrieval_equivalence"]["status"] != "PASS":
        refusal_reasons.append("EQUIVALENCE_UNVERIFIABLE")

    return {
        "verdict": "CERTIFIED" if certified else "REFUSED",
        "refusal_reasons": refusal_reasons,
        "legs": legs,
        "refusal_boundary": REFUSAL_BOUNDARY,
    }
