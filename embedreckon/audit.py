"""The provenance audit.

Stamps every partition of an export CONSISTENT, MIXED or
UNVERIFIABLE against a declared provenance manifest. The legs:

1. Claim resolution -- a partition with no manifest claim and no
   per-record provenance is UNSTAMPED (UNVERIFIABLE). Per-record
   provenance that disagrees with the claim (model or version) is
   positive evidence of mixture: MANIFEST_MISMATCH.
2. Dimension -- any vector whose length differs from the claimed
   dimension: DIMENSION_MISMATCH.
3. Duplicate ids anywhere in the export: DUPLICATE_ID in every
   partition holding one.
4. Truncation signature -- a substantial tranche of vectors whose
   trailing dimensions are exactly zero (truncated, then repadded
   to restore the declared dimension): TRUNCATED_REPADDED_SIGNATURE.
5. Signature agreement -- partitions claiming the same model are
   compared pairwise (cross-partition negative-tail statistic).
   The largest set of mutually agreeing partitions defines the
   claimed space's geometry; a partition outside that set whose
   cross statistic against the set's pooled vectors exceeds the
   mixture floor is MIXED (CROSS_PARTITION_SIGNATURE_MISMATCH) --
   this is the leg that catches a partition silently written by a
   different model while every record claims the declared one.
   A partition that agrees with no majority and does not deviate
   from it either is UNVERIFIABLE (SIGNATURE_INCONCLUSIVE): with
   two disagreeing partitions and no majority, the audit will not
   guess which one is foreign.

CONSISTENT means "consistent with the declared provenance on every
leg that could be run" -- the report lists which legs ran. It is
never a proof of single-space; that claim is not available from
any leg here.
"""

from collections import Counter

from . import REFUSAL_BOUNDARY, stats
from .model import claim_for, partitions_of
from .screen import screen_population

ZERO_TAIL_TRANCHE = 0.10  # >10% repadded vectors = corruption tranche

MIXED_CODES = {
    "MANIFEST_MISMATCH", "DIMENSION_MISMATCH", "DUPLICATE_ID",
    "TRUNCATED_REPADDED_SIGNATURE", "WITHIN_PARTITION_MIXTURE_SIGNAL",
    "CROSS_PARTITION_SIGNATURE_MISMATCH",
}


def _largest_agreement_set(names, agree):
    """Largest subset of names pairwise agreeing (brute force;
    partition counts in an index are small)."""
    best = []
    n = len(names)
    for mask in range(1, 1 << n):
        subset = [names[i] for i in range(n) if mask & (1 << i)]
        if len(subset) <= len(best):
            continue
        ok = True
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if not agree.get((subset[i], subset[j]), False):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            best = subset
    return best


def _signature_legs(grouped, claims, pairs):
    """Cross-partition signature analysis, grouped by claimed model.

    Returns {partition: (reason_codes, leg_notes)}.
    """
    out = {name: ([], {}) for name in grouped}
    by_model = {}
    for name, recs in grouped.items():
        claim = claims[name]
        if claim is not None and len(recs) >= stats.MIN_PARTITION_STAT:
            by_model.setdefault(claim.get("model"), []).append(name)

    for model, names in by_model.items():
        if len(names) < 2:
            for name in names:
                out[name][1]["cross_partition"] = (
                    "not run: only partition claiming this model")
            continue
        vecs = {name: [r.vector for r in grouped[name]] for name in names}
        cross = {}
        agree = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                frac, _used = stats.frac_neg_cross(vecs[a], vecs[b], pairs=pairs)
                cross[(a, b)] = cross[(b, a)] = frac
                agree[(a, b)] = agree[(b, a)] = frac <= stats.CLEAN_CEILING
        cluster = _largest_agreement_set(names, agree)
        for name in names:
            reasons, notes = out[name]
            others = {o: round(cross[(name, o)], 5) for o in names if o != name}
            notes["cross_partition_frac_neg"] = others
            if len(cluster) < 2:
                if any(f > stats.CLEAN_CEILING for f in others.values()):
                    reasons.append("SIGNATURE_INCONCLUSIVE")
                    notes["cross_partition"] = (
                        "partitions disagree and no majority agreement "
                        "set exists; the audit will not guess which is foreign")
                else:
                    notes["cross_partition"] = "all pairwise statistics at clean level"
                continue
            if name in cluster:
                notes["cross_partition"] = (
                    f"member of the majority agreement set {sorted(cluster)}")
                continue
            pool = [v for member in cluster for v in vecs[member]]
            frac, _used = stats.frac_neg_cross(vecs[name], pool, pairs=pairs)
            notes["cross_partition_frac_neg_vs_agreement_set"] = round(frac, 5)
            if frac >= stats.MIXTURE_FLOOR:
                reasons.append("CROSS_PARTITION_SIGNATURE_MISMATCH")
                notes["cross_partition"] = (
                    "deviates from the majority agreement set's geometry")
            elif frac > stats.CLEAN_CEILING:
                reasons.append("SIGNATURE_INCONCLUSIVE")
                notes["cross_partition"] = (
                    "outside the agreement set but deviation is inside "
                    "the refusal band")
            else:
                notes["cross_partition"] = (
                    "outside the agreement set; pooled deviation at clean level")
    return out


def audit_export(records, manifest, pairs=stats.SCREEN_PAIRS):
    grouped = partitions_of(records)
    claims = {name: claim_for(manifest, name) for name in grouped}

    id_counts = Counter(r.id for r in records if not r.tombstone)
    duplicate_ids = {rid for rid, c in id_counts.items() if c > 1}

    sig = _signature_legs(grouped, claims, pairs)

    partitions = {}
    for name, recs in grouped.items():
        claim = claims[name]
        vecs = [r.vector for r in recs]
        reasons = []
        legs = {}

        # Leg 1: claim resolution + record provenance agreement.
        stamped = [r for r in recs if r.provenance and r.provenance.get("model")]
        if claim is None and not stamped:
            reasons.append("UNSTAMPED")
            legs["claim"] = "no manifest claim, no record provenance"
        else:
            legs["claim"] = "resolved"
            if claim is not None:
                mismatched = 0
                for r in stamped:
                    prov = r.provenance
                    if prov.get("model") != claim.get("model"):
                        mismatched += 1
                    elif (claim.get("model_version") is not None
                          and prov.get("model_version") is not None
                          and prov.get("model_version") != claim.get("model_version")):
                        mismatched += 1
                if mismatched:
                    reasons.append("MANIFEST_MISMATCH")
                    legs["record_provenance"] = (
                        f"{mismatched}/{len(stamped)} stamped records "
                        "disagree with the claim")
                else:
                    legs["record_provenance"] = (
                        f"{len(stamped)} stamped records agree")

        # Leg 2: dimension.
        if claim is not None and claim.get("dimension") is not None:
            want = int(claim["dimension"])
            bad = sum(1 for v in vecs if len(v) != want)
            if bad:
                reasons.append("DIMENSION_MISMATCH")
                legs["dimension"] = f"{bad} vectors not of dimension {want}"
            else:
                legs["dimension"] = f"all vectors dimension {want}"
        else:
            dims = {len(v) for v in vecs}
            if len(dims) > 1:
                reasons.append("DIMENSION_MISMATCH")
                legs["dimension"] = f"mixed dimensions {sorted(dims)}, no claim"
            else:
                legs["dimension"] = "single dimension, unclaimed"

        # Leg 3: duplicate ids.
        dups_here = [r.id for r in recs if r.id in duplicate_ids]
        if dups_here:
            reasons.append("DUPLICATE_ID")
            legs["duplicates"] = f"{len(dups_here)} records with duplicated ids"
        else:
            legs["duplicates"] = "none"

        # Leg 4: truncation/repadding signature.
        dim0 = len(vecs[0]) if vecs else 0
        tail_frac, modal_tail, _share = stats.zero_tail_profile(vecs, dim0)
        if tail_frac > ZERO_TAIL_TRANCHE:
            reasons.append("TRUNCATED_REPADDED_SIGNATURE")
            legs["truncation"] = (
                f"{tail_frac:.0%} of vectors repadded from {dim0 - modal_tail} "
                "dimensions (trailing zeros)")
        else:
            legs["truncation"] = f"zero-tail fraction {tail_frac:.3f}"

        # Leg 5: within-population screen + cross-partition signature.
        screen_res = screen_population(vecs, pairs=pairs)
        legs["within_screen"] = screen_res["verdict"]
        if screen_res["verdict"] == "SUSPECT-MIXED":
            reasons.append("WITHIN_PARTITION_MIXTURE_SIGNAL")
        sig_reasons, sig_notes = sig[name]
        reasons.extend(sig_reasons)
        legs.update(sig_notes)

        if any(r in MIXED_CODES for r in reasons):
            stamp = "MIXED"
        elif reasons:
            stamp = "UNVERIFIABLE"
        else:
            stamp = "CONSISTENT"

        partitions[name] = {
            "stamp": stamp,
            "reason_codes": reasons,
            "n_vectors": len(vecs),
            "claim": claim,
            "legs": legs,
            "screen": screen_res,
        }

    summary = Counter(p["stamp"] for p in partitions.values())
    return {
        "partitions": partitions,
        "summary": dict(summary),
        "duplicate_ids_total": len(duplicate_ids),
        "refusal_boundary": REFUSAL_BOUNDARY,
    }
