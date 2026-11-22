"""The compatibility probe.

Consumes paired embeddings the user supplies -- the same texts
embedded by two models -- and measures whether the two spaces can
be related, and with what loss. The strongest verdict available is
ALIGNED-WITH-MEASURED-LOSS, with the numbers attached; the probe
never stamps a pair "compatible" unqualified, because every
alignment it can measure loses something and the loss is the
deliverable.

Measurements:

- raw same-text cosine vs a shuffled-pairing baseline (are the
  spaces already the same geometry?);
- unmapped neighbourhood preservation (do the two spaces induce
  the same top-k neighbour sets over the shared texts?);
- an orthogonal Procrustes fit of B onto A (stdlib Jacobi
  eigendecomposition): relative residual, cross-space neighbour
  preservation after mapping (does a mapped B vector retrieve the
  neighbours its A twin retrieves?), and a hold-out version of the
  same -- the rotation is fit on half the pairs and evaluated on
  the other half, so the fit cannot grade its own homework.

Verdicts: ALIGNED-WITH-MEASURED-LOSS when either the raw geometry
or the Procrustes map preserves neighbourhoods at/above the
registered floor; INCOMPATIBLE when raw statistics are at chance
and no orthogonal map recovers neighbourhood structure;
UNVERIFIABLE when the evidence is in between, the pair count is
too small, or the declared models are same-family versions (the
registered refusal boundary: family-level signatures do not
support version-level claims).
"""

import random

from . import REFUSAL_BOUNDARY, stats

MIN_PAIRS = 30
PRESERVATION_ALIGNED = 0.40
PRESERVATION_INCOMPATIBLE = 0.20
RESIDUAL_INCOMPATIBLE = 0.90
RAW_ALIGNED_COSINE = 0.90


def probe_pairs(pairs, meta=None):
    meta = meta or {}
    result = {
        "n_pairs": len(pairs),
        "model_a": meta.get("model_a"),
        "model_b": meta.get("model_b"),
        "verdict": None,
        "reason_codes": [],
        "measurements": {},
        "refusal_boundary": REFUSAL_BOUNDARY,
    }
    if len(pairs) < MIN_PAIRS:
        result["verdict"] = "UNVERIFIABLE"
        result["reason_codes"].append("TOO_FEW_PAIRS")
        return result
    a_vecs = [p["a"] for p in pairs]
    b_vecs = [p["b"] for p in pairs]
    if any(len(v) != len(a_vecs[0]) for v in a_vecs) or \
            any(len(v) != len(b_vecs[0]) for v in b_vecs):
        result["verdict"] = "UNVERIFIABLE"
        result["reason_codes"].append("INCONSISTENT_DIMENSIONS")
        return result
    if len(a_vecs[0]) != len(b_vecs[0]):
        # Vectors of different dimensions cannot be compared or
        # rotated into each other at all; a learned map would be a
        # translator, which this tool does not ship by design.
        result["verdict"] = "INCOMPATIBLE"
        result["reason_codes"].append("DIMENSION_MISMATCH")
        result["measurements"]["dimension_a"] = len(a_vecs[0])
        result["measurements"]["dimension_b"] = len(b_vecs[0])
        return result

    n = len(pairs)
    raw_cos = sum(stats.cosine(a_vecs[i], b_vecs[i]) for i in range(n)) / n
    order = list(range(n))
    random.Random(5).shuffle(order)
    shuffled_cos = sum(stats.cosine(a_vecs[i], b_vecs[order[i]])
                       for i in range(n)) / n
    preservation_raw = stats.neighbour_preservation(a_vecs, b_vecs)
    result["measurements"].update({
        "raw_same_text_cosine": round(raw_cos, 4),
        "shuffled_baseline_cosine": round(shuffled_cos, 4),
        "neighbour_preservation_unmapped": round(preservation_raw, 4),
    })

    if raw_cos >= RAW_ALIGNED_COSINE and preservation_raw >= PRESERVATION_ALIGNED:
        result["verdict"] = "ALIGNED-WITH-MEASURED-LOSS"
        result["reason_codes"].append("RAW_GEOMETRY_ALIGNED")
        result["measurements"]["measured_loss"] = {
            "neighbour_loss_unmapped": round(1.0 - preservation_raw, 4),
            "mean_cosine_gap": round(1.0 - raw_cos, 4),
        }
        return result

    rotation, residual, _sing = stats.procrustes(b_vecs, a_vecs)
    b_mapped = [stats.apply_rotation(v, rotation) for v in b_vecs]
    preservation_mapped = stats.cross_preservation(a_vecs, b_mapped)
    # Hold-out: fit on the even pairs, evaluate on the odd ones.
    even = list(range(0, n, 2))
    odd = list(range(1, n, 2))
    rot_h, _res_h, _s = stats.procrustes([b_vecs[i] for i in even],
                                         [a_vecs[i] for i in even])
    a_odd = [a_vecs[i] for i in odd]
    b_odd_mapped = [stats.apply_rotation(b_vecs[i], rot_h) for i in odd]
    preservation_holdout = stats.cross_preservation(a_odd, b_odd_mapped)
    result["measurements"].update({
        "procrustes_residual": round(residual, 4),
        "neighbour_preservation_mapped": round(preservation_mapped, 4),
        "neighbour_preservation_holdout": round(preservation_holdout, 4),
    })

    best = max(preservation_raw, preservation_mapped)
    if best >= PRESERVATION_ALIGNED:
        result["verdict"] = "ALIGNED-WITH-MEASURED-LOSS"
        result["reason_codes"].append("ALIGNMENT_PRESERVES_NEIGHBOURHOODS")
        result["measurements"]["measured_loss"] = {
            "neighbour_loss_best_route": round(1.0 - best, 4),
            "neighbour_loss_holdout": round(1.0 - preservation_holdout, 4),
            "procrustes_residual": round(residual, 4),
        }
    elif (preservation_mapped <= PRESERVATION_INCOMPATIBLE
          and preservation_raw <= PRESERVATION_INCOMPATIBLE
          and residual >= RESIDUAL_INCOMPATIBLE):
        result["verdict"] = "INCOMPATIBLE"
        result["reason_codes"].append("NO_ALIGNMENT_RECOVERS_STRUCTURE")
    else:
        result["verdict"] = "UNVERIFIABLE"
        result["reason_codes"].append("ALIGNMENT_INCONCLUSIVE")
        fam_a, fam_b = meta.get("family_a"), meta.get("family_b")
        if fam_a and fam_b and fam_a == fam_b:
            result["reason_codes"].append("SAME_FAMILY_REFUSAL_BOUNDARY")
    return result
