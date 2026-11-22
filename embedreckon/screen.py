"""The no-provenance statistical screen.

Stamps a population SINGLE-SPACE-CONSISTENT, SUSPECT-MIXED or
UNVERIFIABLE. Population level only: the screen never names
individual vectors and never attributes a population to a model.
A null result is consistency, not proof -- contamination below the
registered separation floor (see stats.py) stamps UNVERIFIABLE,
and populations whose signatures the statistic cannot read
(quantized, uniformly truncated, too small) are refused rather
than stamped.
"""

from . import REFUSAL_BOUNDARY, stats

SCREEN_VERDICTS = ("SINGLE-SPACE-CONSISTENT", "SUSPECT-MIXED", "UNVERIFIABLE")


def screen_population(vecs, pairs=stats.SCREEN_PAIRS, seed=7):
    """Screen one population (a list of vectors of one dimension)."""
    n = len(vecs)
    result = {
        "n_vectors": n,
        "dimension": len(vecs[0]) if vecs else 0,
        "verdict": None,
        "reason_codes": [],
        "statistics": {},
        "refusal_boundary": REFUSAL_BOUNDARY,
    }
    if n < stats.MIN_POPULATION:
        result["verdict"] = "UNVERIFIABLE"
        result["reason_codes"].append("POPULATION_TOO_SMALL")
        return result
    dim = result["dimension"]
    if any(len(v) != dim for v in vecs):
        result["verdict"] = "UNVERIFIABLE"
        result["reason_codes"].append("INCONSISTENT_DIMENSIONS")
        return result

    tail_frac, modal_tail, tail_share = stats.zero_tail_profile(vecs, dim)
    result["statistics"]["zero_tail_fraction"] = round(tail_frac, 4)
    if tail_share >= 0.90 and modal_tail >= max(1, dim // 4):
        # A uniformly truncated population is a legitimate
        # reduced-dimension space; the full-dimension null does not
        # apply, so the screen refuses rather than stamps.
        result["verdict"] = "UNVERIFIABLE"
        result["reason_codes"].append("TRUNCATED_POPULATION")
        return result
    if stats.quantization_signature(vecs):
        # Quantization moves exactly the statistics the screen
        # reads; sensitivity cannot be stated honestly.
        result["verdict"] = "UNVERIFIABLE"
        result["reason_codes"].append("QUANTIZED_POPULATION")
        return result

    frac, used = stats.frac_neg_within(vecs, seed=seed, pairs=pairs)
    result["statistics"]["frac_neg"] = round(frac, 5)
    result["statistics"]["pairs_sampled"] = used
    result["statistics"]["tau_neg"] = stats.TAU_NEG
    result["statistics"]["mean_pairwise_cosine"] = round(
        stats.mean_pairwise_cosine(vecs, seed=seed), 4)
    result["statistics"]["clean_ceiling"] = stats.CLEAN_CEILING
    result["statistics"]["mixture_floor"] = stats.MIXTURE_FLOOR

    if frac >= stats.MIXTURE_FLOOR:
        result["verdict"] = "SUSPECT-MIXED"
        result["reason_codes"].append("NEGATIVE_TAIL_EXCESS")
    elif frac <= stats.CLEAN_CEILING:
        result["verdict"] = "SINGLE-SPACE-CONSISTENT"
        result["reason_codes"].append("NEGATIVE_TAIL_AT_CLEAN_LEVEL")
    else:
        result["verdict"] = "UNVERIFIABLE"
        result["reason_codes"].append("BELOW_SEPARATION_FLOOR")
    return result


def screen_export(records, pairs=stats.SCREEN_PAIRS):
    """Screen every partition of an export separately, plus the whole."""
    from .model import partitions_of

    grouped = partitions_of(records)
    report = {"partitions": {}, "refusal_boundary": REFUSAL_BOUNDARY}
    for name, recs in grouped.items():
        report["partitions"][name] = screen_population(
            [r.vector for r in recs], pairs=pairs)
    return report
