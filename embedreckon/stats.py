"""Vector statistics: the negative-tail mixture statistic, structural
signatures, and the orthogonal-Procrustes alignment machinery.

The screen statistic
--------------------
In a single embedding space whose population shares any background
component (real corpora always induce one: common topics, common
vocabulary, a shared mean direction), strongly negative pairwise
cosines are rare. Vectors from a *different* space are oriented
independently of the population, so cross-space pairs produce
strongly negative cosines at a substantial rate. The statistic is
the fraction of sampled pairs whose cosine falls below TAU_NEG.

Registered thresholds (measured on the fixture arms at build time;
see validate.py's measurement table, which re-derives them):

- clean populations measured 0.0000 across seeds, including
  scalar-quantized and uniformly truncated arms;
- a 5% same-dimension foreign tranche measured >= 0.0138;
- a 1% tranche measured 0.0023-0.0082 -- inside the refusal band,
  which is why the band exists.

So CLEAN_CEILING = 0.0015 and MIXTURE_FLOOR = 0.010, with everything
between stamped UNVERIFIABLE. These constants are the published
refusal boundary; they are not tuned per input.
"""

import math
import random

TAU_NEG = -0.05          # "strongly negative" cosine threshold
CLEAN_CEILING = 0.0015   # at/below: consistent with a single space
MIXTURE_FLOOR = 0.010    # at/above: mixture signal
SCREEN_PAIRS = 6000      # sampled pairs per statistic (default)
MIN_POPULATION = 300     # below this, the screen refuses
MIN_REFERENCE = 300      # cross-partition reference minimum
MIN_PARTITION_STAT = 100  # cross-partition target minimum


def norm(vec):
    return math.sqrt(sum(x * x for x in vec))


def cosine(a, b, na=None, nb=None):
    na = norm(a) if na is None else na
    nb = norm(b) if nb is None else nb
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _frac_neg_sampled(vecs_a, vecs_b, seed, pairs):
    """Fraction of sampled pairs with cosine < TAU_NEG.

    If vecs_b is None, pairs are drawn within vecs_a; otherwise each
    pair is one vector from vecs_a and one from vecs_b.
    """
    rng = random.Random(seed)
    na = [norm(v) for v in vecs_a]
    nb = na if vecs_b is None else [norm(v) for v in vecs_b]
    other = vecs_a if vecs_b is None else vecs_b
    neg = 0
    used = 0
    la, lb = len(vecs_a), len(other)
    for _ in range(pairs):
        i = rng.randrange(la)
        j = rng.randrange(lb)
        if vecs_b is None and i == j:
            continue
        va, vb = vecs_a[i], other[j]
        if na[i] == 0.0 or nb[j] == 0.0:
            continue
        d = sum(x * y for x, y in zip(va, vb)) / (na[i] * nb[j])
        used += 1
        if d < TAU_NEG:
            neg += 1
    return (neg / used if used else 0.0), used


def frac_neg_within(vecs, seed=7, pairs=SCREEN_PAIRS):
    return _frac_neg_sampled(vecs, None, seed, pairs)


def frac_neg_cross(vecs, reference, seed=13, pairs=SCREEN_PAIRS):
    return _frac_neg_sampled(vecs, reference, seed, pairs)


def mean_pairwise_cosine(vecs, seed=7, pairs=2000):
    rng = random.Random(seed)
    norms = [norm(v) for v in vecs]
    n = len(vecs)
    total = 0.0
    used = 0
    for _ in range(pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j or norms[i] == 0.0 or norms[j] == 0.0:
            continue
        total += sum(x * y for x, y in zip(vecs[i], vecs[j])) / (norms[i] * norms[j])
        used += 1
    return total / used if used else 0.0


def trailing_zero_dims(vec):
    """Number of trailing dimensions that are exactly 0.0."""
    count = 0
    for x in reversed(vec):
        if x == 0.0:
            count += 1
        else:
            break
    return count


def zero_tail_profile(vecs, dim):
    """(fraction of vectors with >= 25% trailing zero dims,
    modal trailing-zero length among those, fraction sharing it)."""
    threshold = max(1, dim // 4)
    tails = [trailing_zero_dims(v) for v in vecs]
    flagged = [t for t in tails if t >= threshold]
    frac = len(flagged) / len(vecs) if vecs else 0.0
    if not flagged:
        return frac, 0, 0.0
    modal = max(set(flagged), key=flagged.count)
    share = flagged.count(modal) / len(vecs)
    return frac, modal, share


def quantization_signature(vecs, sample=128):
    """True if the population's values sit on a coarse grid.

    Samples up to `sample` vectors and takes the median per-dimension
    distinct-value count; continuous embeddings give ~sample distinct
    values per dimension, scalar-quantized ones far fewer.
    """
    if not vecs:
        return False
    sample_vecs = vecs[:sample]
    dim = len(sample_vecs[0])
    distinct = []
    for d in range(dim):
        distinct.append(len({round(v[d], 9) for v in sample_vecs}))
    distinct.sort()
    median = distinct[len(distinct) // 2]
    return median <= max(16, len(sample_vecs) // 2)


# ---------------------------------------------------------------------
# Orthogonal Procrustes via Jacobi eigendecomposition (stdlib only).
# The rotation is the orthogonal polar factor of B^T A, computed as
# R = M (M^T M)^-1/2 with the inverse square root taken from the
# eigendecomposition of the symmetric matrix M^T M. This
# implementation was verified against known rotations (residual
# ~1e-15) before being adopted.
# ---------------------------------------------------------------------

def _jacobi_eigh(matrix, sweeps=100):
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n)
                            for j in range(n) if i != j))
        if off < 1e-13:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-15:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


def _cross_t(a_rows, b_rows):
    d = len(a_rows[0])
    return [[sum(a_rows[k][i] * b_rows[k][j] for k in range(len(a_rows)))
             for j in range(d)] for i in range(d)]


def procrustes(b_vecs, a_vecs):
    """Orthogonal R minimizing ||B R - A||_F.

    Returns (R, relative_residual, smallest_singular_value_ratio).
    The singular ratio (smallest/largest singular value of B^T A)
    reports how well-conditioned the alignment is: near zero, the
    fit is fitting noise.
    """
    m = _cross_t(b_vecs, a_vecs)
    mtm = _cross_t(m, m)
    evals, q = _jacobi_eigh(mtm)
    d = len(m)
    inv_sqrt = [[sum(q[i][k] * (1.0 / math.sqrt(max(evals[k], 1e-12))) * q[j][k]
                     for k in range(d)) for j in range(d)] for i in range(d)]
    r = [[sum(m[i][k] * inv_sqrt[k][j] for k in range(d)) for j in range(d)]
         for i in range(d)]
    num = 0.0
    den = 0.0
    for b, a in zip(b_vecs, a_vecs):
        br = [sum(b[k] * r[k][j] for k in range(d)) for j in range(d)]
        num += sum((x - y) ** 2 for x, y in zip(br, a))
        den += sum(y * y for y in a)
    residual = math.sqrt(num / den) if den else float("inf")
    sing = [math.sqrt(max(e, 0.0)) for e in evals]
    ratio = (min(sing) / max(sing)) if sing and max(sing) > 0 else 0.0
    return r, residual, ratio


def apply_rotation(vec, r):
    d = len(vec)
    return [sum(vec[k] * r[k][j] for k in range(d)) for j in range(d)]


def neighbour_preservation(reference_vecs, candidate_vecs, k=10):
    """Mean top-k neighbour-set overlap over shared item indices.

    Both lists are indexed by the same items (e.g. the same texts):
    for each item, its top-k neighbours under reference_vecs'
    geometry vs under candidate_vecs' geometry.
    """
    n = len(reference_vecs)
    if n < 2:
        return 0.0
    k = min(k, n - 1)

    def topk(vecs, i):
        vi = vecs[i]
        ni = norm(vi)
        scored = []
        for j in range(n):
            if j == i:
                continue
            nj = norm(vecs[j])
            if ni == 0.0 or nj == 0.0:
                continue
            scored.append((sum(x * y for x, y in zip(vi, vecs[j])) / (ni * nj), j))
        scored.sort(reverse=True)
        return {j for _, j in scored[:k]}

    total = 0.0
    for i in range(n):
        total += len(topk(reference_vecs, i) & topk(candidate_vecs, i)) / k
    return total / n


def cross_preservation(reference_vecs, mapped_vecs, k=10):
    """Retrieval agreement after mapping: for each item, the top-k
    neighbours of mapped_vecs[i] *among the reference vectors* vs
    the top-k neighbours of reference_vecs[i] among themselves.

    Unlike neighbour_preservation (which compares two internal
    geometries and is invariant under orthogonal maps), this is the
    cross-space question a migration actually asks: after aligning
    B into A's space, does a B vector retrieve the same neighbours
    its A twin retrieves?
    """
    n = len(reference_vecs)
    if n < 2:
        return 0.0
    k = min(k, n - 1)
    norms_r = [norm(v) for v in reference_vecs]

    def topk_among(vec, skip):
        nv = norm(vec)
        if nv == 0.0:
            return set()
        scored = []
        for j in range(n):
            if j == skip or norms_r[j] == 0.0:
                continue
            scored.append((sum(x * y for x, y in zip(vec, reference_vecs[j]))
                           / (nv * norms_r[j]), j))
        scored.sort(reverse=True)
        return {j for _, j in scored[:k]}

    total = 0.0
    for i in range(n):
        total += len(topk_among(reference_vecs[i], i)
                     & topk_among(mapped_vecs[i], i)) / k
    return total / n
