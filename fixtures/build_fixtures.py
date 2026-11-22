"""Deterministic fixture builder for embedreckon.

Two pseudo-embedding models, no real model anywhere: a shared
latent topic space (16 topics; each document is a sparse topic
mixture over a common background component), with model A and
model B as different seeded random projections of that space at
the SAME output dimension (64). Within a space, neighbours are
meaningful; across spaces they are not; dimensions match, so
shape checks pass -- the ada-002 -> text-embedding-3-small trap,
reproduced honestly. Model C projects an unrelated latent corpus
(the probe's incompatible pair).

Running this file regenerates every committed fixture. The tests
and validate.py also import the population builders directly to
measure the screen across many seeds in memory.
"""

import json
import math
import os
import random

LATENT_DIM = 16
DIM = 64
TOPICS = 16

SEED_A = 1001
SEED_B = 2002
SEED_C = 3003


def make_latent(rng, n, offset=0.0):
    docs = []
    for _ in range(n):
        t1, t2 = rng.sample(range(TOPICS), 2)
        z = [0.0] * LATENT_DIM
        z[t1] = 0.75
        z[t2] = 0.45
        for i in range(LATENT_DIM):
            z[i] += 0.18 + offset + rng.gauss(0, 0.02)
        docs.append(z)
    return docs


def make_projection(seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1.0 / math.sqrt(LATENT_DIM))
             for _ in range(LATENT_DIM)] for _ in range(DIM)]


PA = make_projection(SEED_A)
PB = make_projection(SEED_B)
PC = make_projection(SEED_C)


def embed(projection, z):
    return [sum(row[k] * z[k] for k in range(LATENT_DIM)) for row in projection]


def population(seed, n, frac_b=0.0, quantize=False, trunc=0):
    """A population of vectors: (1-frac_b) model A, frac_b model B,
    shuffled. Optionally scalar-quantized or truncated-and-repadded."""
    rng = random.Random(seed)
    lat = make_latent(rng, n)
    vecs = []
    nb = int(round(n * frac_b))
    for i, z in enumerate(lat):
        v = embed(PB if i >= n - nb else PA, z)
        if trunc:
            v = v[:trunc] + [0.0] * (DIM - trunc)
        if quantize:
            step = 0.05
            v = [round(x / step) * step for x in v]
        vecs.append(v)
    rng.shuffle(vecs)
    return vecs


def _round(vec):
    return [round(x, 5) for x in vec]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _rec(rid, partition, vec, model=None, version=None):
    row = {"id": rid, "partition": partition, "vector": _round(vec)}
    if model is not None:
        row["provenance"] = {"model": model, "model_version": version,
                             "dimension": DIM, "metric": "cosine"}
    return row


def build_audit_index(out_dir):
    rng = random.Random(4242)
    lat = make_latent(rng, 3100)
    rows = []
    idx = 0

    def take(count):
        nonlocal idx
        part = lat[idx:idx + count]
        idx += count
        return part

    for i, z in enumerate(take(600)):  # docs-a: clean, as claimed
        rows.append(_rec(f"a-{i:04d}", "docs-a", embed(PA, z), "pseudo-a", "1.0"))
    for i, z in enumerate(take(600)):  # docs-b: records claim v0.9, manifest v1.0
        rows.append(_rec(f"b-{i:04d}", "docs-b", embed(PA, z), "pseudo-a", "0.9"))
    for i, z in enumerate(take(400)):  # docs-foreign: model B, records claim A
        rows.append(_rec(f"f-{i:04d}", "docs-foreign", embed(PB, z), "pseudo-a", "1.0"))
    mixed = take(800)  # docs-mixed: 5% model B tranche
    for i, z in enumerate(mixed):
        proj = PB if i >= 760 else PA
        rows.append(_rec(f"m-{i:04d}", "docs-mixed", embed(proj, z), "pseudo-a", "1.0"))
    for i, z in enumerate(take(400)):  # docs-corrupt: 30% truncated+repadded
        v = embed(PA, z)
        if i < 120:
            v = v[:32] + [0.0] * (DIM - 32)
        rows.append(_rec(f"c-{i:04d}", "docs-corrupt", v, "pseudo-a", "1.0"))
    for i, z in enumerate(take(300)):  # docs-unstamped: no provenance anywhere
        rows.append(_rec(f"u-{i:04d}", "docs-unstamped", embed(PA, z)))

    _write_jsonl(os.path.join(out_dir, "audit_index.jsonl"), rows)
    claim = {"model": "pseudo-a", "model_version": "1.0",
             "dimension": DIM, "metric": "cosine"}
    manifest = {
        "format": "embedreckon-manifest/1",
        "partitions": {name: dict(claim) for name in
                       ("docs-a", "docs-b", "docs-foreign",
                        "docs-mixed", "docs-corrupt")},
    }
    with open(os.path.join(out_dir, "audit_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)


def build_screen_exports(out_dir):
    for name, seed, frac in (("screen_clean", 101, 0.0),
                             ("screen_mixed5", 202, 0.05),
                             ("screen_mixed1", 303, 0.01)):
        vecs = population(seed, 800, frac_b=frac)
        rows = [{"id": f"s-{i:04d}", "partition": "population",
                 "vector": _round(v)} for i, v in enumerate(vecs)]
        _write_jsonl(os.path.join(out_dir, f"{name}.jsonl"), rows)


def build_probe_pairs(out_dir):
    rng = random.Random(11)
    lat = make_latent(rng, 240)
    rows = []
    for i, z in enumerate(lat):
        row = {"text_id": f"t-{i:04d}", "a": _round(embed(PA, z)),
               "b": _round(embed(PB, z))}
        if i == 0:
            row.update({"model_a": "pseudo-a", "model_b": "pseudo-b",
                        "family_a": "pseudo-family-x",
                        "family_b": "pseudo-family-y"})
        rows.append(row)
    _write_jsonl(os.path.join(out_dir, "probe_pairs_ab.jsonl"), rows)

    lat2 = make_latent(random.Random(99), 240, offset=0.35)
    rows = []
    for i, (z1, z2) in enumerate(zip(lat, lat2)):
        row = {"text_id": f"t-{i:04d}", "a": _round(embed(PA, z1)),
               "b": _round(embed(PC, z2))}
        if i == 0:
            row.update({"model_a": "pseudo-a", "model_b": "pseudo-c",
                        "family_a": "pseudo-family-x",
                        "family_b": "pseudo-family-z"})
        rows.append(row)
    _write_jsonl(os.path.join(out_dir, "probe_pairs_unrelated.jsonl"), rows)


def _topk_ids(query_vec, ids, vecs, k=10):
    def cos(a, b):
        d = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return d / (na * nb) if na and nb else 0.0
    scored = sorted(range(len(ids)), key=lambda i: -cos(query_vec, vecs[i]))
    return [ids[i] for i in scored[:k]]


def build_migration(out_dir):
    rng = random.Random(2024)
    lat = make_latent(rng, 500)
    ids = [f"doc-{i:04d}" for i in range(500)]
    tombstones = [f"gone-{i:02d}" for i in range(12)]
    a_vecs = [embed(PA, z) for z in lat]
    b_vecs = [embed(PB, z) for z in lat]

    old_rows = [_rec(ids[i], "default", a_vecs[i], "pseudo-a", "1.0")
                for i in range(500)]
    old_rows += [{"id": t, "tombstone": True} for t in tombstones]
    _write_jsonl(os.path.join(out_dir, "mig_old.jsonl"), old_rows)

    clean_rows = [_rec(ids[i], "default", b_vecs[i], "pseudo-b", "1.0")
                  for i in range(500)]
    _write_jsonl(os.path.join(out_dir, "mig_clean_new.jsonl"), clean_rows)

    broken_rows = [_rec(ids[i], "default", b_vecs[i], "pseudo-b", "1.0")
                   for i in range(500) if i % 50 != 49]  # 10 ids missing (2%)
    for j, t in enumerate(tombstones[:6]):  # 6 deleted ids resurrected
        broken_rows.append(_rec(t, "default", b_vecs[j], "pseudo-b", "1.0"))
    _write_jsonl(os.path.join(out_dir, "mig_broken_new.jsonl"), broken_rows)

    reel_rows = [_rec(ids[i], "default", a_vecs[i], "pseudo-b", "1.0")
                 for i in range(500)]  # old-space vectors, new-model stamps
    _write_jsonl(os.path.join(out_dir, "mig_reel_new.jsonl"), reel_rows)

    manifest = {"format": "embedreckon-manifest/1",
                "index": {"model": "pseudo-b", "model_version": "1.0",
                          "dimension": DIM, "metric": "cosine"}}
    with open(os.path.join(out_dir, "mig_manifest_b.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    qlat = make_latent(random.Random(777), 40)
    golden = []
    for qi, qz in enumerate(qlat):
        ref = _topk_ids(embed(PA, qz), ids, a_vecs, k=10)
        golden.append({"query_id": f"q-{qi:02d}",
                       "vector": _round(embed(PB, qz)),
                       "reference_top_k": ref})
    _write_jsonl(os.path.join(out_dir, "mig_golden.jsonl"), golden)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    build_audit_index(out_dir)
    build_screen_exports(out_dir)
    build_probe_pairs(out_dir)
    build_migration(out_dir)
    print("fixtures written to", out_dir)


if __name__ == "__main__":
    main()
