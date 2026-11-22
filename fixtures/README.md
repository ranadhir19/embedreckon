# Fixtures — planted hazards with known-correct verdicts

All fixtures are generated deterministically by
[`build_fixtures.py`](build_fixtures.py) (run it to regenerate;
the committed files are its output). No real embedding model is
involved: documents live in a shared latent topic space (16
topics), and the pseudo-models are seeded random projections of
that space at the **same** output dimension (64) — model A
(seed 1001), model B (seed 2002), model C (seed 3003, unrelated
latent corpus). Within a space neighbours are meaningful; across
spaces they are not; dimensions match, so shape checks pass.

## audit_index.jsonl + audit_manifest.json

The manifest claims model `pseudo-a` v1.0 / dim 64 / cosine for
five partitions. Expected stamps:

| Partition | Planted hazard | Expected stamp | Decisive reason |
|---|---|---|---|
| `docs-a` (600) | none — clean, as claimed | CONSISTENT | — |
| `docs-b` (600) | records stamped v0.9, manifest claims v1.0 | MIXED | MANIFEST_MISMATCH |
| `docs-foreign` (400) | written entirely by model B; every record claims model A | MIXED | CROSS_PARTITION_SIGNATURE_MISMATCH |
| `docs-mixed` (800) | 5% model-B tranche inside a model-A partition | MIXED | WITHIN_PARTITION_MIXTURE_SIGNAL |
| `docs-corrupt` (400) | 30% of vectors truncated to 32 dims, then zero-repadded to 64 | MIXED | TRUNCATED_REPADDED_SIGNATURE |
| `docs-unstamped` (300) | no manifest claim, no record provenance | UNVERIFIABLE | UNSTAMPED |

Teaching detail: `docs-foreign`'s own population is a perfectly
consistent single space — the wrong one. Its within-partition
screen says SINGLE-SPACE-CONSISTENT; only the cross-partition
signature leg catches the lie. And `docs-corrupt`'s screen is
likewise clean: repadded vectors keep their geometry on the
retained dimensions, so the corruption is a bookkeeping catch
(zero-tail signature), not a mixture signal.

## screen_*.jsonl — one population each, no provenance

| File | Planted content | Expected verdict |
|---|---|---|
| `screen_clean.jsonl` | 800 model-A vectors | SINGLE-SPACE-CONSISTENT (frac_neg 0.000) |
| `screen_mixed5.jsonl` | 800 vectors, 5% model B | SUSPECT-MIXED (frac_neg 0.017) |
| `screen_mixed1.jsonl` | 800 vectors, 1% model B | UNVERIFIABLE — BELOW_SEPARATION_FLOOR (frac_neg 0.002) |

The 1% file is the refusal boundary's teaching case: the signal is
real (0.002 vs a clean 0.000) but inside the registered band, so
the screen refuses rather than guess MIXED or stamp a false
CONSISTENT.

## probe_pairs_*.jsonl — 240 paired texts each

| File | Pair | Expected verdict | Measured |
|---|---|---|---|
| `probe_pairs_ab.jsonl` | pseudo-a vs pseudo-b (same latent space, different projections) | ALIGNED-WITH-MEASURED-LOSS | raw cosine 0.054 (chance); mapped preservation 0.787, hold-out 0.868, Procrustes residual 0.296 |
| `probe_pairs_unrelated.jsonl` | pseudo-a vs pseudo-c (unrelated latent corpus) | INCOMPATIBLE | mapped preservation 0.114, residual 1.189 |

## Migration fixtures — mig_old.jsonl is shared

500 live ids (`doc-0000`–`doc-0499`) embedded by model A, plus 12
tombstone records (`gone-00`–`gone-11`). `mig_manifest_b.json`
claims pseudo-b v1.0. `mig_golden.jsonl` holds 40 golden queries
embedded by model B with the old (model-A) index's top-10 as
reference. Certification threshold 0.35 (the CLI default).

| New index | Planted content | Expected verdict | Decisive leg |
|---|---|---|---|
| `mig_clean_new.jsonl` | all 500 ids re-embedded by model B | CERTIFIED | all legs pass; mean overlap 0.535 |
| `mig_broken_new.jsonl` | 10 live ids missing (2%); 6 tombstoned ids resurrected | REFUSED | completeness (missing 10, resurrected 6); equivalence still passes (0.538) and is reported |
| `mig_reel_new.jsonl` | model-A vectors stamped as pseudo-b — the "completed" migration that never re-embedded | REFUSED | retrieval equivalence: mean overlap 0.008 |
