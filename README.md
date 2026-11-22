# embedreckon

**The embedding-space auditor and migration certifier.**

AI systems retrieve from vector indexes that other pipelines built
— and nothing checks the index itself. When a team ships a new
embedding model, live queries embed in the new model's space while
the stored documents still carry old-model vectors. The two spaces
are incompatible — including at identical dimensions, where every
shape check passes — so search returns wrong documents, silently,
with no error anywhere. The migration fixes that fix this
(dual-write, shadow index, backfill, cutover) are documented and
vendor-endorsed, but their verification step is the operator
comparing quality on their own queries. Nobody stamps the index.

embedreckon stamps the index. Given a vector export and a declared
provenance manifest, it audits every partition; given an
unprovenanced population, it screens it for mixed-space
contamination under a published refusal boundary; given paired
embeddings of the same texts from two models, it measures whether
their spaces can be aligned and with what loss; and given an old
index, a new index and a golden query set, it **certifies or
refuses a migration before cutover**.

## The refusal boundary (printed in every report)

> embedreckon stamps populations and partitions, never individual
> vectors; it never attributes a vector to a named model. A
> statistical null result is SINGLE-SPACE-CONSISTENT, never
> "verified single-space": contamination whose negative-tail
> signal falls in the registered refusal band (at fixture scale,
> roughly the 1–2% foreign-tranche range) stamps UNVERIFIABLE,
> not a guessed MIXED and not a false CONSISTENT. Statistics
> alone never certify a migration: certification requires the
> provenance audit, the completeness check and measured retrieval
> equivalence to all pass; any leg that cannot be evaluated
> refuses the certificate.

This boundary is a design constraint, not a disclaimer. Space
translation and per-vector model attribution are dual-use (the
research literature demonstrates attribute inference from
unattributed vector databases), so the tool ships no translator
and makes no per-vector claims. The compatibility probe measures
a pair the user declares, on pairs the user supplies.

## Commands

```bash
# 1. Provenance audit: stamp every partition against a manifest
embedreckon audit --export index.jsonl --manifest manifest.json --out report.json

# 2. Statistical screen: no provenance, population level only
embedreckon screen --export population.jsonl --out report.json

# 3. Compatibility probe: paired embeddings, same texts, two models
embedreckon probe --pairs pairs.jsonl --out report.json

# 4. Migration certification: refuse or certify before cutover
embedreckon certify --old old.jsonl --new new.jsonl \
    --manifest new-manifest.json --golden golden.jsonl \
    --threshold 0.35 --out certificate.json
```

Pure Python standard library. Fully offline. No model calls, no
provider keys, no connections to any vector store — it reads
exports you produce.

### Audit verdicts

Each partition is stamped **CONSISTENT**, **MIXED** or
**UNVERIFIABLE** with reason codes: `UNSTAMPED`,
`MANIFEST_MISMATCH` (record provenance disagrees with the claim),
`DIMENSION_MISMATCH`, `DUPLICATE_ID`,
`TRUNCATED_REPADDED_SIGNATURE` (a tranche truncated, then
zero-padded back to the declared dimension),
`WITHIN_PARTITION_MIXTURE_SIGNAL`,
`CROSS_PARTITION_SIGNATURE_MISMATCH`, `SIGNATURE_INCONCLUSIVE`.

The cross-partition leg is the one that catches the quiet case: a
partition written entirely by the wrong model, every record
claiming the declared one. Partitions claiming the same model are
compared pairwise; the largest mutually agreeing set defines the
claimed geometry, and a partition deviating from it is MIXED.
Two partitions that disagree with no majority between them are
both UNVERIFIABLE — the audit will not guess which is foreign.

CONSISTENT means *consistent with the declared provenance on every
leg that could be run* (the report lists the legs). It is never a
proof of single-space.

### The screen statistic

In a single space whose population shares any background component
(real corpora always induce one), strongly negative pairwise
cosines are rare; foreign-space vectors produce them at a
substantial rate. The screen samples pairs and measures the
fraction below cosine −0.05. Registered thresholds, measured on
the fixture arms (and re-measured by `validate.py` on every run):

- clean populations: **0.0000** — including quantized and
  truncated arms, which the screen refuses rather than reads;
- 5% same-dimension foreign tranche: **0.014–0.023** → SUSPECT-MIXED;
- 1% tranche: **0.002–0.008** → inside the refusal band →
  UNVERIFIABLE.

Measured separation (validate.py, kill criterion (i)'s fixture
arm): false-positive rate **0/100** clean populations; 5% tranche
detected in **11/12** populations (the twelfth refused, not
stamped clean); 1% tranche refused in **12/12**, never stamped
CONSISTENT.

### The probe's strongest word is "aligned, with measured loss"

The probe reports raw same-text cosine against a shuffled
baseline, unmapped neighbourhood preservation, and an orthogonal
Procrustes fit (stdlib Jacobi eigendecomposition) with its
residual, mapped cross-space preservation, and a hold-out
preservation (fit on half the pairs, evaluated on the rest).
Verdicts: **ALIGNED-WITH-MEASURED-LOSS** (loss numbers attached),
**INCOMPATIBLE**, **UNVERIFIABLE**. There is no "compatible"
verdict, unqualified, in the set — every alignment measured loses
something, and the loss is the deliverable.

### Certification has three legs and no partial credit

Completeness (the live id set exactly: missing ids and
resurrected tombstones refuse), single-space (the new index
passes the audit, and every partition is claimed by the
manifest), and retrieval equivalence (golden-set top-k overlap
against a declared threshold). Any leg failing — or impossible
to evaluate — refuses the certificate, with every leg's measured
values in the report either way.

## Input formats

Vector export (JSONL): `{"id", "partition", "vector",
"provenance"?}`; tombstones as `{"id", "tombstone": true}`.
Manifest (JSON): `{"format": "embedreckon-manifest/1", "index":
{...}, "partitions": {...}}` — model, model_version, dimension,
metric. Probe pairs (JSONL): `{"text_id", "a", "b"}`. Golden set
(JSONL): `{"query_id", "vector", "reference_top_k"}`. Full
specifications are in the module docstring of
`embedreckon/model.py`.

## Fixtures: the proof is planted

`fixtures/` builds two deterministic pseudo-embedding models —
seeded random projections of a shared latent topic space at the
same dimension — and plants hazards whose correct verdicts are
known in advance: a silently foreign partition, 5% and 1%
tranches, a truncated-and-repadded tranche, a migration missing
2% of ids and resurrecting deleted ones, a clean migration that
certifies, and the canonical failure (old-space vectors under
new-model provenance) refused on equivalence at overlap 0.008.
See `fixtures/README.md` for the full manifest of hazards and
measured outcomes, and run:

```bash
python3 validate.py   # unit suite + CLI smoke + screen measurement arm
```

## What v1 does not do

Named plainly (the full coverage statement is in `validate.py`):
direct readers for specific stores' export formats (v1 reads its
own documented JSONL/manifest formats); learned (non-orthogonal)
alignment maps; the ingest-side manifest writer (the landscape
scan's Position 3 — v1 consumes manifests, it does not write
them); and the publication-stage kill-criterion arms — the
screen's separation test on a real two-model index pair, and the
three-real-exports provenance census — which are evaluations to
run before public claims, not build-stage checks. The detector
claim ships bounded by the fixture measurement until those arms
run.

## License

MIT. Zero third-party dependencies. The IP/legal compliance
record — including the population-only / no-attribution boundary
as an original design decision — is in
[COMPLIANCE.md](COMPLIANCE.md); the name-collision record is in
[NAME_CHECK.md](NAME_CHECK.md).
