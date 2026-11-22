#!/usr/bin/env python3
"""Validation harness for embedreckon.

Runs (1) the full unit test suite over the planted-hazard
fixtures, (2) end-to-end CLI smoke tests for all four commands
asserting the headline verdicts in the JSON reports, and (3) the
screen measurement arm -- kill criterion (i)'s fixture half:
the statistical screen's separation performance measured across
many populations (false-positive rate on clean arms including
quantized and truncated ones; detection at a 5% foreign tranche;
refusal at a 1% tranche). Exit 0 only if everything passes.
Failures are fixed in the code, never by weakening these checks.

The coverage statement at the end is part of the deliverable: it
states honestly which of the landscape scan's Position 1 claims
v0.1.0 implements, which are partial, and which are not in v1.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fixtures"))

import build_fixtures as bf  # noqa: E402
from embedreckon.screen import screen_population  # noqa: E402

COVERAGE_STATEMENT = """\
== coverage statement: scan Position 1 claims vs v1 ==
IMPLEMENTED (6):
  1. Provenance audit: every partition stamped CONSISTENT / MIXED /
     UNVERIFIABLE with machine-readable reason codes (UNSTAMPED,
     MANIFEST_MISMATCH, DIMENSION_MISMATCH, DUPLICATE_ID,
     TRUNCATED_REPADDED_SIGNATURE, WITHIN_PARTITION_MIXTURE_SIGNAL,
     CROSS_PARTITION_SIGNATURE_MISMATCH, SIGNATURE_INCONCLUSIVE);
     MIXED is reserved for positive evidence, missing evidence is
     UNVERIFIABLE, and two disagreeing partitions with no majority
     are refused rather than guessed.
  2. Statistical screen at population level under the published
     refusal boundary: SINGLE-SPACE-CONSISTENT (never "verified"),
     SUSPECT-MIXED with the measured statistic, UNVERIFIABLE for
     populations below the separation floor, too small, quantized,
     or uniformly truncated. No per-vector claims, no attribution
     of any vector to a named model, no translator shipped -- the
     scan's dual-use boundary is enforced by the verdict model.
  3. Compatibility probe over user-supplied paired embeddings:
     raw-vs-shuffled cosine, unmapped preservation, orthogonal
     Procrustes (stdlib Jacobi) with residual, mapped and hold-out
     cross-space preservation. Strongest verdict is
     ALIGNED-WITH-MEASURED-LOSS with the loss numbers attached;
     "compatible" unqualified is not in the verdict set.
  4. Migration certification on three independent legs
     (completeness incl. resurrected tombstones, single-space via
     the audit, golden-set retrieval equivalence vs a declared
     threshold); any leg FAIL or UNVERIFIABLE refuses the
     certificate -- there is no partial certification. The
     certificate names every check, threshold and measured value.
  5. Fixtures with planted hazards and known-correct verdicts:
     a silently foreign partition, 5% and 1% tranches, a
     truncated-and-repadded tranche, a migration missing 2% of
     ids and resurrecting deleted ones, a clean migration that
     certifies, and the canonical reel failure (old-space vectors
     under new-model provenance) refused on equivalence.
  6. The screen's separation performance is MEASURED, not
     asserted: see the measurement arm below (kill criterion
     (i)'s fixture half), re-run on every validate.
PARTIAL (2):
  7. Input formats: v1 reads embedreckon's own JSONL export and
     manifest formats only. Direct readers for Qdrant / Weaviate /
     pgvector exports are future work; any store that can export
     ids + vectors + provenance fields to JSONL can be audited
     today, but the conversion is the user's.
  8. Compatibility probe: orthogonal (rotation-only) alignment.
     The scan's research line also supports relative-representation
     and learned-map comparisons; v1 measures the orthogonal route
     and reports its residual honestly rather than shipping a
     learned translator (which the dual-use boundary excludes).
NOT IN V1 (3):
  9. Kill criterion (i)'s second arm -- the same separation test
     on one real two-model index pair built from public embedding
     datasets -- is a publication-stage evaluation, NOT run at
     build. The detector claim ships bounded by the fixture
     measurement until that arm runs.
 10. Kill criterion (ii)'s arm -- three real-world index exports
     and the fraction arriving with no usable provenance and no
     constructible golden set -- is likewise publication-stage
     and NOT run at build.
 11. No ingest-side manifest writer (the scan's Position 3): v1
     consumes manifests; writing them at ingest is a separate
     tool. A quantseal receipt can be referenced by a manifest,
     but v1 does not join the two automatically.
KILL CRITERION (iii) RE-CHECKED AT BUILD (2026-11-22): no vector
store (Qdrant / Weaviate / LanceDB) and no eval vendor ships an
independent mixed-space audit or pre-cutover migration
certification emitting per-partition verdicts. The nearest
adjacent entrant found is an ingest-side signing/policy gateway
(prevention at write time, waitlist product) -- it does not audit
existing indexes or certify migrations, so differentiation holds.
"""


def run_cli(argv, out_path):
    cmd = [sys.executable, "-m", "embedreckon"] + argv + ["--out", out_path]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"CLI failed: {' '.join(argv)}\n{proc.stdout}\n{proc.stderr}")
    with open(out_path, encoding="utf-8") as fh:
        return json.load(fh)


def cli_smoke_checks(tmp):
    fix = os.path.join(ROOT, "fixtures")
    checks = []

    report = run_cli(["audit", "--export", f"{fix}/audit_index.jsonl",
                      "--manifest", f"{fix}/audit_manifest.json"],
                     os.path.join(tmp, "audit.json"))
    stamps = {k: v["stamp"] for k, v in report["partitions"].items()}
    checks.append(("cli audit stamps", stamps == {
        "docs-a": "CONSISTENT", "docs-b": "MIXED", "docs-foreign": "MIXED",
        "docs-mixed": "MIXED", "docs-corrupt": "MIXED",
        "docs-unstamped": "UNVERIFIABLE"}))

    for name, want in (("screen_clean", "SINGLE-SPACE-CONSISTENT"),
                       ("screen_mixed5", "SUSPECT-MIXED"),
                       ("screen_mixed1", "UNVERIFIABLE")):
        report = run_cli(["screen", "--export", f"{fix}/{name}.jsonl"],
                         os.path.join(tmp, f"{name}.json"))
        got = report["partitions"]["population"]["verdict"]
        checks.append((f"cli screen {name}", got == want))

    report = run_cli(["probe", "--pairs", f"{fix}/probe_pairs_ab.jsonl"],
                     os.path.join(tmp, "probe_ab.json"))
    checks.append(("cli probe ab",
                   report["verdict"] == "ALIGNED-WITH-MEASURED-LOSS"))
    report = run_cli(["probe", "--pairs", f"{fix}/probe_pairs_unrelated.jsonl"],
                     os.path.join(tmp, "probe_unrel.json"))
    checks.append(("cli probe unrelated", report["verdict"] == "INCOMPATIBLE"))

    for name, want, reason in (
            ("mig_clean_new", "CERTIFIED", None),
            ("mig_broken_new", "REFUSED", "COMPLETENESS_FAILED"),
            ("mig_reel_new", "REFUSED", "RETRIEVAL_REGRESSION")):
        report = run_cli(["certify", "--old", f"{fix}/mig_old.jsonl",
                          "--new", f"{fix}/{name}.jsonl",
                          "--manifest", f"{fix}/mig_manifest_b.json",
                          "--golden", f"{fix}/mig_golden.jsonl"],
                         os.path.join(tmp, f"cert_{name}.json"))
        ok = report["verdict"] == want and (
            reason is None or reason in report["refusal_reasons"])
        checks.append((f"cli certify {name}", ok))
    return checks


def measurement_arm():
    """Kill criterion (i), fixture half: measure, don't assert."""
    rows = []

    def arm(label, seeds, n, pairs, **kw):
        verdicts = Counter()
        fracs = []
        for seed in seeds:
            res = screen_population(bf.population(seed, n, **kw), pairs=pairs)
            verdicts[res["verdict"]] += 1
            if "frac_neg" in res["statistics"]:
                fracs.append(res["statistics"]["frac_neg"])
        rows.append((label, len(seeds), verdicts, fracs))
        return verdicts

    clean = arm("clean", range(1000, 1100), 600, 4000)
    quant = arm("clean, scalar-quantized", range(2000, 2012), 600, 4000,
                quantize=True)
    trunc = arm("clean, uniformly truncated", range(3000, 3012), 600, 4000,
                trunc=32)
    five = arm("5% foreign tranche", range(4000, 4012), 800, 6000, frac_b=0.05)
    one = arm("1% foreign tranche", range(5000, 5012), 800, 6000, frac_b=0.01)

    print("== screen measurement arm (kill criterion (i), fixture half) ==")
    for label, n, verdicts, fracs in rows:
        frac_txt = ""
        if fracs:
            frac_txt = f" [frac_neg min {min(fracs):.4f} max {max(fracs):.4f}]"
        print(f"  {label} (n={n} populations): {dict(verdicts)}{frac_txt}")

    fp = clean["SUSPECT-MIXED"]
    checks = [
        (f"clean-arm false positives {fp}/100 <= 1% (kill criterion (i))", fp <= 1),
        ("quantized clean arms: no SUSPECT-MIXED",
         quant["SUSPECT-MIXED"] == 0),
        ("truncated clean arms: no SUSPECT-MIXED",
         trunc["SUSPECT-MIXED"] == 0),
        (f"5% tranche detected {five['SUSPECT-MIXED']}/12 >= 11",
         five["SUSPECT-MIXED"] >= 11),
        (f"1% tranche refused {one['UNVERIFIABLE']}/12 >= 10, none CONSISTENT",
         one["UNVERIFIABLE"] >= 10 and one["SINGLE-SPACE-CONSISTENT"] == 0),
    ]
    return checks


def main():
    failures = []

    print("== unit test suite ==")
    suite = unittest.TestLoader().discover(os.path.join(ROOT, "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        failures.append("unit test suite")

    print("== CLI smoke checks ==")
    with tempfile.TemporaryDirectory() as tmp:
        for label, ok in cli_smoke_checks(tmp):
            print(f"  {'PASS' if ok else 'FAIL'}: {label}")
            if not ok:
                failures.append(label)

    for label, ok in measurement_arm():
        print(f"  {'PASS' if ok else 'FAIL'}: {label}")
        if not ok:
            failures.append(label)

    print(COVERAGE_STATEMENT)
    if failures:
        print("VALIDATION FAILED:", "; ".join(failures))
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
