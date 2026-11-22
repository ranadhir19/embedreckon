"""Test suite for embedreckon.

The fixture verdicts are known in advance (planted hazards with
planted-correct stamps); the suite asserts the tool returns them.
Failures are fixed in the code, never by weakening these tests.
"""

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fixtures"))

import build_fixtures as bf  # noqa: E402

from embedreckon import stats  # noqa: E402
from embedreckon.audit import audit_export  # noqa: E402
from embedreckon.certify import certify  # noqa: E402
from embedreckon.model import Record, load_export, load_golden, load_manifest, load_pairs  # noqa: E402
from embedreckon.probe import probe_pairs  # noqa: E402
from embedreckon.screen import screen_population  # noqa: E402

FIX = os.path.join(ROOT, "fixtures")
PAIRS = 3000  # reduced sampling for test speed; verdicts are unchanged


def load(name):
    return load_export(os.path.join(FIX, name))


class AuditFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records, issues = load("audit_index.jsonl")
        assert not issues
        manifest = load_manifest(os.path.join(FIX, "audit_manifest.json"))
        cls.report = audit_export(records, manifest, pairs=PAIRS)

    def stamp(self, name):
        return self.report["partitions"][name]

    def test_clean_partition_consistent(self):
        part = self.stamp("docs-a")
        self.assertEqual(part["stamp"], "CONSISTENT")
        self.assertEqual(part["reason_codes"], [])

    def test_version_mismatch_is_mixed(self):
        part = self.stamp("docs-b")
        self.assertEqual(part["stamp"], "MIXED")
        self.assertIn("MANIFEST_MISMATCH", part["reason_codes"])

    def test_foreign_partition_caught_by_signature(self):
        part = self.stamp("docs-foreign")
        self.assertEqual(part["stamp"], "MIXED")
        self.assertIn("CROSS_PARTITION_SIGNATURE_MISMATCH", part["reason_codes"])
        # Its own population is a perfectly consistent single space --
        # the wrong one. The within screen alone cannot catch it.
        self.assertEqual(part["screen"]["verdict"], "SINGLE-SPACE-CONSISTENT")

    def test_mixed_partition_caught(self):
        part = self.stamp("docs-mixed")
        self.assertEqual(part["stamp"], "MIXED")
        self.assertIn("WITHIN_PARTITION_MIXTURE_SIGNAL", part["reason_codes"])

    def test_corrupt_partition_caught(self):
        part = self.stamp("docs-corrupt")
        self.assertEqual(part["stamp"], "MIXED")
        self.assertIn("TRUNCATED_REPADDED_SIGNATURE", part["reason_codes"])

    def test_unstamped_partition_unverifiable(self):
        part = self.stamp("docs-unstamped")
        self.assertEqual(part["stamp"], "UNVERIFIABLE")
        self.assertIn("UNSTAMPED", part["reason_codes"])


class AuditSyntheticTests(unittest.TestCase):
    def _records(self, partition, vecs, model="pseudo-a", version="1.0"):
        return [Record(f"{partition}-{i}", partition, v,
                       {"model": model, "model_version": version,
                        "dimension": bf.DIM, "metric": "cosine"})
                for i, v in enumerate(vecs)]

    def test_duplicate_id(self):
        vecs = bf.population(11, 300)
        recs = self._records("p", vecs)
        recs.append(Record("p-0", "p", vecs[1],
                           {"model": "pseudo-a", "model_version": "1.0",
                            "dimension": bf.DIM, "metric": "cosine"}))
        manifest = {"partitions": {"p": {"model": "pseudo-a",
                                         "model_version": "1.0",
                                         "dimension": bf.DIM,
                                         "metric": "cosine"}}}
        report = audit_export(recs, manifest, pairs=PAIRS)
        self.assertEqual(report["partitions"]["p"]["stamp"], "MIXED")
        self.assertIn("DUPLICATE_ID", report["partitions"]["p"]["reason_codes"])

    def test_dimension_mismatch(self):
        vecs = [v[:32] for v in bf.population(12, 300)]
        recs = self._records("p", vecs)
        manifest = {"partitions": {"p": {"model": "pseudo-a",
                                         "model_version": "1.0",
                                         "dimension": bf.DIM,
                                         "metric": "cosine"}}}
        report = audit_export(recs, manifest, pairs=PAIRS)
        self.assertIn("DIMENSION_MISMATCH",
                      report["partitions"]["p"]["reason_codes"])
        self.assertEqual(report["partitions"]["p"]["stamp"], "MIXED")

    def test_two_disagreeing_partitions_no_majority(self):
        # One A partition, one B partition, both claiming model A:
        # they disagree and there is no majority -- the audit must
        # refuse to guess which one is foreign.
        recs = self._records("x", bf.population(13, 300))
        recs += self._records("y", bf.population(14, 300, frac_b=1.0))
        manifest = {"partitions": {
            name: {"model": "pseudo-a", "model_version": "1.0",
                   "dimension": bf.DIM, "metric": "cosine"}
            for name in ("x", "y")}}
        report = audit_export(recs, manifest, pairs=PAIRS)
        for name in ("x", "y"):
            part = report["partitions"][name]
            self.assertEqual(part["stamp"], "UNVERIFIABLE")
            self.assertIn("SIGNATURE_INCONCLUSIVE", part["reason_codes"])


class ScreenTests(unittest.TestCase):
    def _verdict(self, name):
        records, issues = load(name)
        assert not issues
        vecs = [r.vector for r in records]
        return screen_population(vecs)

    def test_clean_fixture(self):
        res = self._verdict("screen_clean.jsonl")
        self.assertEqual(res["verdict"], "SINGLE-SPACE-CONSISTENT")
        self.assertLessEqual(res["statistics"]["frac_neg"], stats.CLEAN_CEILING)

    def test_mixed5_fixture(self):
        res = self._verdict("screen_mixed5.jsonl")
        self.assertEqual(res["verdict"], "SUSPECT-MIXED")
        self.assertGreaterEqual(res["statistics"]["frac_neg"], stats.MIXTURE_FLOOR)

    def test_mixed1_fixture_is_the_refusal_band(self):
        # The 1% tranche is the teaching case: the honest stamp is
        # UNVERIFIABLE -- not a guessed MIXED, not a false CONSISTENT.
        res = self._verdict("screen_mixed1.jsonl")
        self.assertEqual(res["verdict"], "UNVERIFIABLE")
        self.assertIn("BELOW_SEPARATION_FLOOR", res["reason_codes"])

    def test_too_small_refused(self):
        res = screen_population(bf.population(21, 100))
        self.assertEqual(res["verdict"], "UNVERIFIABLE")
        self.assertIn("POPULATION_TOO_SMALL", res["reason_codes"])

    def test_quantized_refused(self):
        res = screen_population(bf.population(22, 400, quantize=True))
        self.assertEqual(res["verdict"], "UNVERIFIABLE")
        self.assertIn("QUANTIZED_POPULATION", res["reason_codes"])

    def test_uniformly_truncated_refused(self):
        res = screen_population(bf.population(23, 400, trunc=32))
        self.assertEqual(res["verdict"], "UNVERIFIABLE")
        self.assertIn("TRUNCATED_POPULATION", res["reason_codes"])


class ProbeTests(unittest.TestCase):
    def _probe(self, name):
        pairs, issues, meta = load_pairs(os.path.join(FIX, name))
        assert not issues
        return probe_pairs(pairs, meta)

    def test_ab_aligned_with_measured_loss(self):
        report = self._probe("probe_pairs_ab.jsonl")
        self.assertEqual(report["verdict"], "ALIGNED-WITH-MEASURED-LOSS")
        self.assertIn("measured_loss", report["measurements"])
        self.assertGreaterEqual(
            report["measurements"]["neighbour_preservation_mapped"], 0.40)
        # Raw cross-space cosine is at chance: the spaces are NOT
        # directly comparable, only alignable with loss.
        self.assertLess(report["measurements"]["raw_same_text_cosine"], 0.2)

    def test_unrelated_incompatible(self):
        report = self._probe("probe_pairs_unrelated.jsonl")
        self.assertEqual(report["verdict"], "INCOMPATIBLE")
        self.assertGreaterEqual(report["measurements"]["procrustes_residual"], 0.9)

    def test_too_few_pairs(self):
        pairs, _i, meta = load_pairs(os.path.join(FIX, "probe_pairs_ab.jsonl"))
        report = probe_pairs(pairs[:10], meta)
        self.assertEqual(report["verdict"], "UNVERIFIABLE")
        self.assertIn("TOO_FEW_PAIRS", report["reason_codes"])

    def test_same_model_raw_aligned(self):
        pairs, _i, meta = load_pairs(os.path.join(FIX, "probe_pairs_ab.jsonl"))
        same = [{"text_id": p["text_id"], "a": p["a"], "b": list(p["a"])}
                for p in pairs[:60]]
        report = probe_pairs(same, meta)
        self.assertEqual(report["verdict"], "ALIGNED-WITH-MEASURED-LOSS")
        self.assertIn("RAW_GEOMETRY_ALIGNED", report["reason_codes"])
        self.assertLess(report["measurements"]["measured_loss"]["mean_cosine_gap"], 0.01)

    def test_dimension_mismatch_incompatible(self):
        pairs, _i, meta = load_pairs(os.path.join(FIX, "probe_pairs_ab.jsonl"))
        cut = [{"text_id": p["text_id"], "a": p["a"], "b": p["b"][:32]}
               for p in pairs[:60]]
        report = probe_pairs(cut, meta)
        self.assertEqual(report["verdict"], "INCOMPATIBLE")
        self.assertIn("DIMENSION_MISMATCH", report["reason_codes"])

    def test_verdict_never_unqualified_compatible(self):
        for name in ("probe_pairs_ab.jsonl", "probe_pairs_unrelated.jsonl"):
            report = self._probe(name)
            self.assertNotEqual(report["verdict"], "COMPATIBLE")


class CertifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old, _ = load("mig_old.jsonl")
        cls.manifest = load_manifest(os.path.join(FIX, "mig_manifest_b.json"))
        cls.golden, _ = load_golden(os.path.join(FIX, "mig_golden.jsonl"))

    def _certify(self, new_name, manifest=None):
        new, _ = load(new_name)
        return certify(self.old, new, manifest or self.manifest,
                       self.golden, pairs=PAIRS)

    def test_clean_migration_certifies(self):
        cert = self._certify("mig_clean_new.jsonl")
        self.assertEqual(cert["verdict"], "CERTIFIED")
        for leg in cert["legs"].values():
            self.assertEqual(leg["status"], "PASS")
        self.assertGreaterEqual(
            cert["legs"]["retrieval_equivalence"]["mean_topk_overlap"], 0.35)

    def test_broken_migration_refused_on_completeness(self):
        cert = self._certify("mig_broken_new.jsonl")
        self.assertEqual(cert["verdict"], "REFUSED")
        self.assertIn("COMPLETENESS_FAILED", cert["refusal_reasons"])
        comp = cert["legs"]["completeness"]
        self.assertEqual(comp["missing_count"], 10)
        self.assertEqual(comp["resurrected_count"], 6)
        # The other legs still ran and are reported: no partial
        # certification, and no hiding behind the failed leg.
        self.assertEqual(cert["legs"]["retrieval_equivalence"]["status"], "PASS")

    def test_reel_scenario_refused_on_equivalence(self):
        # The canonical failure: migration "completed", provenance
        # claims the new model, vectors still in the old space.
        cert = self._certify("mig_reel_new.jsonl")
        self.assertEqual(cert["verdict"], "REFUSED")
        self.assertIn("RETRIEVAL_REGRESSION", cert["refusal_reasons"])
        self.assertLess(
            cert["legs"]["retrieval_equivalence"]["mean_topk_overlap"], 0.05)

    def test_unverifiable_leg_refuses(self):
        bad_manifest = {"format": "embedreckon-manifest/1", "index": {}}
        cert = self._certify("mig_clean_new.jsonl", manifest=bad_manifest)
        self.assertEqual(cert["verdict"], "REFUSED")
        self.assertIn("SINGLE_SPACE_UNVERIFIABLE", cert["refusal_reasons"])


class StatsTests(unittest.TestCase):
    def test_procrustes_recovers_known_rotation(self):
        import math
        import random
        rng = random.Random(1)
        d = 20
        a = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(50)]
        rot = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        for _ in range(30):
            p, q = rng.sample(range(d), 2)
            th = rng.uniform(0, 6.28)
            c, s = math.cos(th), math.sin(th)
            for k in range(d):
                rkp, rkq = rot[k][p], rot[k][q]
                rot[k][p] = c * rkp - s * rkq
                rot[k][q] = s * rkp + c * rkq
        b = [[sum(row[k] * rot[k][j] for k in range(d)) for j in range(d)]
             for row in a]
        _r, residual, _ratio = stats.procrustes(b, a)
        self.assertLess(residual, 1e-6)

    def test_frac_neg_clean_population(self):
        frac, _used = stats.frac_neg_within(bf.population(31, 600), pairs=3000)
        self.assertLessEqual(frac, stats.CLEAN_CEILING)


if __name__ == "__main__":
    unittest.main()
