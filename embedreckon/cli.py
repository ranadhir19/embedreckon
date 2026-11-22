"""Command-line interface.

    embedreckon audit   --export index.jsonl --manifest manifest.json
                        [--out report.json]
    embedreckon screen  --export population.jsonl [--out report.json]
    embedreckon probe   --pairs pairs.jsonl [--out report.json]
    embedreckon certify --old old.jsonl --new new.jsonl
                        --manifest new-manifest.json --golden golden.jsonl
                        [--threshold 0.35] [--out certificate.json]

Every command prints a human-readable summary and, with --out,
writes the full JSON report. Every report carries the refusal
boundary in its header.
"""

import argparse
import json
import sys

from . import __version__
from .audit import audit_export
from .certify import DEFAULT_OVERLAP_THRESHOLD, certify
from .model import load_export, load_golden, load_manifest, load_pairs
from .probe import probe_pairs
from .screen import screen_export


def _write_out(args, report):
    if getattr(args, "out", None):
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"report written: {args.out}")


def _note_issues(report, issues):
    if issues:
        report["parse_issues"] = issues
        print(f"WARNING: {len(issues)} input line(s) could not be parsed "
              "and were NOT silently dropped; see parse_issues in the report.")


def print_audit_summary(report):
    print("provenance audit")
    for name, part in report["partitions"].items():
        codes = ", ".join(part["reason_codes"]) or "-"
        print(f"  {name}: {part['stamp']} "
              f"({part['n_vectors']} vectors; reasons: {codes})")
    s = report["summary"]
    print("  summary: " + ", ".join(f"{k} {v}" for k, v in sorted(s.items())))


def cmd_audit(args):
    records, issues = load_export(args.export)
    manifest = load_manifest(args.manifest) if args.manifest else None
    report = audit_export(records, manifest)
    report["tool"] = f"embedreckon {__version__}"
    report["command"] = "audit"
    _note_issues(report, issues)
    print_audit_summary(report)
    _write_out(args, report)
    return 0


def print_screen_summary(report):
    print("statistical screen (population level only; no per-vector claims)")
    for name, res in report["partitions"].items():
        frac = res["statistics"].get("frac_neg")
        frac_txt = f"frac_neg {frac}" if frac is not None else "statistic not run"
        codes = ", ".join(res["reason_codes"])
        print(f"  {name}: {res['verdict']} ({res['n_vectors']} vectors; "
              f"{frac_txt}; reasons: {codes})")


def cmd_screen(args):
    records, issues = load_export(args.export)
    report = screen_export(records)
    report["tool"] = f"embedreckon {__version__}"
    report["command"] = "screen"
    _note_issues(report, issues)
    print_screen_summary(report)
    _write_out(args, report)
    return 0


def print_probe_summary(report):
    m = report["measurements"]
    print(f"compatibility probe: {report['model_a']} vs {report['model_b']} "
          f"({report['n_pairs']} pairs)")
    for key, value in m.items():
        print(f"  {key}: {value}")
    codes = ", ".join(report["reason_codes"]) or "-"
    print(f"  verdict: {report['verdict']} (reasons: {codes})")
    print("  note: the strongest verdict this probe issues is "
          "ALIGNED-WITH-MEASURED-LOSS; it never stamps a pair "
          "\"compatible\" unqualified.")


def cmd_probe(args):
    pairs, issues, meta = load_pairs(args.pairs)
    report = probe_pairs(pairs, meta)
    report["tool"] = f"embedreckon {__version__}"
    report["command"] = "probe"
    _note_issues(report, issues)
    print_probe_summary(report)
    _write_out(args, report)
    return 0


def print_certify_summary(report):
    legs = report["legs"]
    comp = legs["completeness"]
    print("migration certification")
    print(f"  completeness: {comp['status']} "
          f"(live old {comp['live_ids_old']}, new {comp['ids_new']}, "
          f"missing {comp['missing_count']}, "
          f"resurrected {comp['resurrected_count']}, "
          f"added {comp['added_ids_count']})")
    ss = legs["single_space"]
    stamps = ", ".join(f"{k} {v}" for k, v in ss["partition_stamps"].items())
    print(f"  single-space: {ss['status']} ({stamps})")
    eq = legs["retrieval_equivalence"]
    print(f"  retrieval equivalence: {eq['status']} "
          f"(mean top-k overlap {eq['mean_topk_overlap']} vs threshold "
          f"{eq['threshold']}, {eq['n_queries']} golden queries)")
    reasons = ", ".join(report["refusal_reasons"]) or "-"
    print(f"  verdict: {report['verdict']} (refusal reasons: {reasons})")


def cmd_certify(args):
    old_records, old_issues = load_export(args.old)
    new_records, new_issues = load_export(args.new)
    manifest = load_manifest(args.manifest)
    golden, golden_issues = load_golden(args.golden)
    report = certify(old_records, new_records, manifest, golden,
                     threshold=args.threshold)
    report["tool"] = f"embedreckon {__version__}"
    report["command"] = "certify"
    _note_issues(report, old_issues + new_issues + golden_issues)
    print_certify_summary(report)
    _write_out(args, report)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="embedreckon",
        description="The embedding-space auditor and migration certifier.")
    parser.add_argument("--version", action="version",
                        version=f"embedreckon {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="stamp every partition against a manifest")
    p.add_argument("--export", required=True)
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("screen", help="screen populations for mixed-space contamination")
    p.add_argument("--export", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("probe", help="measure a model pair's alignability on paired embeddings")
    p.add_argument("--pairs", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("certify", help="certify or refuse a migration before cutover")
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--golden", required=True)
    p.add_argument("--threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    p.add_argument("--out")
    p.set_defaults(func=cmd_certify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
