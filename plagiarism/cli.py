# plagiarism/cli.py
import argparse
import json
from .config import load_config
from .compare import compare_two_files, compare_hierarchies


def parse_args():

    p = argparse.ArgumentParser(description="Plagiarism detection pipeline (modular).")
    p.add_argument("file_a", help="First Python file")
    p.add_argument("file_b", help="Second Python file")
    p.add_argument("--config", "-c", help="YAML config file (optional)", default=None)
    p.add_argument("--hierarchy", "-H", action="store_true", help="Show hierarchical per-part similarity")
    p.add_argument("--json", "-j", help="Write JSON report to given file (optional)", default=None)
    return p.parse_args()


def main():

    args = parse_args()
    config = load_config(args.config)
    try:
        if args.hierarchy:

            report = compare_hierarchies(args.file_a, args.file_b, config)
            print(f"Hierarchical comparison: {args.file_a} <-> {args.file_b}")

            print("Functions:")
            for fn, info in report.get("functions", {}).items():
                print(f"  {fn}: final={info['final']:.4f} (token={info['token']:.4f}, ast={info['ast']:.4f}, cfg={info['cfg']})")
            print("Classes:")
            for cn, cinfo in report.get("classes", {}).items():
                print(f"  {cn}: final={cinfo['final']:.4f} (token={cinfo['token']:.4f}, ast={cinfo['ast']:.4f}, cfg={cinfo['cfg']})")
                for m, minfo in cinfo.get("methods", {}).items():
                    print(f"    {m}: final={minfo['final']:.4f} (token={minfo['token']:.4f}, ast={minfo['ast']:.4f}, cfg={minfo['cfg']})")
            print("Variables:")
            for v, vinfo in report.get("variables", {}).items():
                print(f"  {v}: final={vinfo['final']:.4f} (token={vinfo['token']:.4f}, ast={vinfo['ast']:.4f})")


            if args.json:
                with open(args.json, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2)
                print(f"JSON report written to {args.json}")
        else:

            result = compare_two_files(args.file_a, args.file_b, config)
            print("Comparison results:")
            print(f"  token similarity: {result['token']:.4f}")
            print(f"  AST similarity:   {result['ast']:.4f}")
            print(f"  CFG similarity:   {result['cfg']:.4f}")
            print(f"  FINAL score:      {result['final']:.4f}")
            print()
            import os
            print(f"Normalized tokens {os.path.basename(args.file_a)}: {len(result['normalized_tokens_a'])} tokens")
            print(f"Normalized tokens {os.path.basename(args.file_b)}: {len(result['normalized_tokens_b'])} tokens")
            if args.json:
                with open(args.json, "w", encoding="utf-8") as fh:
                    json.dump(result, fh, indent=2)
                print(f"JSON summary written to {args.json}")
    except Exception as e:
        print("Error during comparison:", e)
        raise