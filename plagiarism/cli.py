import argparse
from .config import load_config
from .compare import compare_two_files


def parse_args():
    p = argparse.ArgumentParser(description="Plagiarism detection pipeline (modular).")
    p.add_argument("file_a", help="First Python file")
    p.add_argument("file_b", help="Second Python file")
    p.add_argument("--config", "-c", help="YAML config file (optional)", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    try:
        result = compare_two_files(args.file_a, args.file_b, config)
    except Exception as e:
        print("Error during comparison:", e)
        raise

    print("Comparison results:")
    print(f"  token similarity: {result['token']:.4f}")
    print(f"  AST similarity:   {result['ast']:.4f}")
    print(f"  CFG similarity:   {result['cfg']:.4f}")
    print(f"  FINAL score:      {result['final']:.4f}")
    print()
    import os
    print(f"Normalized tokens {os.path.basename(args.file_a)}: {len(result['normalized_tokens_a'])} tokens")
    print(f"Normalized tokens {os.path.basename(args.file_b)}: {len(result['normalized_tokens_b'])} tokens")
