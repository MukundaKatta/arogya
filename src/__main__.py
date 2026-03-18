"""CLI for arogya."""
import sys, json, argparse
from .core import Arogya

def main():
    parser = argparse.ArgumentParser(description="Arogya — AI Diagnostic Bias Detector. Automated equity auditing framework for medical AI across 12 demographic dimensions.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Arogya()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.detect(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"arogya v0.1.0 — Arogya — AI Diagnostic Bias Detector. Automated equity auditing framework for medical AI across 12 demographic dimensions.")

if __name__ == "__main__":
    main()
