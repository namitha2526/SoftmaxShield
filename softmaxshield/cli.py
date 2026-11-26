"""Command-line interface for SoftmaxShield static verification engine."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from softmaxshield.core.config import ShieldConfig
from softmaxshield.corrector.engine import SelfCorrectionEngine
from softmaxshield.pipeline import ShieldPipeline
from softmaxshield.reporter.formatter import ConsoleReporter, JSONReporter


def setup_logging(verbose: bool) -> None:
    """Configure console logging based on verbosity flag."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="softmaxshield",
        description="SoftmaxShield: Multi-layer Static Verification Engine for LLM-Generated Python Code",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 'check' subcommand
    check_parser = subparsers.add_parser("check", help="Verify a Python file for hallucinations")
    check_parser.add_argument("file", help="Path to Python file to verify")
    check_parser.add_argument("--json", action="store_true", help="Output report in JSON format")
    check_parser.add_argument("--strict", action="store_true", help="Enable strict verification mode")
    check_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for token probability analysis (default: 0.85)",
    )
    check_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable detailed diagnostic logging"
    )
    check_parser.add_argument(
        "--logprobs-file",
        type=str,
        default=None,
        help="Optional path to JSON file containing token-level log probabilities",
    )
    check_parser.add_argument(
        "--auto-correct",
        "--fix",
        action="store_true",
        dest="auto_correct",
        help="Trigger the self-correction engine if issues are detected",
    )
    check_parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Maximum self-correction rounds (default: 3)",
    )
    check_parser.add_argument(
        "--output", "-o", type=str, default=None, help="Save report to specified output path"
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for SoftmaxShield CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "check":
        setup_logging(args.verbose)

        target_path = Path(args.file)
        if not target_path.exists():
            sys.stderr.write(f"Error: File '{args.file}' not found.\n")
            return 2

        # Load token logprobs if provided
        token_logprobs = None
        if args.logprobs_file:
            logprobs_path = Path(args.logprobs_file)
            if not logprobs_path.exists():
                sys.stderr.write(f"Error: Logprobs file '{args.logprobs_file}' not found.\n")
                return 2
            try:
                token_logprobs = json.loads(logprobs_path.read_text(encoding="utf-8"))
            except Exception as err:
                sys.stderr.write(f"Error parsing logprobs JSON: {err}\n")
                return 2

        # Configure Shield
        config = ShieldConfig(
            strict_mode=args.strict,
            confidence_threshold=args.confidence_threshold,
            max_correction_rounds=args.max_rounds,
        )

        pipeline = ShieldPipeline(config)

        # Run Self-Correction if requested
        if args.auto_correct:
            source_code = target_path.read_text(encoding="utf-8", errors="replace")
            corrector = SelfCorrectionEngine(pipeline=pipeline, config=config)
            result = corrector.run(source_code, file_name=str(target_path))

            # Write back corrected code if resolved
            if result.success and result.rounds_attempted > 0:
                target_path.write_text(result.final_code, encoding="utf-8")
                sys.stdout.write(f"Successfully auto-corrected '{target_path}' in {result.rounds_attempted} round(s).\n")

            report = result.final_report or pipeline.verify_code(result.final_code, file_name=str(target_path))
        else:
            report = pipeline.verify_file(target_path, token_logprobs=token_logprobs)

        # Format output
        output_str = JSONReporter.format(report) if args.json else ConsoleReporter.format(report, verbose=args.verbose)

        if args.output:
            Path(args.output).write_text(output_str, encoding="utf-8")
        else:
            sys.stdout.write(output_str + "\n")

        return 0 if report.passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

