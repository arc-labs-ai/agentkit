"""Discover and run opt-in real workflow or conformance evaluation matrices."""

import argparse
import asyncio
import json
from decimal import Decimal
from pathlib import Path

from agentkit.core.contracts.types import Mode
from tests.evals.catalog import describe_cases
from tests.evals.coverage import summarize_coverage
from tests.evals.engine.config import MODES, EvalConfig
from tests.evals.engine.contracts import EvalDomain
from tests.evals.engine.runner import run_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded real application workflows across four agentkit backends")
    parser.add_argument(
        "--list", action="store_true", help="show catalogue without resolving credentials or calling models"
    )
    parser.add_argument("--coverage", action="store_true", help="show planned feature witnesses and missing evidence")
    parser.add_argument("--mode", action="append", type=Mode, choices=MODES)
    parser.add_argument("--suite", action="append", choices=("workload", "smoke", "contract", "probes"))
    parser.add_argument("--domain", action="append", type=EvalDomain, choices=tuple(EvalDomain))
    parser.add_argument("--case", action="append")
    parser.add_argument("--env-file", type=Path, default=EvalConfig().env_file)
    parser.add_argument("--report-dir", type=Path, default=EvalConfig().report_dir)
    parser.add_argument("--price-file", type=Path, help="versioned host price table for custom model configurations")
    parser.add_argument("--max-total-usd", type=Decimal, default=EvalConfig().max_total_usd)
    parser.add_argument("--max-case-usd", type=Decimal, default=EvalConfig().max_case_usd)
    parser.add_argument("--jobs", type=int, default=EvalConfig().jobs)
    parser.add_argument(
        "--repetitions", type=int, default=1, help="independent trials; budget is shared across all trials"
    )
    args = parser.parse_args()
    if args.list:
        print(json.dumps(describe_cases(), indent=2, default=str))
        return
    if args.coverage:
        print(json.dumps(summarize_coverage().as_dict(), indent=2))
        return
    config = EvalConfig(
        modes=tuple(args.mode or MODES),
        env_file=args.env_file,
        report_dir=args.report_dir,
        max_total_usd=args.max_total_usd,
        max_case_usd=args.max_case_usd,
        jobs=args.jobs,
        repetitions=args.repetitions,
        price_file=args.price_file,
    )
    path, passed = asyncio.run(
        run_matrix(
            config,
            suites=tuple(args.suite or ("workload",)),
            case_filter=tuple(args.case or ()),
            domain_filter=tuple(args.domain or ()),
        )
    )
    print(f"Report: {path}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
