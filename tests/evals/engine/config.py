"""Explicit eval configuration and secret loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from agentkit.core.contracts.types import Mode
from agentkit.models.catalog.pricing import PriceTable
from agentkit.models.contracts.types import Binding, CallLimits, CredentialRef

ROOT = Path(__file__).resolve().parents[3]
MODES = tuple(Mode)
DEFAULT_MODELS = {
    Mode.CLAUDE_CODE: "claude-haiku-4-5",
    Mode.CODEX: "gpt-5.4-mini",
    Mode.ANTHROPIC_API: "claude-haiku-4-5",
    Mode.OPENAI_API: "gpt-5.4-mini",
}


@dataclass(frozen=True)
class EvalConfig:
    modes: tuple[Mode, ...] = MODES
    env_file: Path = ROOT / ".env.eval"
    report_dir: Path = ROOT / "tests/evals/reports"
    max_total_usd: Decimal = Decimal("8")
    max_case_usd: Decimal = Decimal("1")
    wall_clock_s: float = 120
    jobs: int = 4
    repetitions: int = 1
    price_file: Path | None = None

    def __post_init__(self) -> None:
        if not self.max_total_usd.is_finite() or self.max_total_usd <= 0:
            raise ValueError("matrix budget must be finite and positive")
        if not self.max_case_usd.is_finite() or self.max_case_usd <= 0:
            raise ValueError("case budget must be finite and positive")
        if self.wall_clock_s <= 0:
            raise ValueError("eval wall clock must be positive")
        if self.jobs < 1:
            raise ValueError("evaluation concurrency must be positive")
        if self.repetitions < 1:
            raise ValueError("evaluation repetitions must be positive")

    def credentials(self) -> CredentialRef:
        from dotenv import dotenv_values

        values = dotenv_values(self.env_file, interpolate=False)
        # Credentials stay in a resolver, never the serializable config/report.
        keys = {
            k: str(values.get(k) or os.environ.get(k) or "")
            for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
        }
        return CredentialRef(lambda: {k: v for k, v in keys.items() if v})

    def binding(self, mode: Mode, credential: CredentialRef | None) -> Binding:
        model = os.environ.get(f"AGENTKIT_EVAL_MODEL_{mode.upper()}", DEFAULT_MODELS[mode])
        return Binding(mode, model, credential)

    def limits(self) -> CallLimits:
        return CallLimits(
            max_cost_usd=self.max_case_usd,
            max_turns=8,
            max_output_tokens=2048,
            wall_clock_s=self.wall_clock_s,
            idle_s=45,
            first_event_s=45,
        )

    def prices(self) -> PriceTable:
        return PriceTable.load(self.price_file) if self.price_file else PriceTable.default()
