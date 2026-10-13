"""Run the support example with a host-configured model credential."""

import argparse
import asyncio
import os

from agentkit.core.contracts.types import Mode
from agentkit.core.events import EventHub
from agentkit.core.runtime import CallRuntime
from agentkit.models import Binding, CredentialRef
from agentkit.tools import InMemoryFiles
from tests.examples.support_harness import SupportHarness


async def _run(mode: Mode, model: str) -> None:
    credentials = CredentialRef(
        lambda: {
            name: value
            for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
            if (value := os.environ.get(name))
        }
    )
    files = InMemoryFiles()
    await files.create(
        "/orgs/1/approved/refunds.md",
        "Refund policy v1: a captured duplicate payment is eligible for review. "
        "Refunds require an approved finance action; support must never claim one was issued.",
    )
    async with CallRuntime() as runtime:
        async with EventHub() as events:
            harness = SupportHarness(
                runtime,
                Binding(mode, model, credentials),
                org_id=1,
                files=files,
                orders={"order-1042": "Order 1042: duplicate payment captured; no refund issued."},
                policy_documents=(("/orgs/1/approved/refunds.md", await files.view("/orgs/1/approved/refunds.md")),),
                events=events,
            )
            answer = await harness.respond("What should we do about this duplicate payment?", order_id="order-1042")
            print(f"{answer.outcome}: {answer.text}")
            # Saving a model note is an explicit host decision, separate from answering.


def main() -> None:
    parser = argparse.ArgumentParser(description="Support workflow using public agentkit contracts")
    parser.add_argument("--mode", type=Mode, choices=tuple(Mode), required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.mode, args.model))


if __name__ == "__main__":
    main()
