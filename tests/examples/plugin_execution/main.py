"""Credential-free assessment by default; --execute explicitly enables a model call."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from agentkit.core.runtime import CallRuntime
from agentkit.models import ExecutionProfile, ExecutionRequirements
from tests.examples.plugin_execution.configuration.providers import MODES, provider_binding
from tests.examples.plugin_execution.wiring.runtime import compose_call, execute


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=MODES, default="openai_api")
    parser.add_argument("--model")
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in ExecutionProfile],
        default=ExecutionProfile.HOST_TOOLS_ONLY.value,
        help="Explicit execution contract; CLI-compatible permits native behavior and is not containment",
    )
    parser.add_argument("--auth-file", type=Path, help="Explicit native saved-session credential file")
    parser.add_argument("--execute", action="store_true", help="Authorize a potentially paid provider call")
    options = parser.parse_args()
    binding = provider_binding(options.provider, model=options.model, auth_file=options.auth_file)
    requirements = ExecutionRequirements(profile=options.profile)
    async with CallRuntime(native_tool_policy="warn", unpriced_policy="proceed_unpriced") as runtime:
        assessment = await runtime.evaluate_execution(binding, execution=requirements)
        if not options.execute:
            print(json.dumps({"compatible": assessment.compatible, **asdict(assessment)}, default=str, indent=2))
            return
        call = compose_call("plugin-example")
        result = await execute(runtime, binding, call, execution=requirements)
        print(
            json.dumps(
                {
                    "outcome": result.outcome,
                    "attempts": result.attempts,
                    "value": result.value.model_dump() if result.value else None,
                    "failure": asdict(result.failure) if result.failure else None,
                    "usage_complete": result.usage_complete,
                    "cost_usd": str(result.cost_usd) if result.cost_usd is not None else None,
                    "events": call.lifecycle.events,
                },
                default=str,
                indent=2,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
