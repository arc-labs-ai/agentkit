"""Execution requirements and evidence; containment never means tools are absent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class NativeExecution(StrEnum):
    """Opt in to native execution with mandatory security boundaries."""

    UNCONSTRAINED = "unconstrained"
    CONTAINED = "contained"


class ExecutionRequirement(StrEnum):
    """Independent requirements; satisfying one does not imply any other."""

    NATIVE_TOOLS_UNAVAILABLE = "native_tools_unavailable"
    HOST_TOOLS = "host_tools"
    MCP_ALLOWLIST = "mcp_allowlist"
    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    NETWORK = "network"
    PROCESSES = "processes"
    CREDENTIALS = "credentials"
    CONFIGURATION = "configuration"
    SCHEMA = "schema"
    CANCELLATION = "cancellation"
    USAGE = "usage"
    OBSERVED_SPENDING_THRESHOLD = "observed_spending_threshold"
    HARD_SPENDING_CAP = "hard_spending_cap"


class CapabilityStatus(StrEnum):
    """Unverified is a refusal when the capability is required."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class ExecutionRequirements:
    """Caller requirements, never evidence that an isolation mechanism exists.

    Empty roots/destinations deny access; None leaves that boundary unspecified.
    Contained execution requires every security boundary, including credentials.
    """

    native_execution: NativeExecution = NativeExecution.UNCONSTRAINED
    required: frozenset[ExecutionRequirement] = frozenset()
    read_roots: tuple[Path | str, ...] | None = None
    write_roots: tuple[Path | str, ...] | None = None
    network_destinations: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "native_execution", NativeExecution(self.native_execution))
        required = {ExecutionRequirement(x) for x in self.required}
        for field, requirement in (
            ("read_roots", ExecutionRequirement.FILESYSTEM_READ),
            ("write_roots", ExecutionRequirement.FILESYSTEM_WRITE),
        ):
            roots = getattr(self, field)
            if roots is not None:
                if any(not Path(root).is_absolute() or ".." in Path(root).parts for root in roots):
                    raise ValueError("execution roots must be absolute paths without parent traversal")
                object.__setattr__(self, field, tuple(Path(root) for root in roots))
                required.add(requirement)
        if self.network_destinations is not None:
            if any(not isinstance(x, str) or not x.strip() for x in self.network_destinations):
                raise ValueError("network destinations must be nonempty strings")
            object.__setattr__(self, "network_destinations", tuple(self.network_destinations))
            required.add(ExecutionRequirement.NETWORK)
        if self.native_execution == NativeExecution.CONTAINED:
            for boundary in ("read_roots", "write_roots", "network_destinations"):
                if getattr(self, boundary) is None:
                    object.__setattr__(self, boundary, ())
            required.update(
                {
                    ExecutionRequirement.FILESYSTEM_READ,
                    ExecutionRequirement.FILESYSTEM_WRITE,
                    ExecutionRequirement.NETWORK,
                    ExecutionRequirement.PROCESSES,
                    ExecutionRequirement.CREDENTIALS,
                    ExecutionRequirement.CONFIGURATION,
                    ExecutionRequirement.CANCELLATION,
                }
            )
        object.__setattr__(self, "required", frozenset(required))

    def constrain(self, ceiling: ExecutionRequirements) -> ExecutionRequirements:
        """Retain a session's requirements and intersect explicit authorities.

        This pure operation narrows requests; it is not filesystem enforcement.
        Symlinks still require a verified backend, which no shipped profile has.
        """

        def roots(
            left: tuple[Path | str, ...] | None, right: tuple[Path | str, ...] | None
        ) -> tuple[Path | str, ...] | None:
            if left is None:
                return right
            if right is None:
                return left
            return tuple(
                sorted(
                    {
                        Path(a) if Path(a).is_relative_to(b) else Path(b)
                        for a in left
                        for b in right
                        if Path(a).is_relative_to(b) or Path(b).is_relative_to(a)
                    }
                )
            )

        destinations = self.network_destinations
        if ceiling.network_destinations is not None:
            destinations = (
                ceiling.network_destinations
                if destinations is None
                else tuple(x for x in destinations if x in ceiling.network_destinations)
            )
        return ExecutionRequirements(
            native_execution=(
                NativeExecution.CONTAINED
                if NativeExecution.CONTAINED in (self.native_execution, ceiling.native_execution)
                else NativeExecution.UNCONSTRAINED
            ),
            required=self.required | ceiling.required,
            read_roots=roots(self.read_roots, ceiling.read_roots),
            write_roots=roots(self.write_roots, ceiling.write_roots),
            network_destinations=destinations,
        )


@dataclass(frozen=True)
class ExecutionAssessment:
    """Safe, secret-free capability fact and stable machine-readable reason."""

    requirement: ExecutionRequirement
    status: CapabilityStatus
    code: str
    explanation: str


@dataclass(frozen=True)
class ExecutionCompatibility:
    """Full matrix and the subset which prevents this invocation."""

    mode: str
    backend: str
    backend_version: str
    binary_version: str | None
    assessments: tuple[ExecutionAssessment, ...]
    incompatibilities: tuple[ExecutionAssessment, ...]
    adapter: str = ""

    @property
    def compatible(self) -> bool:
        return not self.incompatibilities
