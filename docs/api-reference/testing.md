# `agentkit.testing`

Test doubles for every Protocol — `FakeLLM`, `FakeFetch`, `FakeSearch`,
`FakeMemory`, `FakeTool`, `FakeCompactor`, `FakeGrounder`, `FakeClock`
— plus the `make_test_ctx()` builder that wires them into a `RunContext`.

Two doubles are not port doubles, because the CLI cognitions are not
behind a port: `FakeClaudeCli` and `FakeCodexCli` stand in for the
subprocess itself (with the shared `CliRun` / `CliStderr` /
`CliInvocation` value types, and `codex_turn()` to build one well-formed
Codex turn), so a test still runs the real parsing, event mapping and
meter charging.

::: agentkit.testing
    options:
      show_root_heading: false
      show_source: false
      members_order: source
