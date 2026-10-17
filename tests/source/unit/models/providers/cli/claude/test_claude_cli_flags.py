"""What the `claude` cognition compiles a configuration DOWN TO.

WHY THIS FILE EXISTS
--------------------
``ClaudeCliCognition`` is a declarative front end for an argument list. Every
field on it is a promise that some flag reaches the binary, spelled the way the
binary expects — and every one of those promises can break without a single
exception being raised anywhere. A tuple stringified with ``str()`` instead of
``",".join()`` still produces a ``--fallback-model`` value. A resume id passed
through ``--session-id`` still starts a session. A dropped ``if`` in
``__post_init__`` still constructs an object. The run goes green, the CLI does
something else, and the only symptom is a bill or a transcript nobody reads.

So the assertions here are about two seams and nothing else:

* **``__post_init__``** — the combinations that are refused BEFORE a spawn.
  Each of these is a configuration the CLI itself rejects two to five seconds
  in, on a stderr the caller has to parse, or (worse) silently accepts and
  ignores. The value of the pre-flight is entirely in it being there; a
  validation that stops firing is indistinguishable from one that never
  existed.
* **the argv builder**, reached through ``drive`` with an injected ``spawn=``
  rather than by calling ``_build_argv`` directly wherever that is practical.
  The injected double records the argv the cognition REALLY produced, after
  system-prompt resolution, secret materialisation and env construction — the
  three steps between a field and a flag where a field has actually been lost
  before.

Nothing here spawns a binary or needs one installed. ``test_cli_version_matrix``
is the file that asks the real CLI whether it still accepts these arguments;
this one asks whether we still build them.
"""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentkit.core.testing.fakes import FakeClaudeCli
from agentkit.models.providers.cli.claude.driver import (
    _BARE_CREDENTIAL_ENV,
    _CLAUDE_AUTH_ENV,
    ClaudeCliCognition,
)

# One successful turn. The content is irrelevant — every test below reads the
# RECORDED INVOCATION, not the answer — but a run that ends in a proper
# ``result`` payload is what keeps a failure here unambiguous: an assertion
# about argv that fails because the stream was malformed would be reported as
# an argv bug.
_OK = (
    {"type": "assistant", "message": {"id": "final", "content": [{"type": "text", "text": "ok"}]}},
    {
        "type": "result",
        "subtype": "success",
        "result": "ok",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
)


async def _argv(cognition: ClaudeCliCognition, cli: FakeClaudeCli, *, prompt: str | None = None) -> tuple[str, ...]:
    """Drive ``cognition`` once against ``cli`` and return the argv it spawned.

    Drains the stream to the terminal ``final`` event rather than breaking out
    early, so the cognition's own teardown (scratch removal, meter charge) runs
    and cannot leave a ``ResourceWarning`` for a later test to trip over under
    ``filterwarnings = ["error"]``.

    The agent is a ``SimpleNamespace`` because that is what the production
    caller hands this method (``CliAdapter`` builds exactly this), and because
    ``_resolve_json_schema`` reads the output adapter with ``getattr`` for this
    reason: nothing here needs a real agent, and building one would couple a
    flags test to the agent constructor.
    """
    agent = SimpleNamespace(name="flags", prompt=prompt, output=None)
    async for event in cognition.drive(agent, "do the thing", None, None):
        if event.type == "final":
            assert event.result is not None and not event.result.partial, event.result.evals
    assert len(cli.invocations) == 1, "expected exactly one spawn to assert on"
    return cli.invocations[0].argv


def _value_after(argv: tuple[str, ...], flag: str) -> str:
    """The single argument following ``flag``, asserting the flag is present."""
    assert flag in argv, f"{flag} missing from argv: {argv}"
    return argv[argv.index(flag) + 1]


# ─────────────────────────────────────────────────────────────────────────────
# __post_init__: the configurations refused before a subprocess exists
# ─────────────────────────────────────────────────────────────────────────────


def test_strict_mcp_config_on_its_own_is_refused_rather_than_silently_emptying_mcp() -> None:
    """``--strict-mcp-config`` means "ignore every MCP source I did not name".
    Named alongside nothing, that is not a strict configuration, it is an empty
    one: the session starts with no MCP servers at all while the caller believes
    they have just hardened a list they forgot to pass. Dropping this check
    turns a typo into a session whose tools quietly do not exist, and the only
    evidence is a model that claims it cannot find a tool it was promised.
    """
    with pytest.raises(ValueError, match="strict_mcp_config only means something alongside"):
        ClaudeCliCognition(strict_mcp_config=True)

    # The pairing is what makes it legal, so the refusal cannot be a blanket
    # ban on the flag.
    ClaudeCliCognition(strict_mcp_config=True, mcp_config=('{"mcpServers":{}}',))


def test_a_stable_prompt_prefix_under_a_replaced_system_prompt_is_refused_as_a_no_op() -> None:
    """``--exclude-dynamic-system-prompt-sections`` moves the per-machine
    sections OUT OF THE CLI'S OWN default prompt, so it has nothing to act on
    once that prompt has been replaced. Accepting the pair hands back a run
    that looks configured for a cache-stable multi-tenant prefix and is not —
    the caller's reason for setting the flag was a cache hit rate they will
    never see, and nothing in the output says which of their two settings won.
    """
    with pytest.raises(ValueError, match="stable_prompt_prefix has no effect with"):
        ClaudeCliCognition(stable_prompt_prefix=True, system_prompt_mode="replace")

    # Append mode is the combination the flag exists for; it must stay legal.
    ClaudeCliCognition(stable_prompt_prefix=True, system_prompt_mode="append")


def test_an_add_dir_that_is_not_a_directory_is_refused_before_the_cli_dies_on_it(tmp_path: Path) -> None:
    """``--add-dir`` is checked here because the CLI checks it too — three
    seconds into a spawn, on a stderr the caller has to parse, after the run
    has already paid for process start-up. A mistyped path is the single most
    likely thing to be wrong about this field (it is a path typed by a human
    into a config file), and catching it at construction turns a failed run
    into a stack trace that names the offending string.
    """
    missing = tmp_path / "typo-in-the-config"
    with pytest.raises(ValueError, match="add_dirs entries are not directories"):
        ClaudeCliCognition(add_dirs=(missing,))

    # A FILE is not a directory either — ``exists()`` would have accepted it,
    # and the CLI would not.
    not_a_dir = tmp_path / "notes.txt"
    not_a_dir.write_text("")
    with pytest.raises(ValueError, match="add_dirs entries are not directories"):
        ClaudeCliCognition(add_dirs=(not_a_dir,))

    real = tmp_path / "workspace"
    real.mkdir()
    assert ClaudeCliCognition(add_dirs=(real,)).add_dirs == (real,)


def test_a_non_uuid_session_id_is_refused_instead_of_costing_a_spawn_to_discover() -> None:
    """``--session-id`` must be a valid UUID and the CLI only says so once the
    binary has started, which makes an invalid one the most expensive kind of
    typo: a whole process start-up spent learning something a regex knows. It
    is also the field callers reach for when they mean ``resume_session_id=``,
    so the refusal is the only place that correction gets made — a run that
    accepted ``"yesterdays-session"`` would fail with an auth-shaped CLI error
    and teach the caller nothing about the flag they actually wanted.
    """
    with pytest.raises(ValueError, match="session_id must be a valid UUID"):
        ClaudeCliCognition(session_id="yesterdays-session")

    valid = str(uuid.uuid4())
    assert ClaudeCliCognition(session_id=valid).session_id == valid


def test_an_empty_tools_tuple_is_refused_rather_than_emitting_a_bare_tools_flag() -> None:
    """``--tools`` is variadic, so ``tools=()`` compiles to the flag with no
    values after it — an argv the CLI rejects at parse time, from a field that
    reads like "no tools" and could equally mean "don't touch the default set".
    Both readings have a spelling (``tools=('',)`` and ``tools=None``), and
    refusing the ambiguous one is what stops a caller who meant "disable
    everything" from shipping a session that holds every built-in tool.
    """
    with pytest.raises(ValueError, match=r"tools=\(\) is ambiguous"):
        ClaudeCliCognition(tools=())

    # Both disambiguated spellings stay constructible.
    assert ClaudeCliCognition(tools=("",)).tools == ("",)
    assert ClaudeCliCognition(tools=None).tools is None


def test_fork_session_without_anything_to_fork_from_is_refused_at_construction() -> None:
    """``--fork-session`` branches the session being RESUMED; with neither
    ``resume_session_id=`` nor ``continue_session=True`` there is no session to
    branch and the CLI errors on the combination itself. Caught here it names
    the two fields that would make it meaningful, which is the whole difference
    between a one-line fix and reading a usage dump out of a failed run.
    """
    with pytest.raises(ValueError, match="fork_session only applies when resuming"):
        ClaudeCliCognition(fork_session=True)

    # Either resume target makes it meaningful.
    ClaudeCliCognition(fork_session=True, continue_session=True)
    ClaudeCliCognition(fork_session=True, resume_session_id="prior-session")


# ─────────────────────────────────────────────────────────────────────────────
# argv: fields that have to survive the trip to a flag
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_fallback_model_chain_reaches_the_cli_as_one_comma_separated_value() -> None:
    """``--fallback-model`` takes a comma-separated chain tried in order, and a
    tuple is the only way to express more than one. Stringifying the tuple
    instead of joining it produces ``"('a', 'b')"`` — an argument the CLI
    accepts as a MODEL NAME, so the run does not fail at parse time; it fails
    at fallback time, in the minutes after the primary model started erroring,
    which is exactly when nobody is in a position to debug a quoting bug.
    """
    cli = FakeClaudeCli.script(_OK)
    chain = ClaudeCliCognition(
        fallback_model=("claude-opus-4-6", "claude-sonnet-4-6"),
        spawn=cli,
    )
    argv = await _argv(chain, cli)
    assert _value_after(argv, "--fallback-model") == "claude-opus-4-6,claude-sonnet-4-6"

    # A single model is passed through as itself — the same line, the other
    # branch, and the reason the join cannot simply be unconditional.
    solo_cli = FakeClaudeCli.script(_OK)
    solo = ClaudeCliCognition(fallback_model="claude-haiku-4-5", spawn=solo_cli)
    assert _value_after(await _argv(solo, solo_cli), "--fallback-model") == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_an_agent_prompt_is_appended_to_the_clis_system_prompt_not_substituted_for_it() -> None:
    """The CLI's own system prompt is where its tool guidance lives — how to
    use Read, when to use Bash, what the permission prompts mean. ``append`` is
    the default for that reason: an ``Agent(prompt=...)`` states a persona, not
    a replacement for the harness. Compiling it to ``--system-prompt`` anyway
    strips that guidance and degrades tool use across every run, with no error
    and no flag in the output to point at — the model simply gets worse at
    using the tools it still has, which reads as a model problem.
    """
    cli = FakeClaudeCli.script(_OK)
    cognition = ClaudeCliCognition(spawn=cli)
    argv = await _argv(cognition, cli, prompt="You are terse.")
    assert _value_after(argv, "--append-system-prompt") == "You are terse."
    assert "--system-prompt" not in argv

    # And ``replace`` must still reach the replacing flag, or the mode is a
    # field that does nothing in the other direction.
    replacing_cli = FakeClaudeCli.script(_OK)
    replacing = ClaudeCliCognition(system_prompt_mode="replace", spawn=replacing_cli)
    replacing_argv = await _argv(replacing, replacing_cli, prompt="You are terse.")
    assert _value_after(replacing_argv, "--system-prompt") == "You are terse."
    assert "--append-system-prompt" not in replacing_argv


def test_the_system_prompt_file_pair_tracks_the_same_mode_as_the_inline_pair() -> None:
    """A system prompt too large for argv travels as a file instead, which
    doubles the flag pair — and the file variant is the one no eye checks,
    because it only appears above a size threshold no ordinary test reaches.
    Swapping ``--append-system-prompt-file`` for ``--system-prompt-file``
    silently converts "add to Claude Code's prompt" into "replace it" for
    exactly the callers with the biggest prompts, i.e. the ones who most
    depend on the harness guidance they just lost.

    Driven through ``_build_argv`` rather than ``drive`` on purpose: reaching
    the file branch end-to-end means materialising a prompt larger than the
    platform's argv limit, which buys nothing this assertion does not already
    make and costs every run of this file the memory to hold it.
    """
    append = ClaudeCliCognition()._build_argv(
        "", system_prompt="You are terse.", system_prompt_path="/scratch/system-prompt.txt", stream_input=True
    )
    assert _value_after(append, "--append-system-prompt-file") == "/scratch/system-prompt.txt"
    assert "--system-prompt-file" not in append

    replace = ClaudeCliCognition(system_prompt_mode="replace")._build_argv(
        "", system_prompt="You are terse.", system_prompt_path="/scratch/system-prompt.txt", stream_input=True
    )
    assert _value_after(replace, "--system-prompt-file") == "/scratch/system-prompt.txt"
    assert "--append-system-prompt-file" not in replace


@pytest.mark.asyncio
async def test_a_resume_session_id_is_passed_as_resume_and_never_as_session_id() -> None:
    """``--resume <id>`` continues a conversation; ``--session-id <id>`` NAMES a
    fresh one. The two take an identical argument and the CLI accepts either, so
    sending a resume id through the wrong one does not fail — it starts a brand
    new empty session under the id of the conversation the caller wanted back.
    Every turn of context is gone and the run looks entirely healthy, which is
    the failure mode the three separate fields on this class exist to prevent.
    """
    cli = FakeClaudeCli.script(_OK)
    prior = "prior-conversation"
    cognition = ClaudeCliCognition(resume_session_id=prior, spawn=cli)
    argv = await _argv(cognition, cli)
    assert _value_after(argv, "--resume") == prior
    assert "--session-id" not in argv

    # The naming flag still carries ``session_id``, so the two are not merely
    # both routed to ``--resume``.
    named = str(uuid.uuid4())
    named_cli = FakeClaudeCli.script(_OK)
    naming = ClaudeCliCognition(session_id=named, spawn=named_cli)
    named_argv = await _argv(naming, named_cli)
    assert _value_after(named_argv, "--session-id") == named
    assert "--resume" not in named_argv


@pytest.mark.asyncio
async def test_a_restricted_tool_set_reaches_the_cli_as_variadic_tools_entries() -> None:
    """``tools=`` is the only field that RESTRICTS the CLI's built-in set;
    ``allowed_tools``/``disallowed_tools`` govern approval, not availability.
    Losing it means the session keeps every built-in tool — Bash and Write
    included — while the configuration on record says it holds two. That is a
    privilege escalation that produces no error, and because the restriction
    was the control, nothing downstream notices it is gone.

    The flag is variadic, so each name is its own argv entry rather than a
    comma-joined value; ``('',)`` — the CLI's spelling of "no tools" — has to
    survive as a single empty argument, which a join would erase.
    """
    cli = FakeClaudeCli.script(_OK)
    cognition = ClaudeCliCognition(tools=("Read", "Grep"), spawn=cli)
    argv = await _argv(cognition, cli)
    flag = argv.index("--tools")
    assert argv[flag : flag + 3] == ("--tools", "Read", "Grep")

    # The disable-everything spelling keeps its empty argument.
    none_cli = FakeClaudeCli.script(_OK)
    no_tools = ClaudeCliCognition(tools=("",), spawn=none_cli)
    none_argv = await _argv(no_tools, none_cli)
    none_flag = none_argv.index("--tools")
    assert none_argv[none_flag : none_flag + 2] == ("--tools", "")

    # ``tools=None`` must emit nothing, or the default set is restricted by a
    # field the caller never set.
    default_cli = FakeClaudeCli.script(_OK)
    assert "--tools" not in await _argv(ClaudeCliCognition(spawn=default_cli), default_cli)


# ─────────────────────────────────────────────────────────────────────────────
# --bare: the credential the mode stops reading
# ─────────────────────────────────────────────────────────────────────────────


def test_bare_mode_with_no_credential_in_the_child_env_warns_before_the_cli_misdirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--bare`` makes the CLI skip OAuth and the system keychain, so a
    ``claude`` that works perfectly in the developer's terminal fails here with
    "Not logged in · Please run /login" — advice that points at exactly the
    wrong fix, since logging in again changes nothing a bare run reads. The
    warning is the only thing that connects the auth error to the flag that
    caused it; without it the investigation starts at the credential store,
    which is the one place the answer is not.

    Asserted on the environment the CHILD gets, not on ``os.environ``: a
    stripping ``env_policy`` can remove the credential that made the run look
    fine, and checking the wrong dict is how this check passed a run that had
    no credential left.
    """
    for name in {*_CLAUDE_AUTH_ENV, *_BARE_CREDENTIAL_ENV}:
        monkeypatch.delenv(name, raising=False)

    cognition = ClaudeCliCognition(bare=True, env_policy="isolated")
    with pytest.warns(UserWarning, match="no API credential is in the environment"):
        env = cognition._build_env()
    assert not any(env.get(name) for name in _BARE_CREDENTIAL_ENV)

    # A credential present in the child's environment is the whole point of the
    # mode working, so it must not warn.
    supplied = ClaudeCliCognition(bare=True, env_policy="isolated", env={"ANTHROPIC_API_KEY": "synthetic"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert supplied._build_env()["ANTHROPIC_API_KEY"] == "synthetic"

    # Without ``bare=True`` there is nothing to warn about: the CLI reads OAuth
    # and the keychain, and a credential-free environment is the normal case
    # for every developer signed in with a subscription.
    ordinary = ClaudeCliCognition(env_policy="isolated")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ordinary._build_env()


def test_bare_mode_stays_quiet_when_settings_might_carry_an_api_key_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``settings=`` can hold an ``apiKeyHelper``, which supplies the
    credential from outside the environment entirely — so an env-var scan
    cannot tell a misconfigured bare run from a correctly configured one once
    settings are in play. Warning anyway is how a correct configuration gets
    told it is broken, and a warning that fires on working setups is a warning
    people silence; the next genuinely credential-less bare run then says
    nothing at all. The suppression is what keeps the warning worth having.
    """
    for name in {*_CLAUDE_AUTH_ENV, *_BARE_CREDENTIAL_ENV}:
        monkeypatch.delenv(name, raising=False)

    with_settings = ClaudeCliCognition(
        bare=True,
        env_policy="isolated",
        settings='{"apiKeyHelper":"/usr/local/bin/issue-key"}',
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = with_settings._build_env()
    # Nothing was quietly injected to buy the silence — the env really is
    # credential-free and the settings file is the reason we tolerate it.
    assert not any(env.get(name) for name in _BARE_CREDENTIAL_ENV)
