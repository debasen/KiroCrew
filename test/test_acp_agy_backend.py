"""The agy backend: vocabulary, launch record, host auth, and adapter protocol.

Google Antigravity CLI (agy) as an ACP backend in Kiro Crew.
"""

from __future__ import annotations

import pytest

from kiro_crew import acp_backends
from kiro_crew.acp.adapters.agy import AgyAcpServer
from kiro_crew.acp.types import PROVIDER_LABEL_AGY, PROVIDER_LABEL_BY_BACKEND
from kiro_crew.agent_sdk import backends as sdk_backends
from kiro_crew.agent_sdk import host_auth

AGY = acp_backends.ACP_BACKEND_AGY


def test_agy_is_a_known_and_selectable_backend() -> None:
    """Known gates the kwarg; selectable is what the switch may persist."""
    assert AGY == "agy"
    assert AGY in sdk_backends.ACP_BACKENDS_KNOWN
    assert AGY in sdk_backends.BASELINE_SELECTABLE_BACKENDS


def test_agy_provider_label() -> None:
    """Label used in logs, cards and UI."""
    assert PROVIDER_LABEL_AGY == "agy"
    assert PROVIDER_LABEL_BY_BACKEND[AGY] == "agy"


def test_agy_routing() -> None:
    """Routing disposition in ACP_BACKEND_ROUTING."""
    assert sdk_backends.routing_for(AGY) is sdk_backends.Routing.SEEDED_SETTINGS


def test_agy_launch_record() -> None:
    """SelfServedLaunch definition for agy."""
    launch = sdk_backends.launch_for(AGY)
    assert launch.binary == "agy-acp"
    assert launch.bin_env_var == "AGY_ACP_BIN"
    assert launch.spawn_label == "agy-acp"
    assert launch.install_command == "agy-acp"


def test_agy_host_auth_declaration() -> None:
    """Host auth declaration for agy."""
    decl = host_auth.declaration_for(AGY)
    assert decl.backend == "agy"
    assert decl.entitlement_source == host_auth.ENTITLEMENT_OWN_CREDENTIAL_FILE
    assert ".gemini/antigravity-cli/settings.json" in decl.credential_leaves
    assert ".gemini/antigravity-cli/cache/onboarding.json" in decl.credential_leaves
    assert decl.host_logout_retires_children is False


def test_agy_capability_sets() -> None:
    """Verify agy membership across capability sets."""
    assert AGY in sdk_backends.ACP_BACKENDS_LOAD_WITHOUT_MODES
    assert AGY in sdk_backends.ACP_BACKENDS_MODEL_VIA_CONFIG_OPTION
    assert AGY in sdk_backends.ACP_BACKENDS_EFFORT_VIA_CONFIG_OPTION
    assert AGY in sdk_backends.ACP_BACKENDS_HARNESS_OWNED_SESSIONS
    assert AGY in sdk_backends.ACP_BACKENDS_SESSION_MCP_ARRAY
    assert AGY in sdk_backends.ACP_BACKENDS_MEMBER_DISPATCH

    assert AGY in sdk_backends.ACP_BACKENDS_STEER
    assert AGY in sdk_backends.ACP_BACKENDS_COMPACT
    assert AGY in sdk_backends.ACP_BACKENDS_INLINE_COMPACTION
    assert AGY in sdk_backends.ACP_BACKENDS_MARKDOWN_AGENT_SPECS
    assert AGY in sdk_backends.ACP_BACKENDS_SIDE_READONLY

    # Not in unverified runtime sharing or internal sandbox
    assert AGY not in sdk_backends.ACP_BACKENDS_ACP_RUNTIME
    assert AGY not in sdk_backends.ACP_BACKENDS_INTERNAL_SANDBOX
    assert AGY not in sdk_backends.ACP_BACKENDS_SESSION_SHARING
    assert AGY not in sdk_backends.ACP_BACKENDS_SESSION_EVICTION


@pytest.mark.asyncio
async def test_agy_adapter_initialize_and_session_lifecycle() -> None:
    """Test AgyAcpServer ACP JSON-RPC handling."""
    server = AgyAcpServer()
    written_responses: list[tuple[any, dict]] = []
    server._write_response = lambda req_id, res: written_responses.append((req_id, res))

    # initialize
    await server.dispatch_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}}
    )
    assert len(written_responses) == 1
    req_id, init_res = written_responses.pop(0)
    assert req_id == 1
    assert init_res["protocolVersion"] == 1
    assert init_res["agentInfo"]["name"] == "agy"

    # session/set_model
    await server.dispatch_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/set_model",
            "params": {"model": "auto"},
        }
    )
    assert len(written_responses) == 1
    req_id, model_res = written_responses.pop(0)
    assert req_id == 2
    assert model_res == {}
    assert server.default_model == "auto"

    # session/set_config_option
    await server.dispatch_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/set_config_option",
            "params": {"configId": "model", "value": "gemini-2.5-flash"},
        }
    )
    assert len(written_responses) == 1
    req_id, cfg_res = written_responses.pop(0)
    assert req_id == 3
    assert cfg_res == {}
    assert server.default_model == "gemini-2.5-flash"


def test_agy_mirror_and_mcp_projection() -> None:
    """Verify AgyMirror is registered and projects MCP servers."""
    from kiro_crew.providers.mirrors import MIRRORS, PROJECTIONS, ProjectionKind, mirror_for
    from kiro_crew.providers.mirrors.agy import AgyMirror

    assert AGY in MIRRORS
    assert MIRRORS[AGY] is AgyMirror
    mirror = mirror_for(AGY)
    assert isinstance(mirror, AgyMirror)

    proj = PROJECTIONS[AGY]
    assert proj.kind == ProjectionKind.MIRROR

    params = mirror.session_params(
        None,
        stub_elements=[{"name": "stub1", "command": "python3", "args": []}],
    )
    assert "mcpServers" in params
    servers = params["mcpServers"]
    assert any(s.get("name") == "stub1" for s in servers)


def test_agy_setup_mcp_servers(tmp_path: any) -> None:
    """Verify _setup_mcp_servers creates and formats .mcp.json in cwd."""
    import json

    from kiro_crew.acp.adapters.agy import _setup_mcp_servers

    mcp_servers = [
        {
            "name": "srv1",
            "command": "node",
            "args": ["server.js"],
            "env": [{"name": "PORT", "value": "3000"}],
        }
    ]
    mcp_file, orig, added_keys = _setup_mcp_servers(str(tmp_path), mcp_servers)
    assert mcp_file == str(tmp_path / ".mcp.json")
    assert orig is None
    assert isinstance(added_keys, list)

    with open(mcp_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "mcpServers" in data
    assert "srv1" in data["mcpServers"]
    assert data["mcpServers"]["srv1"]["command"] == "node"
    assert data["mcpServers"]["srv1"]["env"] == {"PORT": "3000"}


@pytest.mark.asyncio
async def test_agy_steer_handling() -> None:
    """Verify handle_session_steer emits lifecycle notifications and injects message."""
    from unittest.mock import AsyncMock, MagicMock

    from kiro_crew.acp.adapters.agy import AgyAcpServer, AgySession

    server = AgyAcpServer()
    notifications: list[tuple[str, dict]] = []
    responses: list[tuple[any, dict]] = []
    server._write_notification = lambda method, params: notifications.append((method, params))
    server._write_response = lambda req_id, res: responses.append((req_id, res))

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_stdin = MagicMock()
    mock_stdin.drain = AsyncMock()
    mock_proc.stdin = mock_stdin

    session = AgySession(session_id="test_sess_1", proc=mock_proc, cwd="/tmp")
    server.sessions["test_sess_1"] = session

    await server.dispatch_request(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "_session/steer",
            "params": {"sessionId": "test_sess_1", "message": "focus on tests"},
        }
    )

    assert len(responses) == 1
    assert responses[0] == (42, {"queued": True})

    assert len(notifications) == 2
    assert notifications[0][0] == "session/update"
    assert notifications[0][1]["update"]["sessionUpdate"] == "steering_queued"
    assert notifications[0][1]["update"]["content"] == "focus on tests"

    assert notifications[1][0] == "session/update"
    assert notifications[1][1]["update"]["sessionUpdate"] == "steering_consumed"
    assert notifications[1][1]["update"]["content"] == "focus on tests"

    mock_stdin.write.assert_called_once()
    written_data = mock_stdin.write.call_args[0][0].decode("utf-8")
    assert "focus on tests" in written_data


@pytest.mark.asyncio
async def test_agy_prompt_tool_unwrapping() -> None:
    """Verify call_mcp_tool events are unwrapped to mcp__<server>__<tool>."""
    import json
    from unittest.mock import AsyncMock, MagicMock

    from kiro_crew.acp.adapters.agy import AgyAcpServer, AgySession

    server = AgyAcpServer()
    notifications: list[tuple[str, dict]] = []
    responses: list[tuple[any, dict]] = []
    server._write_notification = lambda method, params: notifications.append((method, params))
    server._write_response = lambda req_id, res: responses.append((req_id, res))

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_stdin = MagicMock()
    mock_stdin.drain = AsyncMock()
    mock_proc.stdin = mock_stdin

    lines = [
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "step_index": 1,
                    "tool_name": "call_mcp_tool",
                    "state": "ACTIVE",
                    "tool_info": {
                        "parameters": {
                            "ServerName": "kirocrew-core",
                            "ToolName": "send_message",
                            "Arguments": {"text": "hello"},
                        }
                    },
                },
            }
        ).encode("utf-8")
        + b"\n",
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "step_index": 1,
                    "state": "DONE",
                    "tool_info": {"output": "ok"},
                },
            }
        ).encode("utf-8")
        + b"\n",
        json.dumps(
            {
                "event": "result",
                "result": {"response": "All done", "usage": {"total_tokens": 150}},
            }
        ).encode("utf-8")
        + b"\n",
    ]

    mock_stdout = MagicMock()
    mock_stdout.readline = AsyncMock(side_effect=lines + [b""])
    mock_proc.stdout = mock_stdout

    session = AgySession(session_id="test_sess_2", proc=mock_proc, cwd="/tmp")
    server.sessions["test_sess_2"] = session

    await server.dispatch_request(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "session/prompt",
            "params": {
                "sessionId": "test_sess_2",
                "prompt": [{"type": "text", "text": "run core tool"}],
            },
        }
    )

    assert len(responses) == 1
    assert responses[0] == (99, {"stopReason": "end_turn"})

    tool_active = [
        n[1]["update"]
        for n in notifications
        if n[1].get("update", {}).get("sessionUpdate") == "tool_call"
    ]
    assert len(tool_active) == 1
    assert tool_active[0]["name"] == "mcp__kirocrew-core__send_message"
    assert tool_active[0]["input"] == {"text": "hello"}

    tool_done = [
        n[1]["update"]
        for n in notifications
        if n[1].get("update", {}).get("sessionUpdate") == "tool_call_update"
    ]
    assert len(tool_done) == 1
    assert tool_done[0]["name"] == "mcp__kirocrew-core__send_message"
