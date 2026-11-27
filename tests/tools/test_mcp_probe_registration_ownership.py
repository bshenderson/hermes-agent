"""A temporary MCP probe must never publish/retire a serving task's registry entries."""
import asyncio
from types import SimpleNamespace


def test_ready_unowned_probe_does_not_retire_serving_tools(monkeypatch):
    from tools import mcp_tool as core, mcp_tool_discovery as discovery
    from tools import mcp_tool_registration as registration
    from tools.registry import registry

    monkeypatch.setattr(core, '_servers', {})
    monkeypatch.setattr(core, '_server_scope_keys', {})
    monkeypatch.setattr(core, '_server_tool_scopes', {})
    name = 'probe-ownership-fixture'
    config = {'url': 'https://fixture.invalid/mcp', 'tools': {'prompts': False, 'resources': False}}
    listed = [SimpleNamespace(name='list_items', description='Read fixture items',
                              inputSchema={'type': 'object', 'properties': {}}, annotations=None)]
    owner = core.MCPServerTask(name)
    owner._config, owner._tools, owner.session = config, listed, object()
    discovery._adopt_server(name, owner)
    owner._registered_tool_names = registration._register_server_tools(name, owner, config)
    tool_name = owner._registered_tool_names[0]
    scope = core._server_registry_scope(registration._server_key_for_task(owner))
    original = registry.snapshot_registration(tool_name, scope=scope)
    assert original is not None
    try:
        probe = core.MCPServerTask(name)
        probe._config, probe._tools, probe.session = config, listed, object()
        probe._ready.set()
        # A tools/list_changed refresh after initialization reaches this production path.
        probe._register_discovered_tools_if_needed()
        asyncio.run(probe.shutdown())
        assert registry.snapshot_registration(tool_name, scope=scope) == original
        assert owner.session is not None
    finally:
        asyncio.run(owner.shutdown())


def test_owned_initial_failure_revival_still_publishes_tools(monkeypatch):
    from tools import mcp_tool as core, mcp_tool_discovery as discovery
    from tools import mcp_tool_registration as registration
    from tools.registry import registry

    monkeypatch.setattr(core, '_servers', {})
    monkeypatch.setattr(core, '_server_scope_keys', {})
    monkeypatch.setattr(core, '_server_tool_scopes', {})
    monkeypatch.setattr(core, '_server_connect_errors', {})
    name = 'revival-ownership-fixture'
    server = core.MCPServerTask(name)
    server._config = {'url': 'https://fixture.invalid/mcp', 'tools': {'prompts': False, 'resources': False}}
    server._tools = [SimpleNamespace(name='list_items', description='Read fixture items',
                                    inputSchema={'type': 'object', 'properties': {}}, annotations=None)]
    server.session = object()
    discovery._adopt_server(name, server)
    key = registration._server_key_for_task(server)
    core._server_connect_errors[key] = 'retained initial failure'
    try:
        server._register_discovered_tools_if_needed()
        assert registry.get_tool_names_for_toolset('mcp-' + name)
        assert key not in core._server_connect_errors
    finally:
        asyncio.run(server.shutdown())
