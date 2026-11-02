"""Configured exact-tool ceiling for authenticated domain-action API turns.

No domain-specific tools or identities are hardcoded here. Tool/resource policy
remains in the profile plugin; this ceiling removes generic capability bypasses.
Unsigned owner/worker turns retain their incumbent tools.
"""
import copy

from agent.domain_action_context import current, optional_current


def policy_for_current(config):
    context = optional_current()
    if context is None:
        return None
    policy = (config.get('api_server') or {}).get('domain_action_tools')
    if not isinstance(policy, dict) or set(policy) != {'toolsets', 'allowed_tools', 'required_tools'}:
        raise ValueError('authenticated_domain_tool_policy_required')
    for name in policy:
        values = policy[name]
        if (not isinstance(values, list) or not values or len(values) > 64
                or any(not isinstance(v, str) or not v or len(v) > 128 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('invalid_authenticated_domain_tool_policy')
    if not set(policy['required_tools']) <= set(policy['allowed_tools']):
        raise ValueError('required_domain_tool_outside_ceiling')
    return copy.deepcopy(policy)


def constrain_agent(agent, policy):
    if policy is None:
        return
    allowed = frozenset(policy['allowed_tools'])
    tools = [tool for tool in agent.tools or [] if tool['function']['name'] in allowed]
    names = {tool['function']['name'] for tool in tools}
    if not set(policy['required_tools']) <= names and any(
        tool['function']['name'] == 'tool_search' for tool in agent.tools or []
    ):
        # Progressive discovery may defer the required plugin. Materialize only
        # the server-selected toolsets, then apply the same exact ceiling; never
        # expose discovery/dispatch bridges or alter ordinary worker settings.
        from model_tools import get_tool_definitions

        raw = get_tool_definitions(
            enabled_toolsets=policy['toolsets'], quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        tools = [tool for tool in raw if tool['function']['name'] in allowed]
        names = {tool['function']['name'] for tool in tools}
    if not set(policy['required_tools']) <= names:
        raise ValueError('required_authenticated_domain_tool_unavailable')
    # Immutable ceiling plus copied origin survives schema refresh. Dispatch
    # compares the propagated context again; missing/expired context denies.
    agent._domain_action_tool_ceiling = allowed
    agent._domain_action_tool_origin = current()
    agent.tools = tools
    agent.valid_tool_names = names


def dispatch_denial(agent, name):
    ceiling = getattr(agent, '__dict__', {}).get('_domain_action_tool_ceiling')
    if ceiling is None:
        return None
    try:
        context = current()
    except (ValueError, TypeError, KeyError):
        return 'authenticated_domain_tool_context_unavailable'
    if context != getattr(agent, '_domain_action_tool_origin', None) or name not in ceiling:
        return 'tool_outside_authenticated_domain_ceiling'
    return None
