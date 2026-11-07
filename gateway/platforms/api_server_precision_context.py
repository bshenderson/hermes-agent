"""Estimated precision sizing before compaction, auxiliary work or generation.

The full-request heuristic is not an exact tokenizer bound. Later injected
context must still fit the selected route and never authorizes route replay.
"""
from copy import deepcopy

from gateway.platforms.api_server_precision_availability import AvailabilityRefused


def estimate_requirement(agent, messages, *, system_prompt='', tools=None):
    from agent.model_metadata import estimate_request_tokens_rough

    reserve = getattr(agent, 'max_tokens', None)
    if type(reserve) is not int or reserve <= 0:
        raise AvailabilityRefused('Precision output allowance is unknown. No model or tools dispatched.')
    return estimate_request_tokens_rough(
        messages, system_prompt=system_prompt, tools=tools,
    ) + reserve


def prepare_requirement(agent, user_message, history):
    from agent.conversation_loop import _restore_or_build_system_prompt
    from agent.turn_context import substitute_api_content

    messages = deepcopy(history or [])
    # Same session prompt/tool restoration as the ordinary turn, without its
    # persistence/lifecycle tail. Ordinary execution still performs that tail.
    try:
        _restore_or_build_system_prompt(agent, None, messages, prepare_only=True)
        for message in messages:
            substitute_api_content(message)
        messages.append({'role': 'user', 'content': deepcopy(user_message)})
        return estimate_requirement(
            agent, deepcopy(getattr(agent, 'prefill_messages', None) or []) + messages,
            system_prompt=agent._cached_system_prompt or '',
            tools=getattr(agent, 'tools', None),
        )
    finally:
        agent._cached_system_prompt = None


def qualify_selection(agent, context, *, profile, model, intent, base_url, api_key, api_mode):
    from gateway.platforms.api_server_precision_availability import probe_local, select_elastic_runtime

    required = prepare_requirement(agent, *context)
    selection = select_elastic_runtime(
        enabled=True, profile=profile, model=model, explicit_selection=False,
        intent=intent, required_context=required,
        probe=lambda candidate: probe_local(base_url, candidate),
    )
    if selection is None:
        raise AvailabilityRefused('Precision context scope unavailable. No model or tools dispatched.')
    if selection['model'] != agent.model:
        tools = deepcopy(getattr(agent, 'tools', None))
        agent.switch_model(new_model=selection['model'], new_provider=agent.provider,
                           api_key=api_key, base_url=base_url, api_mode=api_mode)
        if getattr(agent, 'tools', None) != tools:
            raise AvailabilityRefused('Precision tool surface changed during preparation. No model or tools dispatched.')
        required = prepare_requirement(agent, *context)
    if required > selection['context_length']:
        raise AvailabilityRefused('Precision prepared context exceeds the selected window. No model or tools dispatched.')
    agent._hermes_precision_context_window = selection['context_length']
    agent._hermes_precision_context_guard = guard_prepared_request
    selection['required_context'] = required
    return selection


def guard_prepared_request(agent, messages, tools):
    from agent.turn_loop_errors import PrecisionContextRefused

    window = agent._hermes_precision_context_window
    try:
        required = estimate_requirement(agent, messages, tools=tools)
    except AvailabilityRefused as exc:
        raise PrecisionContextRefused(str(exc)) from exc
    if required > window:
        raise PrecisionContextRefused('Precision prepared request exceeds the selected context window; no route replay.')
    overrides = dict(getattr(agent, 'request_overrides', {}) or {})
    overrides['extra_body'] = {
        **(overrides.get('extra_body') or {}), 'homelab_required_context': required,
    }
    agent.request_overrides = overrides
