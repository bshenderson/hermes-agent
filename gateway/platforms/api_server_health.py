"""Authenticated detailed-health payload, including bounded work provenance."""
import os
import time


def detailed_health(adapter):
    from gateway.platforms.api_server import web, _hermes_version, collect_runtime_readiness
    from gateway.status import (
        derive_gateway_busy, derive_gateway_drainable, normalize_updated_at,
        parse_active_agents, read_runtime_status, _build_pid_record)
    from gateway.run_work_sample import project_work_sample
    from gateway.run import _resolve_gateway_model

    runtime = read_runtime_status() or {}
    gw_state = runtime.get('gateway_state')
    gw_active = parse_active_agents(runtime.get('active_agents', 0))
    identity = _build_pid_record()
    work_sample = project_work_sample(runtime.get('work_sample'), pid=identity['pid'],
                                      start_time=identity['start_time'], total=gw_active)
    # Use live adapter metrics even when the persisted heartbeat is delayed.
    platforms = runtime.get('platforms', {})
    platforms = dict(platforms) if isinstance(platforms, dict) else {}
    api_status = adapter._api_server_status_payload(heartbeat_at=time.time())
    api_platform = dict(platforms.get('api_server', {}))
    api_platform.setdefault('state', 'connected' if adapter.is_connected else 'disconnected')
    api_platform['metrics'] = api_status
    platforms['api_server'] = api_platform
    active_api_runs, process_depth, active_delegations = adapter._readiness_work_counts()
    readiness = collect_runtime_readiness(
        configured_model=_resolve_gateway_model(), runtime_status=runtime,
        active_api_runs=active_api_runs, process_completion_queue_depth=process_depth,
        active_delegations=active_delegations)
    return web.json_response({
        'status': readiness['status'], 'readiness': readiness, 'platform': 'hermes-agent',
        'version': _hermes_version(), 'gateway_state': gw_state, 'platforms': platforms,
        'api_server': api_status, 'metrics_today': api_status['metrics_today'],
        'last_heartbeat': api_status['last_heartbeat'], 'active_agents': gw_active,
        'work_sample': work_sample,
        'gateway_busy': derive_gateway_busy(gateway_running=True, gateway_state=gw_state,
                                           active_agents=gw_active),
        'gateway_drainable': derive_gateway_drainable(gateway_running=True, gateway_state=gw_state),
        'exit_reason': runtime.get('exit_reason'),
        'updated_at': normalize_updated_at(runtime.get('updated_at')), 'pid': os.getpid()})
