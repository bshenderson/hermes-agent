"""Operational result projection extends, never replaces, Media transport."""
import json

from gateway.platforms.api_server_tool_results import bounded_tool_result_projection


def test_enrolled_tool_projects_authoritative_data_and_rendered_text():
    value = {'schema': 'homelab.operational-result.v1', 'kind': 'table',
             'title': 'Fixture services', 'complete': True,
             'facts': {'total': 1}, 'columns': ['Service', 'State'],
             'rows': [['fixture-service', 'stopped']]}
    projected = bounded_tool_result_projection('home_inventory', json.dumps({'operational_result': value}))
    assert projected is not None
    assert projected['operational_result'] == value
    assert 'fixture-service' in projected['operational_result_text']
    assert bounded_tool_result_projection('terminal', {'operational_result': value}) is None


def test_media_projection_and_invalid_operational_result_remain_distinct():
    media = bounded_tool_result_projection('media_generate', {'success': True, 'prompt_id': 'fixture'})
    assert media == {'success': True, 'prompt_id': 'fixture'}
    result = bounded_tool_result_projection('home_inventory', {'operational_result': {'complete': False}})
    assert 'operational_result_error' in result
    assert 'operational_result_text' not in result
