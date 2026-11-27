"""Private dashboard failure logs identify the route template, never request secrets."""
import asyncio
import logging
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response


@pytest.mark.parametrize('exception', [False, True])
def test_health_failure_records_only_route_template(monkeypatch, caplog, exception):
    from hermes_cli import web_server as server

    monkeypatch.setattr(server, 'DASHBOARD_HEALTH', server.DashboardHealth())
    request = Request({'type': 'http', 'method': 'GET',
                       'path': '/api/fixture/private-id', 'query_string': b'token=private-query',
                       'headers': [], 'route': SimpleNamespace(path='/api/fixture/{item_id}')})

    async def fail(_request):
        if exception:
            raise ValueError('private-exception-detail')
        return Response(status_code=503)

    with caplog.at_level(logging.WARNING, logger='hermes_cli.web_server'):
        if exception:
            with pytest.raises(ValueError):
                asyncio.run(server._dashboard_health_middleware(request, fail))
        else:
            assert asyncio.run(server._dashboard_health_middleware(request, fail)).status_code == 503
    assert server.DASHBOARD_HEALTH.snapshot()['status'] == 'degraded'
    assert '/api/fixture/{item_id}' in caplog.text
    assert 'private-id' not in caplog.text
    assert 'private-query' not in caplog.text
    assert 'private-exception-detail' not in caplog.text
