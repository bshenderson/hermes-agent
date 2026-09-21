"""Preserved GXTD-594 executor ownership metrics."""
import threading
import pytest
from gateway.platforms import api_server_lifecycle_metrics

def module():
    return api_server_lifecycle_metrics

def test_execution_lifetime_follows_thread_and_preserves_result():
    mod = module()
    collector = mod.Collector()
    turn = mod.Turn(collector, 'request')
    entered, release = threading.Event(), threading.Event()
    result = {'completed': True, 'final_response': 'PRIVATE-TEXT'}
    returned = []

    def execute():
        entered.set()
        assert release.wait(5)
        return result

    worker = threading.Thread(target=lambda: returned.append(turn.run(execute)))
    worker.start()
    try:
        assert entered.wait(5)
        text = collector.render()
        assert 'hermes_api_executors_active{family="request"} 1' in text
        assert 'hermes_api_executor_outcomes_total{family="request",outcome="completed"} 0' in text
        assert 'PRIVATE-' not in text
    finally:
        release.set()
        worker.join(5)
    assert returned[0] is result
    text = collector.render()
    assert 'hermes_api_executors_active{family="request"} 0' in text
    assert 'hermes_api_executor_outcomes_total{family="request",outcome="completed"} 1' in text


def test_progress_is_actual_callback_not_keepalive_or_creation(monkeypatch):
    mod = module()
    clock = [100.0]
    monkeypatch.setattr(mod.time, 'time', lambda: clock[0])
    collector = mod.Collector()
    turn = mod.Turn(collector, 'durable_run')
    received = []
    text_cb = turn.wrap_callback('text', lambda text: received.append(text))
    progress_cb = turn.wrap_callback('tool', lambda *args, **kwargs: received.append(args))
    assert turn.wrap_callback('text', None) is None  # Do not turn on streaming.

    def execute():
        text = collector.render()
        assert 'hermes_api_progress_unobserved{family="durable_run"} 1' in text
        text_cb(None)
        text_cb('')
        assert 'hermes_api_progress_unobserved{family="durable_run"} 1' in collector.render()
        clock[0] = 110.0
        text_cb('PRIVATE-TEXT')
        assert 'hermes_api_progress_oldest_timestamp_seconds{family="durable_run"} 110.0' in collector.render()
        clock[0] = 120.0
        progress_cb('tool.started', 'PRIVATE-TOOL', args={'secret': 'PRIVATE-ARGS'})
        assert 'hermes_api_progress_oldest_timestamp_seconds{family="durable_run"} 120.0' in collector.render()
        assert 'PRIVATE-' not in collector.render()
        return {'completed': True}

    turn.run(execute)
    assert received == [None, '', 'PRIVATE-TEXT', ('tool.started', 'PRIVATE-TOOL')]
    assert 'hermes_api_progress_tracked_active{family="durable_run"} 0' in collector.render()


@pytest.mark.parametrize('result,outcome', [
    ({'failed': True}, 'failed'), ({'interrupted': True}, 'interrupted'),
    ({'partial': True}, 'incomplete'), ({'completed': False}, 'incomplete'),
    (None, 'unknown'),
])
def test_return_flags_are_bounded(result, outcome):
    mod = module()
    collector = mod.Collector()
    assert mod.Turn(collector, 'request').run(lambda: result) is result
    assert f'hermes_api_executor_outcomes_total{{family="request",outcome="{outcome}"}} 1' in collector.render()


def test_exception_and_disabled_callbacks_preserve_behavior():
    mod = module()
    collector = mod.Collector()
    error = RuntimeError('PRIVATE-ERROR')
    def fail():
        raise error
    with pytest.raises(RuntimeError) as caught:
        mod.Turn(collector, 'request').run(fail)
    assert caught.value is error
    assert 'outcome="exception"} 1' in collector.render()
    disabled = mod.new_turn(object(), 'request')
    assert disabled.wrap_callback('text', fail) is fail
    with pytest.raises(RuntimeError):
        disabled.run(fail)


def test_unknown_progress_does_not_borrow_another_executors_timestamp():
    mod = module()
    collector = mod.Collector()
    first, second = (mod.Turn(collector, 'request') for _ in range(2))
    one = first.wrap_callback('text', lambda *_: None)
    second.wrap_callback('text', lambda *_: None)
    collector.begin(first)
    collector.begin(second)
    try:
        one('PRIVATE-TEXT')
        text = collector.render()
        assert 'hermes_api_progress_unobserved{family="request"} 1' in text
        assert 'hermes_api_progress_oldest_timestamp_seconds{family="request"}' not in text
    finally:
        collector.finish(first, 'completed')
        collector.finish(second, 'completed')
    one('late callback')
    assert 'hermes_api_progress_tracked_active{family="request"} 0' in collector.render()


def test_concurrent_outcomes_and_callback_exception_are_preserved():
    from concurrent.futures import ThreadPoolExecutor
    mod = module()
    collector = mod.Collector()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: mod.Turn(collector, 'request').run(lambda: {'completed': True}), range(100)))
    assert 'hermes_api_executor_outcomes_total{family="request",outcome="completed"} 100' in collector.render()
    turn = mod.Turn(collector, 'request')
    error = ValueError('PRIVATE-CALLBACK-ERROR')
    def bad_callback(*args, **kwargs):
        raise error
    observed = turn.wrap_callback('text', bad_callback)
    with pytest.raises(ValueError) as caught:
        turn.run(lambda: observed('PRIVATE-TEXT'))
    assert caught.value is error
    assert 'hermes_api_executors_active{family="request"} 0' in collector.render()
    assert 'PRIVATE-' not in collector.render()


