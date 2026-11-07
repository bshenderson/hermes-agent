"""GXTD-819 deep-seam coverage: run-body preflight, external worker payload, delivery reads.

Same synthetic fakes as test_cron_secret_env_scope.py (no real vault is contacted).
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent.secret_sources import registry as reg_module
from agent.secret_sources.base import SECRET_SOURCE_API_VERSION, FetchResult, SecretSource
from cron import jobs as jobs_mod
from hermes_cli import env_loader

from tests.cron.test_cron_secret_env_scope import (
    _RecordingBulkSource,
    _RecordingMappedSource,
)


@pytest.fixture(autouse=True)
def _isolated_cron_store():
    """Fresh jobs store per test. load_jobs() resolves against a ContextVar (cron/jobs) that
    sibling modules also override; an unpinned run lets a create_job save land in whatever
    store the last importlib.reload left, so records vanish between create and load."""
    home = tempfile.mkdtemp(prefix="gxtd819-deep-store-")
    ctx = jobs_mod.use_cron_store(home)
    ctx.__enter__()
    yield
    ctx.__exit__(None, None, None)
    shutil.rmtree(home, ignore_errors=True)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  sources: [fakemap, fakebulk]\n"
        "  fakemap:\n"
        "    enabled: true\n"
        "    env:\n"
        "      DECLARED_KEY: fakemap://item/declared\n"
        "      OTHER_KEY: fakemap://item/other\n"
        "  fakebulk:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    mapped = _RecordingMappedSource()
    bulk = _RecordingBulkSource()
    reg_module._reset_registry_for_tests()
    reg_module.register_source(mapped, builtin=True)
    reg_module.register_source(bulk, builtin=True)
    env_loader.reset_secret_source_cache()
    yield home, {"fakemap": mapped, "fakebulk": bulk}
    reg_module._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()


# --- run body preflight (the shared fail-closed point) ------------------------

@pytest.mark.parametrize('authored', [True, False])
def test_no_agent_dotenv_reload_honors_external_scope(homes, monkeypatch, authored):
    from cron import scheduler

    home, _ = homes
    calls = []
    monkeypatch.setattr(scheduler, '_get_hermes_home', lambda: home)
    monkeypatch.setattr(env_loader, 'load_hermes_dotenv', lambda **kw: calls.append(kw))
    monkeypatch.setattr(scheduler, '_block_and_pause_job', lambda *args: (False, '', '', None))
    job = {'id': 'reload-probe', 'no_agent': True}
    if authored:
        job['secret_env'] = []
    scheduler._run_no_agent_job(job, job['id'], 'reload-probe', None)
    assert len(calls) == 1
    assert calls[0].get('load_external_secrets', True) is (not authored)


def _body_job(secret_env):
    return {"id": "job-pf", "execution_id": "exec-pf", "name": "pf", "script": "echo hi",
            "no_agent": True, "secret_env": secret_env}


def test_body_refuses_before_any_side_effect(homes, monkeypatch, tmp_path):
    home, _sources = homes
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)

    calls = []
    monkeypatch.setattr(scheduler, "claim_dispatch",
                        lambda job_id: calls.append("claim") or True)
    monkeypatch.setattr(scheduler, "_start_owned_run",
                        lambda job, eid: calls.append("identity") or object())
    monkeypatch.setattr(scheduler, "run_job",
                        lambda job, **kw: calls.append("run") or (True, "out", "out", None))
    monkeypatch.setattr(scheduler, "self_removal_delivery_allowed", lambda jid: False)

    recorded = {}

    def _mark(job_id, success, error=None, **kw):
        recorded["job_id"] = job_id
        recorded["success"] = success
        recorded["error"] = error

    def _finish(execution_id, success=None, error=None, delivery_outcome=None):
        recorded["execution"] = (execution_id, success, error, delivery_outcome)

    monkeypatch.setattr(scheduler, "mark_job_run", _mark)
    monkeypatch.setattr(scheduler, "finish_execution", _finish)

    result = scheduler._run_one_job_body(_body_job(["NOBODY_SUPPLIES"]))

    assert result is False
    # NOTHING external ran: no dispatch claim, no identity binding, no user code.
    assert calls == []
    assert recorded["job_id"] == "job-pf" and recorded["success"] is False
    assert "NOBODY_SUPPLIES" in recorded["error"]
    assert recorded["execution"][0] == "exec-pf"
    assert recorded["execution"][1] is False
    assert recorded["execution"][3] == "suppressed"
    # The failure copy carries NAMES only.
    assert "NOBODY_SUPPLIES" in recorded["execution"][2]


def test_body_malformed_record_fails_closed_without_crash_delivery(homes, monkeypatch):
    """A malformed secret_env record is a configuration defect: it surfaces via last_error,
    never as a delivered incident (delivery adapters never even got a scope)."""
    home, _ = homes
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    delivered = []
    monkeypatch.setattr(
        scheduler, "_deliver_crash_failure",
        lambda *a, **k: delivered.append(a) or (None, "suppressed"))
    monkeypatch.setattr(scheduler, "self_removal_delivery_allowed", lambda jid: False)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *a, **k: None)

    assert scheduler._run_one_job_body(_body_job("BAD-STRING")) is False
    assert delivered == []


def test_body_passes_preflight_with_valid_scope_and_runs(homes, monkeypatch):
    """Delivery-semantics preserved for a VALID scoped job: the ordinary path proceeds
    (claim → identity → run) exactly as an unscoped job does."""
    home, _sources = homes
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    calls = []
    monkeypatch.setattr(scheduler, "claim_dispatch",
                        lambda job_id: calls.append("claim") or True)

    scopes_at_run = []

    def _run(job, **kw):
        from agent.secret_scope import current_secret_scope
        scopes_at_run.append(dict(current_secret_scope() or {}))
        return True, "out", "out", None

    monkeypatch.setattr(scheduler, "run_job", _run)
    monkeypatch.setattr(scheduler, "self_removal_delivery_allowed", lambda jid: False)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "finish_execution", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "_save_compose_deliver", lambda *a, **k: None)
    monkeypatch.setattr(scheduler, "_finish_completed_run", lambda *a, **k: True)

    class _Identity:
        pass

    monkeypatch.setattr(scheduler, "_start_owned_run",
                        lambda job, eid: calls.append("identity") or _Identity())
    monkeypatch.setattr(scheduler, "enter_cron_execution", lambda *a, **k: _Identity())
    monkeypatch.setattr(scheduler, "exit_cron_execution", lambda tok: None)
    monkeypatch.setattr("tools.terminal_scope.install_profile_terminal_scope",
                        lambda home: _Identity())
    monkeypatch.setattr("tools.terminal_scope.reset_terminal_scope", lambda tok: None)

    # execution token consumption paths must not blow up
    monkeypatch.setattr(scheduler, "_consume_interrupted_flag", lambda jid, tok: False)

    assert scheduler._run_one_job_body(_body_job(["DECLARED_KEY"])) is True
    assert calls == ["claim", "identity"]
    assert scopes_at_run and scopes_at_run[0]["DECLARED_KEY"] == "vault-DECLARED_KEY"
    # The unscoped full pull never ran: the unrelated binding never entered the fire's scope.
    assert "OTHER_KEY" not in scopes_at_run[0]


# --- external worker payload --------------------------------------------------

def test_worker_payload_honors_job_scope(homes, monkeypatch, tmp_path):
    home, _sources = homes
    from cron import scheduler

    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({
            "job": {"id": "job-1", "execution_id": "exec-1",
                    "secret_env": ["DECLARED_KEY"]},
            "profile_home": str(home), "multiplex_active": True,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cron.executions.adopt_claimed_execution",
        lambda execution_id: {"id": execution_id, "status": "running"},
    )
    scopes = []

    def capture(_job, **_kwargs):
        from agent.secret_scope import current_secret_scope
        scopes.append(dict(current_secret_scope() or {}))
        return True

    monkeypatch.setattr(scheduler, "run_one_job", capture)

    assert scheduler._run_external_worker_payload(payload, tmp_path / "exec-1.ready") is True
    assert scopes and scopes[0].get("DECLARED_KEY") == "vault-DECLARED_KEY"
    # The unrelated mapped binding and the bulk names never entered the worker's scope.
    assert "OTHER_KEY" not in scopes[0]
    assert "BULK_ONE" not in scopes[0]


def test_worker_payload_without_field_is_fully_compatible(homes, monkeypatch, tmp_path):
    """Omitted-field compatibility at the worker seam: full hydration still reaches scope."""
    home, _ = homes
    from cron import scheduler

    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({"job": {"id": "job-1", "execution_id": "exec-1"},
                    "profile_home": str(home), "multiplex_active": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cron.executions.adopt_claimed_execution",
        lambda execution_id: {"id": execution_id, "status": "running"},
    )
    scopes = []

    def capture(_job, **_kwargs):
        from agent.secret_scope import current_secret_scope
        scopes.append(dict(current_secret_scope() or {}))
        return True

    monkeypatch.setattr(scheduler, "run_one_job", capture)

    assert scheduler._run_external_worker_payload(payload, tmp_path / "exec-1.ready") is True
    assert scopes and scopes[0].get("DECLARED_KEY") == "vault-DECLARED_KEY"
    assert scopes[0].get("OTHER_KEY") == "vault-OTHER_KEY"


def test_worker_payload_refusal_defers_to_body_without_dying(homes, monkeypatch, tmp_path):
    """An unsatisfiable scope must not kill the worker before its ack (the parent would only
    see an unacknowledged death): the scope degrades to no external values and run_one_job —
    whose body preflight is the shared fail-closed point — still runs."""
    home, _ = homes
    from cron import scheduler

    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({
            "job": {"id": "job-1", "execution_id": "exec-1",
                    "secret_env": ["NOBODY_SUPPLIES"]},
            "profile_home": str(home), "multiplex_active": True,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "cron.executions.adopt_claimed_execution",
        lambda execution_id: {"id": execution_id, "status": "running"},
    )
    ran = []

    def capture(job, **_kwargs):
        from agent.secret_scope import current_secret_scope
        ran.append(dict(current_secret_scope() or {}))
        return True

    monkeypatch.setattr(scheduler, "run_one_job", capture)

    assert scheduler._run_external_worker_payload(payload, tmp_path / "exec-1.ready") is True
    # Worker survived, acked, and reached run_one_job with a scope carrying NO external
    # source values; the body preflight (covered by the body tests) records the refusal.
    assert ran and "NOBODY_SUPPLIES" not in ran[0]
    assert "DECLARED_KEY" not in ran[0]


# --- delivery read path -------------------------------------------------------

def test_scoped_fire_reads_declared_secret_for_delivery_and_uninstalls_clean(homes, monkeypatch):
    home, _ = homes
    from agent.secret_scope import get_secret, reset_secret_scope
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    tokens = scheduler._install_fire_secret_scope({"id": "j", "secret_env": ["DECLARED_KEY"]})
    try:
        # Delivery adapters resolve through get_secret inside the fire scope.
        assert get_secret("DECLARED_KEY") == "vault-DECLARED_KEY"
    finally:
        scheduler._reset_fire_secret_scope(tokens)
    # After the fire, nothing about this profile is latched or cached from the scoped pass.
    assert str(home.resolve()) not in env_loader._APPLIED_HOMES
    assert "DECLARED_KEY" not in env_loader._SECRET_SOURCES


def test_legacy_hydration_still_publishes_full_mapping_for_any_consumer(homes):
    """Unrelated mapped secret never fetched by a SCOPED fire remains available to a later
    full hydration — the two paths cannot interfere in either direction."""
    home, sources = homes
    env_loader.resolve_profile_secret_names(home, ["DECLARED_KEY"])
    assert sources["fakemap"].fetch_calls == 1
    values = env_loader.hydrate_profile_secret_sources(home)
    assert values["OTHER_KEY"] == "vault-OTHER_KEY"
    assert sources["fakemap"].fetch_calls == 2  # the full pass ran the full map fresh
