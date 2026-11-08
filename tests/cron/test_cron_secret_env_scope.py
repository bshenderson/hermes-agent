"""GXTD-819: per-job ``secret_env`` scoping for cron fires.

Opt-in job field: an authored list scopes which names a fire may resolve from the profile's
external secret sources; ``[]`` hydrates NOTHING external; the field being absent keeps today's
whole-profile hydration; a malformed record or a declared name nothing can supply fails the
fire closed BEFORE any side effect. Scoped resolves are PRIVATE — they never enter the
process-wide per-home caches (``_APPLIED_HOMES`` / ``_SECRET_SOURCE_VALUES_BY_HOME`` /
``_SECRET_SOURCES`` / ``_SOURCE_SUPPLIED_NAMES``), so one job's partial scoped result can
never strand or poison the profile's whole mapping. Synthetic sources only: no real vault is
ever contacted.
"""
from __future__ import annotations

import json
import shutil
import tempfile

import pytest

from agent.secret_sources import registry as reg_module
from agent.secret_sources.base import SECRET_SOURCE_API_VERSION, FetchResult, SecretSource
from cron import jobs as jobs_mod
from hermes_cli import env_loader


@pytest.fixture(autouse=True)
def _isolated_cron_store():
    """Fresh jobs store per test. load_jobs() resolves against a ContextVar (cron/jobs) that
    sibling modules also override; an unpinned run lets a create_job save land in whatever
    store the last importlib.reload left, so records vanish between create and load."""
    home = tempfile.mkdtemp(prefix="gxtd819-scope-store-")
    ctx = jobs_mod.use_cron_store(home)
    ctx.__enter__()
    yield
    ctx.__exit__(None, None, None)
    shutil.rmtree(home, ignore_errors=True)


class _RecordingMappedSource(SecretSource):
    """Mapped fake: fetch() is counted so tests can prove it was NOT invoked."""

    api_version = SECRET_SOURCE_API_VERSION
    name = "fakemap"
    label = "Fake Map"
    shape = "mapped"
    scheme = "fakemap"

    def __init__(self):
        self.fetch_calls = 0

    def fetch(self, cfg, home_path):
        self.fetch_calls += 1
        env_map = (cfg or {}).get("env") or {}
        return FetchResult(secrets={name: f"vault-{name}" for name in env_map})

    def narrow_config(self, cfg, only_names):
        cfg = cfg if isinstance(cfg, dict) else {}
        kept = {n: r for n, r in (cfg.get("env") or {}).items() if n in only_names}
        return None if not kept else {**cfg, "env": kept}


class _RecordingBulkSource(SecretSource):
    """Bulk fake: fetches as a unit (always runs when enabled); apply-time scope trims."""

    api_version = SECRET_SOURCE_API_VERSION
    name = "fakebulk"
    label = "Fake Bulk"
    shape = "bulk"
    scheme = "fakebulk"

    def __init__(self):
        self.fetch_calls = 0

    def fetch(self, cfg, home_path):
        self.fetch_calls += 1
        return FetchResult(secrets={"BULK_ONE": "bulk-one", "BULK_TWO": "bulk-two"})


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Profile home configured with one mapped fake (two bindings) + one bulk fake."""
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


# --- hydration layer ---------------------------------------------------------

def test_scoped_hydration_never_invokes_unrelated_mapped_source(homes):
    home, sources = homes
    resolved = env_loader.resolve_profile_secret_names(home, ["DECLARED_KEY"])
    assert resolved == {"DECLARED_KEY": "vault-DECLARED_KEY"}
    # The mapped source ran ONLY with its narrowed config (declared binding), and the
    # unrelated binding was never fetched as part of this pass.
    assert sources["fakemap"].fetch_calls == 1
    # Names outside the scope were never resolved, applied, or recorded anywhere.
    assert "OTHER_KEY" not in resolved
    assert "BULK_ONE" not in resolved
    assert "OTHER_KEY" not in env_loader._SECRET_SOURCES
    assert "BULK_ONE" not in env_loader._SECRET_SOURCES


def test_scoped_hydration_never_writes_process_caches_including_partial_failure(homes):
    home, _ = homes
    # A scoped resolve that resolves only part of its request (bulk trimming) must not
    # latch the home as applied nor seed the per-home snapshot.
    values = env_loader.resolve_profile_secret_names(home, ["DECLARED_KEY", "BULK_ONE"])
    assert values == {"DECLARED_KEY": "vault-DECLARED_KEY", "BULK_ONE": "bulk-one"}
    home_key = str(home.resolve())
    assert home_key not in env_loader._APPLIED_HOMES
    assert home_key not in env_loader._SECRET_SOURCE_VALUES_BY_HOME
    assert home_key not in env_loader._SECRET_SOURCE_RESTORE_BY_HOME
    assert not env_loader._SECRET_SOURCES


def test_empty_scope_hydration_fetches_nothing(homes):
    home, sources = homes
    assert env_loader.resolve_profile_secret_names(home, []) == {}
    assert sources["fakemap"].fetch_calls == 0
    assert sources["fakebulk"].fetch_calls == 0


def test_unscoped_hydration_unchanged_and_populates_caches(homes):
    """Omitted-scope compatibility: the legacy path still publishes everything it used to."""
    home, _ = homes
    values = env_loader.hydrate_profile_secret_sources(home)
    assert set(values) >= {"DECLARED_KEY", "OTHER_KEY", "BULK_ONE", "BULK_TWO"}
    home_key = str(home.resolve())
    assert home_key in env_loader._APPLIED_HOMES
    assert env_loader.get_secret_source_values(home)["OTHER_KEY"] == "vault-OTHER_KEY"


def test_scoped_fire_does_not_poison_later_unscoped_fire(homes):
    """The anti-poisoning ordering: scoped partial FIRST, unscoped full SECOND still hydrates
    everything — the scoped pass neither latched the home nor shrank its snapshot."""
    home, _ = homes
    assert env_loader.resolve_profile_secret_names(home, ["DECLARED_KEY"]) == {
        "DECLARED_KEY": "vault-DECLARED_KEY"}
    values = env_loader.hydrate_profile_secret_sources(home)
    assert set(values) >= {"DECLARED_KEY", "OTHER_KEY", "BULK_ONE", "BULK_TWO"}


def test_scoped_resolve_never_serves_a_stale_snapshot(homes):
    """Freshness discriminator: after a FULL hydration published a snapshot, removing the
    binding must make the next SCOPED resolve fail — a scoped fire validates against CURRENT
    config, never a cached snapshot the unscoped path would keep serving."""
    home, _ = homes
    assert env_loader.hydrate_profile_secret_sources(home)["OTHER_KEY"] == "vault-OTHER_KEY"
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  sources: [fakemap, fakebulk]\n"
        "  fakemap:\n"
        "    enabled: true\n"
        "    env:\n"
        "      DECLARED_KEY: fakemap://item/declared\n"
        "  fakebulk:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    # The unscoped cache still holds the old value (once-per-home latch is its contract)...
    assert env_loader.get_secret_source_values(home)["OTHER_KEY"] == "vault-OTHER_KEY"
    # ...but a scoped resolve for that name comes back EMPTY — the declaring fire would refuse.
    assert env_loader.resolve_profile_secret_names(home, ["OTHER_KEY"]) == {}


def test_onepassword_narrow_config_contract():
    from agent.secret_sources.onepassword import OnePasswordSource

    source = OnePasswordSource()
    cfg = {"enabled": True, "env": {"A": "op://v/i/a", "B": "op://v/i/b", "bad name": "x"}}
    assert source.narrow_config(cfg, frozenset({"A"}))["env"] == {"A": "op://v/i/a"}
    assert source.narrow_config(cfg, frozenset({"C"})) is None
    # Default hook on a source that cannot narrow: unchanged config (trimmed at apply time).
    other = _RecordingBulkSource()
    assert other.narrow_config({"enabled": True}, frozenset({"X"})) == {"enabled": True}


# --- fire-time resolver ------------------------------------------------------

def test_resolve_job_secret_scope_states():
    from cron.jobs import _SECRET_ENV_UNSET, resolve_job_secret_scope

    assert resolve_job_secret_scope({}) is _SECRET_ENV_UNSET
    assert resolve_job_secret_scope({"secret_env": None}) is _SECRET_ENV_UNSET
    assert resolve_job_secret_scope({"secret_env": []}) == frozenset()
    assert resolve_job_secret_scope({"secret_env": ["A", " A ", "B", "A"]}) == frozenset({"A", "B"})
    with pytest.raises(ValueError):
        resolve_job_secret_scope({"secret_env": "A"})
    with pytest.raises(ValueError):
        resolve_job_secret_scope({"secret_env": ["NOT AN ENV NAME"]})
    with pytest.raises(ValueError):
        resolve_job_secret_scope({"secret_env": [7]})


def test_create_job_secret_env_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from cron import jobs

    plain = jobs.create_job("say hi", "10m")
    assert "secret_env" not in plain
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/fixture.sh").write_text("#!/bin/sh\nprintf 'hi\\n'\n")
    scoped = jobs.create_job("script job", "10m", script="fixture.sh", no_agent=True,
                             secret_env=["DECLARED_KEY", "DECLARED_KEY"])
    assert scoped["secret_env"] == ["DECLARED_KEY"]
    empty = jobs.create_job("none job", "10m", script="fixture.sh", no_agent=True,
                            secret_env=[])
    assert empty["secret_env"] == []
    assert jobs.get_job(empty["id"])["secret_env"] == []
    with pytest.raises(ValueError):
        jobs.create_job("bad job", "10m", secret_env="DECLARED_KEY")
    updated = jobs.update_job(plain["id"], {"secret_env": ["X_ONE"]})
    assert updated["secret_env"] == ["X_ONE"]
    cleared = jobs.update_job(plain["id"], {"secret_env": None})
    # None normalizes to None and reads back as compatible (resolver treats None as unset).
    assert cleared.get("secret_env") is None


# --- fire secret scope install ------------------------------------------------

def test_install_fire_scope_scoped_job_resolves_only_declared(homes, monkeypatch):
    home, sources = homes
    from agent.secret_scope import current_secret_scope, reset_secret_scope
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    tokens = scheduler._install_fire_secret_scope({"id": "j1", "secret_env": ["DECLARED_KEY"]})
    try:
        scope = current_secret_scope()
        assert scope["DECLARED_KEY"] == "vault-DECLARED_KEY"
        assert "OTHER_KEY" not in scope and "BULK_ONE" not in scope
    finally:
        scheduler._reset_fire_secret_scope(tokens)
    assert current_secret_scope() is None
    assert str(home.resolve()) not in env_loader._APPLIED_HOMES
    # The mapped source ran narrowed; the bulk source still fetched as a unit (trimmed).
    assert sources["fakemap"].fetch_calls == 1


def test_install_fire_scope_empty_list_fetches_nothing(homes, monkeypatch):
    home, sources = homes
    from agent.secret_scope import current_secret_scope
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    tokens = scheduler._install_fire_secret_scope({"id": "j1", "secret_env": []})
    try:
        assert sources["fakemap"].fetch_calls == 0
        assert sources["fakebulk"].fetch_calls == 0
        assert current_secret_scope() is not None
    finally:
        scheduler._reset_fire_secret_scope(tokens)


def test_install_fire_scope_unsupplied_name_refuses_before_any_use(homes, monkeypatch):
    home, _ = homes
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    with pytest.raises(scheduler._ScopedSecretRefusal) as excinfo:
        scheduler._install_fire_secret_scope({"id": "j1", "secret_env": ["NOBODY_SUPPLIES"]})
    assert excinfo.value.missing == ["NOBODY_SUPPLIES"]
    # Names only, never values.
    assert "NOBODY_SUPPLIES" in str(excinfo.value)


def test_install_fire_scope_malformed_record_raises_valueerror(homes, monkeypatch):
    home, _ = homes
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    with pytest.raises(ValueError):
        scheduler._install_fire_secret_scope({"id": "j1", "secret_env": "BAD"})


def test_install_fire_scope_dotenv_supplied_name_is_not_a_refusal(homes, monkeypatch):
    home, _sources = homes
    (home / ".env").write_text("MY_DOTENV_KEY=x\n", encoding="utf-8")
    from agent.secret_scope import current_secret_scope
    from cron import scheduler

    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    tokens = scheduler._install_fire_secret_scope(
        {"id": "j1", "secret_env": ["MY_DOTENV_KEY", "DECLARED_KEY"]})
    try:
        assert current_secret_scope()["MY_DOTENV_KEY"] == "x"
        assert current_secret_scope()["DECLARED_KEY"] == "vault-DECLARED_KEY"
    finally:
        scheduler._reset_fire_secret_scope(tokens)
