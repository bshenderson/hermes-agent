"""Descriptive labels must never replace routing identities."""
from types import SimpleNamespace

import pytest

from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin


@pytest.mark.parametrize("query", ["", "Human", "wire-b"])
def test_filter_label_and_wire_id_preserves_original_index(query):
    state = {"stage": "model", "selected": 0, "filter": query,
             "provider_data": {"name": "Main", "model_labels": {"wire-b": "Human B"}},
             "model_list": ["wire-a", "wire-b"]}
    cli = SimpleNamespace(_model_picker_state=state)
    cli._filter_model_picker_entries = CLIModelSwitchMixin._filter_model_picker_entries
    cli._render_scroll_list_panel = lambda *a, **k: a[3]
    choices = CLITuiMixin._get_model_picker_display_fragments(cli)
    assert "Human B" in choices
    if query:
        assert state["_filtered_pairs"] == [(1, "Human B")]


def test_moa_is_labeled_as_ensemble_not_one_base_model(monkeypatch):
    from hermes_cli import config, inventory
    monkeypatch.setattr(inventory, "_raw_config_has_enabled_moa_preset", lambda: True)
    monkeypatch.setattr(config, "load_config", lambda: {
        "providers": {"main": {"models": {"route": {"display_name": "Base Model"}}}},
        "moa": {"presets": {"default": {"aggregator": {"provider": "custom:main", "model": "route"},
                  "reference_models": [{"provider": "main", "model": "advisor"}]}}}})
    row = inventory._moa_provider_row()
    assert row["models"] == ["default"]
    assert row["model_labels"]["default"] == "Ensemble: Base Model + 1 advisors (default)"


def test_moa_uses_provider_scoped_alias_presentation(monkeypatch):
    from hermes_cli import config, inventory
    monkeypatch.setattr(inventory, "_raw_config_has_enabled_moa_preset", lambda: True)
    monkeypatch.setattr(config, "load_config", lambda: {
        "providers": {"main": {"model_labels": {"route": "Governed Base"}}},
        "moa": {"presets": {"default": {"aggregator": {"provider": "custom:main", "model": "route"},
                  "reference_models": [{"provider": "main", "model": "advisor"}]}}}})
    row = inventory._moa_provider_row()
    assert row["models"] == ["default"]
    assert "Governed Base" in row["model_labels"]["default"]
