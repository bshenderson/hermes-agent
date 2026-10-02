"""Curated provider identity and label/wire-ID contracts (GXTD-684)."""
from types import SimpleNamespace
from unittest.mock import Mock

from hermes_cli.model_switch_providers import _lap_user_provider_rows, _finalize_picker_rows
from hermes_cli.cli_tui_mixin import CLITuiMixin
from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin


def test_curated_rows_do_not_merge_or_probe_and_empty_capabilities_hide():
    rows = []
    b = SimpleNamespace(seen_slugs=set(), results=rows)
    b.record_section3_pair = Mock(return_value=True)
    b.endpoint_is_current = Mock(return_value=False)
    b.discover_endpoint = Mock(side_effect=AssertionError("curated catalog must not probe"))
    def add(slug, name, url, models, current, empty):
        rows.append(dict(slug=slug, name=name, api_url=url, models=models,
                         is_current=current, total_models=len(models)))
    b.add_endpoint_row = add
    providers = {
        "main": {"api": "http://localhost:8090/v1", "default_model": "hidden",
                 "models": {"hidden": {}}, "picker_models": {"wire-a": {"display_name": "Human A"}}},
        "secondary": {"api": "http://localhost:8090/v1", "picker_models": {"wire-b": {"display_name": "Human B"}}},
        "embed": {"api": "http://localhost:8090/v1", "default_model": "embed-default", "picker_models": {}},
    }
    _lap_user_provider_rows(b, providers)
    assert [r["slug"] for r in rows] == ["main", "secondary"]
    assert rows[0]["models"] == ["wire-a"]
    assert rows[0]["model_labels"] == {"wire-a": "Human A"}
    rows[0]["is_current"] = True
    assert _finalize_picker_rows(rows, providers, "hidden")[0]["models"] == ["wire-a"]
    b.discover_endpoint.assert_not_called()


def test_cli_renders_human_labels_but_selects_wire_id(monkeypatch):
    import hermes_cli.cli_model_switch_mixin as switching
    state = {"stage": "model", "selected": 0, "filter": "",
             "provider_data": {"slug": "main", "name": "Main", "model_labels": {"wire-a": "Human A"}},
             "model_list": ["wire-a"]}
    cli = SimpleNamespace(_model_picker_state=state)
    cli._filter_model_picker_entries = CLIModelSwitchMixin._filter_model_picker_entries
    cli._render_scroll_list_panel = lambda *a, **k: a[3]
    assert CLITuiMixin._get_model_picker_display_fragments(cli)[0] == "Human A"
    selected = []
    monkeypatch.setattr(switching, "_switch_model_from", lambda cli, mid, **kw: selected.append(mid))
    monkeypatch.setattr(switching, "_run_confirm_and_apply", lambda *a, **kw: None)
    cli._close_model_picker = lambda: None
    cli._confirm_and_apply_model_switch_result = lambda *a: None
    CLIModelSwitchMixin._handle_model_picker_selection(cli)
    assert selected == ["wire-a"]
