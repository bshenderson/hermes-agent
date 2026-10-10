"""Classic CLI application construction after the runtime-mixin extraction."""
import pytest
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout import Layout
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.styles import Style
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.widgets import TextArea


@pytest.mark.parametrize("vim_mode", [False, True])
def test_classic_application_constructs_with_selected_editing_mode(monkeypatch, vim_mode):
    import cli
    from hermes_cli import cli_tui_runtime_mixin

    monkeypatch.setattr(cli, "_select_classic_cli_pt_output", lambda stream: DummyOutput())
    monkeypatch.setattr(cli_tui_runtime_mixin.pet_render, "supports_kitty_placeholders", lambda: False)
    instance = object.__new__(cli.HermesCLI)
    instance._vim_mode = vim_mode
    with create_pipe_input() as pipe:
        from prompt_toolkit.application import create_app_session
        with create_app_session(input=pipe, output=DummyOutput()):
            app = instance._tui_build_application(Layout(TextArea()), KeyBindings(), Style([]))
    assert app.editing_mode == (EditingMode.VI if vim_mode else EditingMode.EMACS)
    assert app.erase_when_done is True
