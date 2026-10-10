#!/usr/bin/env python3
"""Hermes Agent CLI — interactive terminal interface (``python cli.py --help`` for usage)."""

# Must be the very first import (UTF-8 stdio on Windows). Missing only mid-``hermes update``.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise  # the bootstrap exists but cannot load: skipping it would skip PM activation

import logging
import os
import functools
# Tests patch shutil/time/datetime through the cli facade; siblings also resolve datetime
# lazily through it — these imports are load-bearing despite being unused in this file.
import shutil
import sys
import re
import atexit
import errno
import time
from collections import deque
from dataclasses import dataclass
from contextlib import contextmanager, suppress
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

os.environ["HERMES_QUIET"] = "1"  # suppress our modules' startup chatter


from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.cli_billing_mixin import CLIBillingMixin
from hermes_cli.cli_loops_mixin import CLILoopsMixin
from hermes_cli.cli_info_mixin import CLIInfoMixin
from hermes_cli.cli_terminal_mixin import CLITerminalMixin
from hermes_cli.cli_modal_mixin import CLIModalMixin
from hermes_cli.cli_stream_mixin import CLIStreamMixin
from hermes_cli.cli_session_mixin import CLISessionMixin
from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
from hermes_cli.cli_voice_mixin import CLIVoiceMixin
from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin
from hermes_cli.cli_tui_mixin import CLITuiMixin
from hermes_cli.cli_process_notifications import CLIProcessNotificationsMixin
from hermes_cli.cli_init_mixin import CLIInitMixin
from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin
# Extracted clusters (mechanical split, #116911); re-exported here so `cli.<name>` stays the seam.
from hermes_cli.cli_shutdown import (
    _CLEANUP_STEPS,
    _arm_exit_watchdog,
    _emit_interrupted_session_end,
    _exit_watchdog_timeout,
    _finalize_single_query,
    _float_env,
    _flush_logging_and_stdio,
    _flush_one_shot_session_store,
    _interrupt_async_delegations,
    _invoke_interrupted_session_end,
    _notify_session_finalize,
    _notify_single_query_session_finalize,
    _oneshot_agent_and_session,
    _should_emit_cleanup_session_finalize,
    _shutdown_agent_memory_provider,
    _shutdown_cached_aux_clients,
    _shutdown_mcp_servers,
    _stop_cli_wake_word,
    _sync_process_session_id,
    _wait_for_oneshot_background_completions,
)
from hermes_cli.cli_auto_maintenance import (
    _run_checkpoint_auto_maintenance,
    _run_state_db_auto_maintenance,
)
from hermes_cli.cli_render import (
    ChatConsole,
    _ACCENT,
    _ACCENT_ANSI_DEFAULT,
    _BOLD,
    _DA1_REPLY_RE,
    _DIM,
    _FALSE_RE,
    _LIGHT_DEFAULT_TERM_PROGRAMS,
    _LIGHT_MODE_REMAP,
    _LIGHT_MODE_REMAP_UPPER,
    _REASONING_TAGS,
    _RST,
    _STREAM_PAD,
    _STREAM_PARTIAL_PREVIEW_LEN,
    _SkinAwareAnsi,
    _TOOL_CALL_TAGS,
    _TRUE_RE,
    _WINDOWS_PATH_WITH_DOT_SEGMENT_RE,
    _accent_hex,
    _add_suspect_rows,
    _append_blank_panel_line,
    _append_panel_line,
    _assistant_content_as_text,
    _assistant_copy_text,
    _b,
    _build_compact_banner,
    _clear_output_history,
    _cli_visible_print,
    _coerce_output_history_limit,
    _cprint,
    _d,
    _detect_light_mode_uncached,
    _heal_cooked_mode_drift,
    _hex_to_ansi,
    _install_skin_light_mode_hook,
    _line_rows,
    _luminance_from_hex,
    _maybe_remap_for_light_mode,
    _output_history_lines,
    _output_history_recording,
    _output_history_rows,
    _output_tail_fitting,
    _painted_columns,
    _PaintedLine,
    _panel_box_width,
    _post_stream_transform_output,
    _prepend_note_to_message,
    _preserve_windows_dot_segments_for_markdown,
    _pt_app_is_running,
    _pt_print_ansi,
    _query_osc11_background,
    _record_output_history,
    _record_output_history_entry,
    _release_paints,
    _render_final_assistant_content,
    _rich_text_from_ansi,
    _set_chrome_floor,
    _strip_markdown_syntax,
    _strip_reasoning_tags,
    _terminal_columns,
    _terminal_reflows,
    _terminal_width_for_streaming,
    _tty_wrap,
    _wrap_panel_text,
    _wrap_panel_text_keep_ws,
)
from hermes_cli.cli_config_load import (
    _AUXILIARY_TASK_ENV,
    _CWD_PLACEHOLDERS,
    _TERMINAL_ENV_MAPPINGS,
    _cli_config_defaults,
    _init_logging_and_display_from_config,
    _load_prefill_messages,
    _merge_file_config,
    _mirror_config_to_env,
    _parse_reasoning_config,
    _parse_service_tier_config,
    _resolve_prefill_messages_file,
    load_cli_config,
)
from hermes_cli.cli_terminal_input import (
    _BACKSLASH_LINE_CONTINUATION_RE,
    _DSR_CPR_ESC_RE,
    _DSR_CPR_VISIBLE_RE,
    _EXTENDED_ENTER_KEYS_SEQ,
    _IMAGE_EXTENSIONS,
    _KITTY_KEYBOARD_PUSH_SEQ,
    _MODIFY_OTHER_KEYS_SEQ,
    _SGR_MOUSE_BARE_RE,
    _SGR_MOUSE_ESC_RE,
    _SGR_MOUSE_VISIBLE_RE,
    _TERMINAL_INPUT_MODE_RESET_SEQ,
    _apply_backslash_line_continuation,
    _apply_bracketed_paste_timeout_patch,
    _bind_prompt_submit_keys,
    _build_cpr_disabled_output,
    _cli_multiline_shortcuts_enabled,
    _collect_query_images,
    _detect_file_drop,
    _disable_prompt_toolkit_cpr_warning,
    _enable_extended_enter_keys,
    _estimate_tui_input_height,
    _file_drop_result,
    _format_image_attachment_badges,
    _hermes_call_output_screen_diff,
    _is_backslash_line_continuation,
    _is_ghostty_terminal,
    _preserve_ctrl_enter_newline,
    _resolve_attachment_path,
    _select_classic_cli_pt_output,
    _should_auto_attach_clipboard_image_on_paste,
    _split_path_input,
    _status_bar_visible_from_display_config,
    _strip_leaked_terminal_responses_with_meta,
    _terminal_may_leak_cpr,
    _terminal_supports_extended_enter_keys,
    _termux_example_image_path,
)
from hermes_cli.cli_single_query import (
    _TERMINAL_PROVIDER_REASONS,
    _TRANSIENT_PROVIDER_REASONS,
    _collect_kanban_task_images,
    _configure_quiet_agent,
    _install_single_query_signal_handlers,
    _int_or,
    _interrupt_agent_for_signal,
    _route_single_query_images,
    _run_kanban_goal_loop_chat,
    _run_kanban_goal_loop_q,
    _run_quiet_single_query,
    _run_single_query_mode,
    _single_query_exit_code,
    _sync_cli_session_id_from_agent,
)

try:
    from prompt_toolkit.patch_stdout import patch_stdout
except ImportError:  # partial/broken prompt_toolkit (#96075); sole use is a `with patch_stdout():`
    from contextlib import nullcontext as patch_stdout
try:
    from prompt_toolkit.enums import EditingMode
except ImportError:  # partial prompt_toolkit stubs in tests
    EditingMode = None
from prompt_toolkit import print_formatted_text as _pt_print
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
try:
    from prompt_toolkit.cursor_shapes import CursorShape
    _STEADY_CURSOR = CursorShape.BLOCK
except (ImportError, AttributeError):
    _STEADY_CURSOR = None

try:
    from hermes_cli import pt_input_extras as _pt_extras

    _pt_extras.install_shift_enter_alias()
    _pt_extras.install_ctrl_enter_alias()
    _pt_extras.install_cmd_backspace_alias()
    _pt_extras.install_modify_other_keys_aliases()
    _pt_extras.install_keypress_data_normalization()
    _pt_extras.install_ignored_terminal_sequences()
    del _pt_extras
except Exception:
    pass
import threading
import queue


def _lazy_shim(module: str, name: str, alias: str | None = None):
    """Import ``module.name`` on first call; keeps heavy imports off startup while ``cli.<name>`` stays patchable."""
    import importlib

    def shim(*args, **kwargs):
        return getattr(importlib.import_module(module), name)(*args, **kwargs)

    shim.__name__ = shim.__qualname__ = alias or name
    return shim


def format_duration_compact(*args, **kwargs):
    seconds = float(args[0] if args else kwargs.get("seconds", 0.0))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        remaining_min = int(minutes % 60)
        return f"{int(hours)}h {remaining_min}m" if remaining_min else f"{int(hours)}h"
    days = hours / 24
    return f"{days:.1f}d"


# model id -> shortest configured alias (process-lifetime cache; config is read once).
_REVERSE_ALIAS_CACHE: dict[str, str] | None = None


def _reverse_alias_for_display(model_name: str) -> str:
    """Shortest alias for ``model_name`` from ``model_aliases:`` or ``model.aliases:``, else ``model_name``."""
    global _REVERSE_ALIAS_CACHE
    if not model_name:
        return model_name
    if _REVERSE_ALIAS_CACHE is None:
        rmap: dict[str, str] = {}

        def _put(m: str, alias: str) -> None:
            if m and (m not in rmap or len(alias) < len(rmap[m])):
                rmap[m] = alias

        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
            ma = cfg.get("model_aliases")
            if isinstance(ma, dict):
                for alias, entry in ma.items():
                    if isinstance(entry, dict):
                        _put(str(entry.get("model", "") or "").strip(), alias)
            mdl = cfg.get("model", {}) or {}
            if isinstance(mdl, dict):
                simple = mdl.get("aliases")
                if isinstance(simple, dict):
                    for alias, val in simple.items():
                        if isinstance(val, str) and val.strip():
                            v = val.strip()
                            _put(v.split("/", 1)[1] if "/" in v else v, alias)
        except Exception:
            pass
        _REVERSE_ALIAS_CACHE = rmap
    return _REVERSE_ALIAS_CACHE.get(model_name, model_name)


def format_token_count_compact(*args, **kwargs):
    value = int(args[0] if args else kwargs.get("value", 0))
    abs_value = abs(value)
    if abs_value < 1_000:
        return str(value)

    sign = "-" if value < 0 else ""
    units = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for threshold, suffix in units:
        if abs_value >= threshold:
            scaled = abs_value / threshold
            text = f"{scaled:.{2 if scaled < 10 else 1 if scaled < 100 else 0}f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"

    return f"{value:,}"


realign_markdown_tables = _lazy_shim("agent.markdown_tables", "realign_markdown_tables")

_COMMAND_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


# ~/.hermes/.env first, project .env as dev fallback; user env files override stale shell exports.
from hermes_constants import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv
from agent.i18n import t as _t

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)


CLI_CONFIG = load_cli_config()


_init_logging_and_display_from_config()

# Neuter AsyncHttpxClientWrapper.__del__ before any AsyncOpenAI client exists: it
# schedules aclose() on the running loop (prompt_toolkit's, during idle), closing
# transports bound to dead worker loops ("Event loop is closed" / "Press ENTER to
# continue..."). A meta_path finder patches ``openai._base_client`` at first import —
# eager import costs ~166ms/30MB cold, and the patch is guaranteed to land before
# instantiation. See ``agent.auxiliary_client.neuter_async_httpx_del``.
try:
    import sys as _httpx_neuter_sys
    import importlib.util as _httpx_neuter_imp_util

    class _AsyncHttpxDelNeuter:
        """Patch ``AsyncHttpxClientWrapper.__del__`` to a no-op when ``openai._base_client`` loads."""

        _armed = True

        def find_spec(self, fullname, path=None, target=None):
            if not self._armed or fullname != "openai._base_client":
                return None
            # Disarm before delegating so the recursive find_spec doesn't loop through us.
            self._armed = False
            try:
                _httpx_neuter_sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = _httpx_neuter_imp_util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            _orig_exec = spec.loader.exec_module

            def _patched_exec(module):
                _orig_exec(module)
                try:
                    cls = getattr(module, "AsyncHttpxClientWrapper", None)
                    if cls is not None:
                        cls.__del__ = lambda self: None  # type: ignore[assignment]
                except Exception:
                    pass

            spec.loader.exec_module = _patched_exec  # type: ignore[method-assign]
            return spec

    _httpx_neuter_sys.meta_path.insert(0, _AsyncHttpxDelNeuter())
except Exception:
    pass


# Agent/tool systems load lazily: bare startup only needs the prompt.
def get_tool_definitions(*args, **kwargs):
    from hermes_cli.mcp_startup import wait_for_mcp_discovery
    from model_tools import get_tool_definitions as _get_tool_definitions

    wait_for_mcp_discovery()
    return _get_tool_definitions(*args, **kwargs)


validate_toolset = _lazy_shim("toolsets", "validate_toolset")


_cleanup_all_terminals = _lazy_shim("tools.terminal_tool", "cleanup_all_environments", "_cleanup_all_terminals")
set_sudo_password_callback = _lazy_shim("tools.terminal_tool", "set_sudo_password_callback")
set_approval_callback = _lazy_shim("tools.terminal_tool", "set_approval_callback")
set_secret_capture_callback = _lazy_shim("tools.skills_tool", "set_secret_capture_callback")
_cleanup_all_browsers = _lazy_shim("tools.browser_tool_lifecycle", "_emergency_cleanup_all_sessions", "_cleanup_all_browsers")

_cleanup_done = False  # _run_cleanup runs exactly once
_cleanup_in_progress = False
_session_end_messages: list[str] = []  # plugin on_session_finalize text awaiting the exit summary
_cli_wake_owner = None
# One-shot finalization runs before process cleanup (plugins see the boundary while the
# agent is attached); atexit cleanup must not finalize those sessions again.
_single_query_finalize_attempted_session_ids: set[str | None] = set()
# /handoff sessions belong to the gateway: finalizing them here would stamp end_reason on
# a row the gateway just reopened, making the handoff leg vanish from history.
# Session IDs that were handed off to the gateway via /handoff. The CLI process exits after a successful
# handoff, but the gateway now owns the session lifecycle — _run_cleanup must NOT call finalize_session on
# these, because doing so sets end_reason on a row the gateway just reopened and is actively writing to
# (#88234). The race made the handoff leg vanish from session history and broke session_search recall for
# the handed-off session.
_handed_off_session_ids: set[str | None] = set()
_active_agent_ref = None  # active AIAgent, for memory-provider shutdown at exit
_deferred_agent_startup_done = False
# Set once the TUI app starts (focus reporting + mouse tracking on); gates the on-exit
# terminal reset so non-TUI one-shot runs never emit codes for modes they never enabled.
_tui_input_modes_active = False


# Set True once the TUI's prompt_toolkit app starts (which enables focus reporting + mouse tracking). Gates
# the on-exit terminal reset so non-TUI one-shot CLI runs — which also register _run_cleanup via atexit —
# don't emit escape codes for modes they never enabled (#36823).
def _mark_tui_input_modes_active() -> None:
    """Record that the TUI app started, so _run_cleanup resets input modes."""
    global _tui_input_modes_active
    _tui_input_modes_active = True


def _prepare_deferred_agent_startup() -> None:
    """Run Termux-deferred agent discovery before the first real agent turn."""
    global _deferred_agent_startup_done
    if _deferred_agent_startup_done:
        return
    if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
        return
    _deferred_agent_startup_done = True
    _accept_hooks = os.environ.get("HERMES_ACCEPT_HOOKS", "").lower() in {"1", "true", "yes", "on"}
    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
    except Exception:
        logger.warning("plugin discovery failed at deferred CLI startup", exc_info=True)
    try:
        from hermes_cli.mcp_startup import start_background_mcp_discovery

        start_background_mcp_discovery(logger=logger, thread_name="termux-cli-mcp-discovery")
    except Exception:
        logger.debug("MCP tool discovery failed at deferred CLI startup", exc_info=True)
    try:
        from agent.shell_hooks import register_from_config
        from agent.outbound_webhooks import register_from_config as register_outbound_webhooks
        from hermes_cli.config import load_config

        _hooks_cfg = load_config()
        register_from_config(_hooks_cfg, accept_hooks=_accept_hooks)
        register_outbound_webhooks(_hooks_cfg)
    except Exception:
        logger.debug("shell-hook registration failed at deferred CLI startup", exc_info=True)


_signal_watchdog_armed = False


def _arm_exit_watchdog_on_shutdown_signal() -> None:
    """Arm the exit backstop the moment a termination signal arrives (idempotent; never raises).

    The graceful unwind has wedge points BEFORE ``_run_cleanup`` arms its own watchdog
    (main thread in a syscall, prompt_toolkit teardown never returning). Leash is 2x
    the cleanup timeout so a progressing cleanup is never cut short. Never arm at
    startup: the timer exits unconditionally.

    SIGTERM/SIGHUP establish unambiguous shutdown intent, but the graceful path from signal →
    ``agent.interrupt()`` → ``app.exit()`` / ``KeyboardInterrupt`` → ``finally`` → ``_run_cleanup`` has
    several wedge points BEFORE ``_run_cleanup`` arms the normal watchdog: a main thread parked in a syscall
    that never observes the unwind, a prompt_toolkit teardown that never returns, or an agent worker
    blocking the ``finally``. When that happens the process has NO backstop and a "dead" CLI lingers
    (observed: ``hermes --tui`` alive ~47 min at 4% CPU after terminal close — the #65998 class).
    """
    global _signal_watchdog_armed
    if _signal_watchdog_armed:
        return
    _signal_watchdog_armed = True
    base = _exit_watchdog_timeout()
    if base <= 0:
        return  # explicitly disabled
    with suppress(Exception):  # never let the backstop break signal handling
        _arm_exit_watchdog(timeout_s=base * 2, from_signal=True)


def _run_cleanup(*, notify_session_finalize: bool = True):
    """Run resource cleanup exactly once."""
    global _cleanup_done, _cleanup_in_progress
    if _cleanup_done:
        return
    _cleanup_done = True
    _cleanup_in_progress = True

    try:
        _arm_exit_watchdog()
        # Reset terminal input modes FIRST: teardown below can take seconds and a later
        # step raising must not skip the reset. No-op unless the TUI ran.
        # See #36823.
        _reset_terminal_input_modes_on_exit()

        for step, swallow in _CLEANUP_STEPS:
            with suppress(swallow):
                globals()[step]()
        if notify_session_finalize:
            cleanup_session_id = _active_agent_ref.session_id if _active_agent_ref else None
            if _should_emit_cleanup_session_finalize(cleanup_session_id):
                # Printed by _print_exit_summary, which clears the screen first.
                _session_end_messages.extend(
                    _notify_session_finalize(session_id=cleanup_session_id, platform="cli", reason="shutdown"))
        try:
            _shutdown_agent_memory_provider(_active_agent_ref)
        except Exception as e:
            logger.warning("CLI cleanup memory shutdown failed: %s", e, exc_info=True)
    finally:
        _cleanup_in_progress = False


def _reset_terminal_input_modes_on_exit() -> None:
    """Disable focus reporting + mouse tracking on TUI exit (best-effort).

    Ctrl+C / SIGTERM / crashes bypass prompt_toolkit's unwind, leaving focus events and
    mouse reports as visible text in the next shell. Writes to stdout when it is the
    terminal, else /dev/tty (the TUI may have run with stdout redirected).

    Called from ``_run_cleanup`` (atexit-registered + invoked on the normal / EOF / interrupt exit paths)
    this covers normal quit, Ctrl+C and SIGTERM/SIGHUP. ``kill -9`` is uncatchable, and the kanban worker's
    ``os._exit(0)`` path bypasses ``atexit``; neither runs this — but both are non-TTY / non-TUI, so there
    is nothing to reset there. See #36823.
    """
    global _tui_input_modes_active
    if not _tui_input_modes_active:
        return
    # Clear first so a re-armed _run_cleanup doesn't re-emit.
    _tui_input_modes_active = False
    try:
        stream = sys.stdout
        if stream is not None and stream.isatty():
            stream.write(_TERMINAL_INPUT_MODE_RESET_SEQ)
            stream.flush()
            return
    except Exception:
        pass
    with suppress(Exception), open("/dev/tty", "w", encoding="ascii") as tty:
        tty.write(_TERMINAL_INPUT_MODE_RESET_SEQ)
        tty.flush()


from hermes_cli.worktree_ops import (
    _git_quiet,
    _git_repo_root,
    _maintain_pack_health,
    _prune_stale_worktrees,
    _repo_is_shallow,
    _setup_worktree,
    _worktree_has_unpushed_commits,
    _worktree_is_dirty,
    release_lsp_clients,
)

# ============================================================================= Git Worktree Isolation
# (#652) =============================================================================
_active_worktree: Optional[dict[str, str]] = None


def _cleanup_worktree(info: dict[str, str] | None = None) -> None:
    """Remove a clean worktree and its branch on exit; preserve recoverable work."""
    global _active_worktree
    info = info or _active_worktree
    if not info:
        return

    wt_path, branch, repo_root = info["path"], info["branch"], info["repo_root"]
    if not Path(wt_path).exists():
        return
    _active_worktree = None

    if _worktree_is_dirty(wt_path, repo_root, timeout=10):
        # Uncommitted work: say so, and don't hint at `remove --force`, which would destroy it.
        _cprint(f"\n\033[33m{_t('cli.worktree.uncommitted_keeping', path=wt_path)}\033[0m")
        return

    if _worktree_has_unpushed_commits(wt_path, timeout=10):
        if _repo_is_shallow(repo_root):
            # Shallow boundary makes the unpushed verdict unreliable; the startup pruner reaps later.
            _cprint(f"\n\033[33m{_t('cli.worktree.shallow_clone_keeping', path=wt_path)}\033[0m")
            print(f"  {_t('cli.worktree.next_session_deepens')}")
        else:
            _cprint(f"\n\033[33m{_t('cli.worktree.unpushed_keeping', path=wt_path)}\033[0m")
            print(f"  {_t('cli.worktree.clean_up_manually', path=wt_path)}")
        return

    # Release the tree's language servers while the path still exists, then unlock so `remove`
    # isn't blocked by the lock placed at creation. Fail-soft.
    release_lsp_clients(wt_path)
    _git_quiet(["worktree", "unlock", wt_path], repo_root, log="git worktree unlock failed (non-fatal)")
    _git_quiet(["worktree", "remove", wt_path, "--force"], repo_root, timeout=15, log="Failed to remove worktree")
    _git_quiet(["branch", "-D", branch], repo_root, log=f"Failed to delete branch {branch}")
    _cprint(f"\033[32m{_t('cli.worktree.cleaned_up', path=wt_path)}\033[0m")


# Light/dark terminal detection (mirrors ui-tui/src/theme.ts detectLightMode()). Priority:
# HERMES_LIGHT/HERMES_TUI_LIGHT env, HERMES_TUI_THEME, HERMES_TUI_BACKGROUND, COLORFGBG
# (bg slot 7/15 = light), OSC 11 query, default dark. Cached so the terminal is queried once.
_LIGHT_MODE_CACHE: bool | None = None


def _detect_light_mode() -> bool:
    global _LIGHT_MODE_CACHE
    if _LIGHT_MODE_CACHE is not None:
        return _LIGHT_MODE_CACHE
    try:
        result = _detect_light_mode_uncached()
    except Exception:
        result = False
    _LIGHT_MODE_CACHE = result
    return result


_install_skin_light_mode_hook()


# Prime the light-mode cache when interactive so OSC 11 happens before prompt_toolkit owns the tty.
with suppress(Exception):
    if sys.stdin.isatty() and sys.stdout.isatty():
        _detect_light_mode()


_OUTPUT_HISTORY_ENABLED = True
_OUTPUT_HISTORY_REPLAYING = False
_OUTPUT_HISTORY_SUPPRESSED = False
_OUTPUT_HISTORY_MAX_LINES = 200
_OUTPUT_HISTORY = deque(maxlen=_OUTPUT_HISTORY_MAX_LINES)


def _configure_output_history(enabled: bool, max_lines=200) -> None:
    """Configure recent CLI output replayed after terminal redraws."""
    global _OUTPUT_HISTORY_ENABLED, _OUTPUT_HISTORY_MAX_LINES, _OUTPUT_HISTORY
    _OUTPUT_HISTORY_ENABLED = bool(enabled)
    _OUTPUT_HISTORY_MAX_LINES = _coerce_output_history_limit(max_lines)
    _OUTPUT_HISTORY = deque(maxlen=_OUTPUT_HISTORY_MAX_LINES)


@contextmanager
def _suspend_output_history():
    global _OUTPUT_HISTORY_SUPPRESSED
    old_value = _OUTPUT_HISTORY_SUPPRESSED
    _OUTPUT_HISTORY_SUPPRESSED = True
    try:
        yield
    finally:
        _OUTPUT_HISTORY_SUPPRESSED = old_value


def _replay_output_history(fit=None, output=None) -> None:
    """Repaint recent output above the prompt after a full screen clear.

    ``fit=(rows, columns, painted, top)`` replays only the newest lines whose wrapped height
    fits ``rows`` (see ``_output_tail_fitting``) — the older ones are still in scrollback
    (#95375) — from screen row ``top`` when known (``_set_chrome_floor``). ``output``: paint
    now, straight to this prompt_toolkit output, where the caller just erased the viewport and
    reset the renderer — ``run_in_terminal`` would first erase below the top row, which
    scroll-on-clear terminals (tmux) take as a clear and copy the blank screen into scrollback.
    """
    global _OUTPUT_HISTORY_REPLAYING
    if not _OUTPUT_HISTORY_ENABLED or not _OUTPUT_HISTORY:
        return
    _OUTPUT_HISTORY_REPLAYING = True
    try:
        rendered_lines = _output_history_lines()
        top = None
        if fit is not None:
            rows, columns, painted, top = fit
            rendered_lines = _output_tail_fitting(rendered_lines, rows, columns, painted)
        if rendered_lines:
            # One payload: per-line pt prints each force a sync redraw (a waterfall of old output).
            if output is None:
                _pt_print(_PT_ANSI("\n".join(rendered_lines)))
            else:
                from prompt_toolkit.renderer import print_formatted_text as _paint_formatted_text
                from prompt_toolkit.styles import Style
                _paint_formatted_text(output, _PT_ANSI("\n".join(rendered_lines) + "\n"), Style([]))
                size = output.get_size()
                if top is not None:  # the chrome's top is now this many rows down
                    top += sum(_line_rows(line, columns) for line in rendered_lines)
                    _set_chrome_floor(max(0, size.rows - top))
                    if size.columns != columns:
                        _add_suspect_rows(top + 1 - size.rows)
            width = _painted_columns() if fit is None else columns
            for line in rendered_lines:  # repainted: they wrap at today's width from now on
                if isinstance(line, _PaintedLine):
                    line.width = width
    except Exception:
        pass
    finally:
        _OUTPUT_HISTORY_REPLAYING = False


_strip_leaked_bracketed_paste_wrappers = _lazy_shim(
    "hermes_cli.input_sanitize", "strip_leaked_bracketed_paste_wrappers", "_strip_leaked_bracketed_paste_wrappers"
)


# OSC sequences (e.g. OSC-8 links): pt's ANSI parser strips the ESC but leaks the payload as text.
_OSC_ESCAPE_RE = re.compile(r"\x1b\][\s\S]*?(?:\x07|\x1b\\)")


def _looks_like_slash_command(text: str) -> bool:
    """``/help`` yes, ``/Users/x/file.md`` no: a command's first word has no further ``/``."""
    if not text or not text.startswith("/"):
        return False
    return "/" not in text.split()[0][1:]


_skill_commands = None
_skill_bundles = None


def _slash_args(cmd: str) -> str:
    """Text after the slash-command word, stripped ("" when absent)."""
    parts = cmd.split(None, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _ensure_skill_commands() -> dict:
    if _skill_commands is not None:
        return _skill_commands
    from agent.skill_commands import get_interactive_skill_commands
    return get_interactive_skill_commands()


def get_skill_commands() -> dict:
    return _ensure_skill_commands()


build_skill_invocation_message = _lazy_shim("agent.skill_commands", "build_skill_invocation_message")
build_preloaded_skills_prompt = _lazy_shim("agent.skill_commands", "build_preloaded_skills_prompt")


def get_skill_bundles() -> dict:
    global _skill_bundles
    if _skill_bundles is None:
        from agent.skill_bundles import get_skill_bundles as _impl

        _skill_bundles = _impl()
    return _skill_bundles


build_bundle_invocation_message = _lazy_shim("agent.skill_bundles", "build_bundle_invocation_message")


def _get_plugin_cmd_handler_names() -> set:
    """Return plugin command names (without slash prefix) for dispatch matching."""
    try:
        from hermes_cli.plugins import get_plugin_commands
        return set(get_plugin_commands().keys())
    except Exception:
        return set()


def _parse_skills_argument(skills: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a CLI skills flag into a deduplicated list of skill identifiers."""
    if not skills:
        return []
    raw_values = [str(item) for item in skills if item is not None] if isinstance(skills, (list, tuple)) else [str(skills)]
    parts = (p.strip() for raw in raw_values for p in raw.split(","))
    return list(dict.fromkeys(p for p in parts if p))


def save_config_value(key_path: str, value: any) -> bool:
    """Persist dot-separated ``key_path`` = value into HERMES_HOME/config.yaml; True on success.

    Never the repo's cli-config.yaml: no config reader loads it, so the value would vanish.
    """
    config_path = get_hermes_home() / 'config.yaml'

    try:
        from hermes_constants import mkdir_under_hermes_home
        mkdir_under_hermes_home(config_path.parent)
        from utils import atomic_roundtrip_yaml_update
        atomic_roundtrip_yaml_update(config_path, key_path, value)
        try:  # owner-only: config files contain API keys
            os.chmod(config_path, 0o600)
        except (OSError, NotImplementedError):
            pass
        return True
    except Exception as e:
        logger.error("Failed to save config: %s", e)
        return False


def _normalize_moa_model(model: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """``moa:<preset>`` -> ``("moa", preset)`` (same routing as ``/moa``); anything else -> ``(None, model)``.

    Returns ``("moa", "<preset>")`` when *model* selects the MoA virtual provider, otherwise ``(None,
    model)`` unchanged. This gives non-interactive ``hermes chat -Q -m moa:<preset>`` the same routing the
    interactive ``/moa`` command and the model picker already use: ``resolve_runtime_provider`` handles
    ``requested_provider == "moa"`` and ``agent_init`` builds the MoAClient off ``provider == "moa"``.
    Without this the raw ``moa:<preset>`` string is sent to the real provider and rejected with a 401/400
    "model not supported" (#56828).
    """
    if isinstance(model, str) and model.strip().lower().startswith("moa:"):
        preset = model.strip().split(":", 1)[1].strip()
        if preset:
            return "moa", preset
    return None, model

_split_model_config_default = _lazy_shim("hermes_cli.config", "split_model_config_default", "_split_model_config_default")


class _VoiceInputMessage:
    """Sentinel for voice-transcribed input so the concise voice prefix never applies to typed text.

    Distinguishes STT output from manually typed text while voice mode is active, so the
    concise-voice-response prefix is applied only to messages that actually came from the microphone
    (#65827).
    """

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text


class _SeededQueryMessage:
    """Sentinel for a ``-q`` prompt seeded into an interactive session; treated LITERALLY (no slash/!/file-drop)."""

    __slots__ = ("images", "text")

    def __init__(self, text: str, images=None):
        self.text = text or ""
        self.images = list(images or [])

    def __str__(self) -> str:
        return self.text


def _should_seed_interactive(query, image, quiet: bool, oneshot: bool) -> bool:
    """``-q`` seeds an interactive session only on a real TTY without ``--oneshot``/``-Q`` (automation answers and exits)."""
    if not (query or image) or oneshot or quiet:
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


@dataclass
class _ChatTurn:
    """Per-turn state shared by the ``chat()`` phases and the agent worker thread.

    ``result`` is written by the worker and read after the join; ``tts_normal_exit`` is
    set only when the TTS worker drained on its own so the last sentence is never cut.
    """

    result: Optional[dict] = None
    mute_notification_reply: bool = False
    use_streaming_tts: bool = False
    box_opened: bool = False
    thinking_started: bool = False
    text_queue: Optional[queue.Queue] = None
    tts_thread: Optional[threading.Thread] = None
    stream_callback: Optional[Any] = None
    stop_event: Optional[threading.Event] = None
    tts_normal_exit: bool = False
    voice_prefix: str = ""
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin


_PASTE_REF_RE = re.compile(r'\[Pasted text #\d+: \d+ lines \u2192 (.+?)\]')


class HermesCLI(CLIInitMixin, CLITuiRuntimeMixin, CLIProcessNotificationsMixin, CLIAgentSetupMixin, CLICommandsMixin, CLIBillingMixin, CLITuiMixin, CLIStatusBarMixin, CLIVoiceMixin, CLIModelSwitchMixin, CLISessionMixin, CLIStreamMixin, CLIModalMixin, CLITerminalMixin, CLIInfoMixin, CLILoopsMixin, CLIChatTurnMixin):
    """Interactive REPL for the Hermes Agent."""

    # Seeded -q first message (see _should_seed_interactive); run() re-creates
    # _pending_input, so it is enqueued only after the fresh queue exists.
    _seeded_first_message: Optional[_SeededQueryMessage] = None
    # Inspection surfaces (banner, /tools, status line) read this on partially built instances too.
    disabled_toolsets: Optional[list[str]] = None

    def __init__(
        self,
        model: str | None = None,
        toolsets: list[str] | None = None,
        provider: str | None = None,
        reasoning: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int | None = None,
        run_budget: float | None = None,
        verbose: Optional[bool] = None,
        compact: bool = False,
        resume: str | None = None,
        checkpoints: bool = False,
        pass_session_id: bool = False,
        ignore_rules: bool = False,
    ):
        """CLI args win over config; ``reasoning`` is per-run only; ``resume`` restores history from SQLite."""
        self._init_display_options(verbose, compact)
        self._init_model_routing(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget,
                                 checkpoints, pass_session_id, ignore_rules)
        self._init_runtime_state(resume)


    def _claim_active_session(self, surface: str = "cli", *, stderr: bool = False) -> bool:
        """Claim a global active-session slot for this CLI process."""
        if self._active_session_lease is not None:
            return True
        try:
            from hermes_cli.active_sessions import format_refusal_stderr, try_acquire_active_session

            lease, message = try_acquire_active_session(
                session_id=self.session_id,
                surface=surface,
                config=self.config,
                # Writer identity: a re-claim by this process replaces its own entry.
                # See #94595.
                metadata={"live_session_id": str(self.session_id)},
            )
        except Exception as exc:
            logger.warning("Failed to claim active session slot: %s", exc)
            return True
        if message:
            print(format_refusal_stderr(message), file=sys.stderr) if stderr else self._console_print(f"[bold red]{message}[/]")
            return False
        self._active_session_lease = lease
        with suppress(Exception):
            atexit.register(self._release_active_session)
        return True

    def _release_active_session(self) -> None:
        lease = getattr(self, "_active_session_lease", None)
        if lease is None:
            return
        try:
            lease.release()
        except Exception:
            logger.debug("Failed to release active session slot", exc_info=True)
        finally:
            self._active_session_lease = None

    _PET_FRAME_INTERVAL = 0.16
    _PET_CFG_INTERVAL = 2.5

    def _install_tool_callbacks(self) -> None:
        """Install tool callbacks that need the live prompt UI."""
        if self._tool_callbacks_installed:
            return
        set_sudo_password_callback(self._sudo_password_callback)
        set_approval_callback(self._approval_callback)
        set_secret_capture_callback(self._secret_capture_callback)
        from agent.vault_backends.unlock import set_code_prompt_callback, set_save_login_prompt_callback, set_unlock_prompt_callback
        set_unlock_prompt_callback(self._vault_unlock_callback)
        set_save_login_prompt_callback(self._vault_save_login_callback)
        set_code_prompt_callback(self._vault_code_callback)
        self._tool_callbacks_installed = True

    def _show_security_advisories(self):
        """Startup banner for unacked security advisories, on stderr (piped stdout stays clean); 24h rate-limited."""
        try:
            from hermes_cli.security_advisories import detect_compromised, startup_banner

            banner = startup_banner(detect_compromised())
            if banner:
                print(banner, file=sys.stderr, flush=True)
        except Exception:
            pass  # never block startup

    def _show_browser_backend_notice(self):
        """Once-per-24h hint when the default Browser Use backend silently fell back to built-in tools."""
        try:
            from tools.browser_use_cli import default_downgrade_notice

            notice = default_downgrade_notice()
            if notice:
                from gateway.warning_notifications import render_notification
                render_notification(lambda: self._console_print(f"[yellow]⚠ {notice}[/yellow]"), platform="cli")
        except Exception:
            logger.debug("browser backend notice failed", exc_info=True)

    def finalize_preloaded_skills(self) -> None:
        """Join the background --skills preload and fold it into the prompt (idempotent).

        Raises ``ValueError`` only when EVERY requested skill was unknown.
        """
        if getattr(self, "_preload_skills_finalized", False):
            return
        thread = getattr(self, "_preload_skills_thread", None)
        if thread is None:
            self._preload_skills_finalized = True
            return
        thread.join(timeout=120)
        self._preload_skills_finalized = True
        err = getattr(self, "_preload_skills_error", None)
        if err is not None:
            raise err
        auto_result = getattr(self, "_auto_load_skills_result", None)
        if auto_result and auto_result[2]:
            logger.warning("skills.auto_load: skill(s) not found or disabled, skipped: %s", ", ".join(auto_result[2]))
        # auto_load names first, then explicit -s names that were not already pinned.
        self.preloaded_skills = list(auto_result[1]) if auto_result else []
        result = getattr(self, "_preload_skills_result", None)
        if not result:
            return
        skills_prompt, loaded_skills, missing_skills = result
        if missing_skills:
            from agent.skill_commands import format_missing_skills
            # A typo'd name must not crash a kanban worker; only a fully-missing set fails loudly.
            if not loaded_skills:
                raise ValueError(format_missing_skills(missing_skills))
            logger.warning("Skipping %s. Continuing with: %s. List available skills with `hermes skills list`.",
                           format_missing_skills(missing_skills), ", ".join(loaded_skills))
        if skills_prompt:
            self.system_prompt = "\n\n".join(p for p in (self.system_prompt, skills_prompt) if p).strip()
        self.preloaded_skills += [name for name in loaded_skills if name not in self.preloaded_skills]

    def _show_tool_availability_warnings(self):
        """Warn about toolsets switched off at startup (missing API keys, unusable terminal backend)."""
        try:
            # Runs on a daemon thread on the snapshot fast path: keep the imports to modules the
            # registry walk already loaded plus the pure notices module (a heavy import here races
            # importlib's module locks against the main thread).
            from model_tools import check_tool_availability
            from hermes_cli.tool_availability_notices import (
                current_terminal_backend, filter_to_enabled_toolsets, tool_availability_warning_lines,
            )
            from tools.terminal_tool import terminal_backend_unavailable_reason
            from toolsets import resolve_toolset

            _, unavailable = check_tool_availability()
            # Only toolsets this CLI session actually has. The selection is usually a composite bundle
            # (``hermes-cli``), so expand it to tool names before matching — a raw name comparison
            # matched nothing on a default install and silently dropped the terminal notice.
            unavailable = filter_to_enabled_toolsets(unavailable, self.enabled_toolsets or [], resolve_toolset)
            lines = tool_availability_warning_lines(
                unavailable, terminal_reason=terminal_backend_unavailable_reason(),
                terminal_backend=current_terminal_backend())
            if lines:
                self._console_print()
                for line in lines:
                    self._console_print(line)
        except Exception:
            pass

    def show_config(self):
        """Display current configuration with kawaii ASCII art."""
        terminal_env = os.getenv("TERMINAL_ENV", "local")
        terminal_cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        terminal_timeout = os.getenv("TERMINAL_TIMEOUT", "60")

        config_path = _hermes_home / 'config.yaml'
        if not config_path.exists():
            config_path = Path(__file__).parent / 'cli-config.yaml'
        config_status = _t("cli.config.loaded") if config_path.exists() else _t("cli.config.not_found")

        # ``api_key`` may be a callable (Entra ID bearer provider): never invoke it. Prefer the
        # LIVE agent's key: the constructor seeds self.api_key from env before provider
        # resolution, so on non-OpenAI providers it can be another vendor's key.
        from agent.azure_identity_adapter import is_token_provider

        display_key = self.api_key
        if self.agent is not None and getattr(self.agent, "api_key", None):
            display_key = self.agent.api_key
        if is_token_provider(display_key):
            api_key_display = _t("cli.config.microsoft_entra_id")
        elif isinstance(display_key, str) and len(display_key) > 12:
            api_key_display = f"{display_key[:8]}...{display_key[-4:]}"
        else:
            api_key_display = _t("cli.config.api_key_not_set")

        title = _t("cli.config.title")
        width = 50
        pad = width - len(title)
        _unset = _t("cli.config.value_not_set")
        ssh_target = (
            f"{os.getenv('TERMINAL_SSH_USER', _unset)}@{os.getenv('TERMINAL_SSH_HOST', _unset)}"
            f":{os.getenv('TERMINAL_SSH_PORT', '22')}"
        ) if terminal_env == "ssh" else None
        # (section key, ((label key, value), ...)); labels are re-padded at print time so
        # translated widths still line up instead of carrying alignment spaces in the catalog.
        sections = (
            ("cli.config.section_model", (
                ("cli.config.label_model", self.model),
                ("cli.config.label_base_url", self.base_url),
                ("cli.config.label_api_key", api_key_display))),
            ("cli.config.section_terminal", (
                ("cli.config.label_environment", terminal_env),
                *((("cli.config.label_ssh_target", ssh_target),) if ssh_target else ()),
                ("cli.config.label_working_dir", terminal_cwd),
                ("cli.config.label_timeout", f"{terminal_timeout}s"),
            )),
            ("cli.config.section_agent", (
                ("cli.config.label_max_turns", self.max_turns),
                ("cli.config.label_toolsets", ", ".join(self.enabled_toolsets) if self.enabled_toolsets else "all"),
                ("cli.config.label_verbose", self.verbose),
            )),
            ("cli.config.section_session", (
                ("cli.config.label_started", self.session_start.strftime("%Y-%m-%d %H:%M:%S")),
                ("cli.config.label_config_file", f"{config_path} {config_status}"),
            )),
        )
        print()
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
        for section_key, rows in sections:
            print()
            print(f"  -- {_t(section_key)} --")
            label_width = max(len(_t(label_key)) for label_key, _ in rows)
            for label_key, value in rows:
                print(f"  {_t(label_key):<{label_width}} {value}")
        print()

    # canonical command -> (method name, pass cmd_original?). Absent commands resolve to
    # ``_handle_<name>_command(cmd)``. Looked up via getattr at dispatch time so
    # monkeypatching works. A handler returning False exits the REPL.
    _SLASH_DISPATCH: dict[str, tuple[str, bool]] = {
        "exit": ("_cmd_exit", True), "quit": ("_cmd_exit", True), "help": ("_cmd_help", True),
        "palette": ("_open_command_palette", False), "whoami": ("_handle_whoami_command", False),
        "profile": ("_handle_profile_command", False), "toolsets": ("show_toolsets", False),
        "config": ("show_config", False), "redraw": ("_cmd_redraw", True), "clear": ("_cmd_clear", True),
        "history": ("show_history", False), "title": ("_cmd_title", True), "new": ("_cmd_new", True),
        "model": ("_handle_model_switch", True), "codex-runtime": ("_handle_codex_runtime", True),
        "retry": ("_cmd_retry", True), "prompt": ("_handle_prompt_compose_command", True),
        "undo": ("_cmd_undo", True), "save": ("save_conversation", True), "skills": ("_cmd_skills", True),
        "platforms": ("_show_gateway_status", False), "status": ("_show_session_status", False),
        "context": ("_show_context_breakdown", True), "egress": ("_cmd_egress", True),
        "statusbar": ("_cmd_statusbar", True), "verbose": ("_toggle_verbose", False), "yolo": ("_toggle_yolo", False),
        "compress": ("_manual_compress", True), "subscription": ("_show_subscription", False),
        "topup": ("_show_billing", True), "insights": ("_show_insights", True), "update": ("_cmd_update", True),
        "version": ("_cmd_version", True), "paste": ("_handle_paste_command", False), "reload": ("_cmd_reload", True),
        "reload-mcp": ("_confirm_and_reload_mcp", True), "reload-skills": ("_cmd_reload_skills", True),
        "plugins": ("_cmd_plugins", True), "stop": ("_handle_stop_command", False),
        "agents": ("_handle_agents_command", False), "bg": ("_handle_background_command", True),
        "queue": ("_cmd_queue", True), "steer": ("_cmd_steer", True), "moa": ("_cmd_moa", True),
    }

    @classmethod
    def _slash_handler(cls, canonical: str) -> tuple[str, bool] | None:
        """(method name, pass cmd_original?) for a registered command, else None."""
        entry = cls._SLASH_DISPATCH.get(canonical)
        if entry is None:
            name = f"_handle_{canonical.replace('-', '_')}_command"
            if callable(getattr(cls, name, None)):
                entry = (name, True)
        return entry

    # Shared-metrics surface for user-typed commands; None where another process owns the count
    # (the TUI slash worker: tui_gateway records the command it forwards).
    _slash_metrics_surface: str | None = "cli"

    def process_command(self, command: str, *, redispatch: bool = False) -> bool:
        """Dispatch a slash command; returns False to exit the REPL. ``redispatch`` marks an internal
        re-entry (quick-command alias, prefix expansion) so the user's command is counted once."""
        cmd_lower = command.lower().strip()  # lowercase only for matching; args keep their case
        cmd_original = command.strip()

        # Aliases resolve via the central registry (hermes_cli/commands.py).
        from hermes_cli.commands import resolve_command as _resolve_cmd
        _base_word = cmd_lower.split()[0].lstrip("/")
        _cmd_def = _resolve_cmd(_base_word)
        canonical = _cmd_def.name if _cmd_def else _base_word
        if not redispatch and self._slash_metrics_surface:
            from hermes_cli.observability.shared_metrics_events import record_slash_command
            record_slash_command(command=canonical, surface=self._slash_metrics_surface)

        # Observer-only pre_command plugin hook (return values ignored; never raises).
        if _cmd_def is not None:
            from hermes_cli.plugins import fire_pre_command_hook
            fire_pre_command_hook(
                surface="cli", command=canonical, alias_used=_base_word, args_raw=_slash_args(cmd_original),
                session_key=getattr(self, "session_id", None), platform="cli",
            )

        # A bare `/resume` prompt is one-shot: any other command disarms it so a later
        # number isn't swallowed as a stale selection.
        # See #34584.
        if canonical not in {"resume", "sessions"}:
            # Armed when a bare `/resume` prints the recent-sessions list so the very next bare numeric
            # input (e.g. `3`) resolves to that session. Holds the exact list used for index resolution;
            # one-shot (cleared on the next submitted input, whether it's the selection or anything else).
            # See #34584.
            self._pending_resume_sessions = None

        entry = self._slash_handler(canonical)
        if entry is None:
            return self._process_unregistered_slash(cmd_original, cmd_lower)
        method_name, pass_arg = entry
        handler = getattr(self, method_name)
        result = handler(cmd_original) if pass_arg else handler()
        return result is not False

    def _process_unregistered_slash(self, cmd_original: str, cmd_lower: str) -> bool:
        """Slash input with no built-in handler; precedence: quick_commands -> plugins -> bundles -> skills -> prefix expansion."""
        base_cmd = cmd_lower.split()[0]
        bare = base_cmd.lstrip("/")
        skill_commands = _ensure_skill_commands()
        skill_bundles = get_skill_bundles()
        quick_commands = self.config.get("quick_commands", {})
        user_args = cmd_original[len(base_cmd):].strip()
        if bare in quick_commands:
            return self._run_quick_command(base_cmd, quick_commands[bare], user_args)
        if bare in _get_plugin_cmd_handler_names():
            self._run_plugin_slash_command(base_cmd, user_args)
        elif base_cmd in skill_bundles:
            self._run_skill_bundle_command(base_cmd, skill_bundles[base_cmd], user_args)
        elif base_cmd in skill_commands:
            self._run_skill_slash_command(base_cmd, skill_commands[base_cmd], user_args)
        else:
            return self._expand_slash_prefix(cmd_original, cmd_lower, skill_commands, skill_bundles)
        return True

    def _run_quick_command(self, base_cmd: str, qcmd: dict, user_args: str) -> bool:
        """User-defined quick command (config.yaml): ``exec`` runs a shell snippet, ``alias`` re-dispatches."""
        qtype = qcmd.get("type")
        if qtype == "alias":
            target = qcmd.get("target", "").strip()
            if target:
                target = target if target.startswith("/") else f"/{target}"
                return self.process_command(f"{target} {user_args}".strip(), redispatch=True)
            self._console_print(f"[bold red]{_t('cli.quick.no_target', command=base_cmd)}[/]")
            return True
        if qtype != "exec":
            self._console_print(f"[bold red]{_t('cli.quick.unsupported_type', command=base_cmd)}[/]")
            return True
        import subprocess
        exec_cmd = qcmd.get("command", "")
        if not exec_cmd:
            self._console_print(f"[bold red]{_t('cli.quick.no_command', command=base_cmd)}[/]")
            return True
        try:
            # shell=True is intentional (user-authored config snippets, never LLM controlled);
            # the env is sanitized because this process holds every API key.
            from tools.environments.local import build_subprocess_env
            from hermes_cli._subprocess_compat import windows_hide_flags
            result = subprocess.run(
                exec_cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=30, env=build_subprocess_env(),
                creationflags=windows_hide_flags(),  # no console flash on Windows (#56747)
            )
            # See #56747.
            output = result.stdout.strip() or result.stderr.strip()
            if output:
                from agent.redact import redact_sensitive_text
                self._console_print(_rich_text_from_ansi(redact_sensitive_text(output)))
            else:
                self._console_print(f"[dim]{_t('cli.quick.no_output')}[/]")
        except subprocess.TimeoutExpired:
            self._console_print(f"[bold red]{_t('cli.quick.timed_out')}[/]")
        except Exception as e:
            self._console_print(f"[bold red]{_t('cli.quick.error', error=str(e))}[/]")
        return True

    def _run_plugin_slash_command(self, base_cmd: str, user_args: str) -> None:
        from hermes_cli.plugins import get_plugin_command_handler, resolve_plugin_command_result

        plugin_handler = get_plugin_command_handler(base_cmd.lstrip("/"))
        if not plugin_handler:
            return
        try:
            result = resolve_plugin_command_result(plugin_handler(user_args))
            if result:
                _cprint(str(result))
        except Exception as e:
            _cprint(f"\033[1;31m{_t('cli.plugin.command_error', error=str(e))}{_RST}")

    def _queue_skill_message(self, msg) -> None:
        if hasattr(self, '_pending_input'):
            self._pending_input.put(msg)

    def _run_skill_bundle_command(self, base_cmd: str, bundle_info: dict, user_instruction: str) -> None:
        """``/<bundle>`` loads several skills at once (bundles win over same-named skills)."""
        bundle_result = build_bundle_invocation_message(base_cmd, user_instruction, task_id=self.session_id)
        if not bundle_result:
            ChatConsole().print(f"[bold red]{_t('cli.skills.bundle_load_failed', command=base_cmd)}[/]")
            return
        msg, loaded_names, missing = bundle_result
        self._queue_loaded_skills(
            msg, _t("cli.skills.loading_bundle", name=bundle_info['name'], count=str(len(loaded_names))), missing)

    def _queue_loaded_skills(self, msg, label: str, missing) -> None:
        print(f"\n⚡ {label}")
        if missing:
            ChatConsole().print(f"[yellow]{_t('cli.skills.skipped_missing', names=', '.join(missing))}[/]")
        self._queue_skill_message(msg)

    def _run_skill_slash_command(self, base_cmd: str, skill_info: dict, rest: str) -> None:
        """``/<skill> ...``; stacked ``/skill-a /skill-b do XYZ`` loads every leading skill (up to 5)."""
        from agent.skill_commands import build_stacked_skill_invocation_message, split_stacked_skill_commands

        # Interactive surface: stacked tokens resolve against the interactive
        # map so plugin skills stack in the CLI like native skills.
        extra_keys, user_instruction = split_stacked_skill_commands(rest, interactive=True)
        if extra_keys:
            stacked_result = build_stacked_skill_invocation_message(
                [base_cmd, *extra_keys], user_instruction, task_id=self.session_id,
            )
            if not stacked_result:
                ChatConsole().print(f"[bold red]{_t('cli.skills.stacked_load_failed', command=base_cmd)}[/]")
                return
            msg, loaded_names, missing = stacked_result
            self._queue_loaded_skills(
                msg, _t("cli.skills.loading_stacked", count=str(len(loaded_names)), names=', '.join(loaded_names)),
                missing,
            )
            return
        msg = build_skill_invocation_message(base_cmd, rest, task_id=self.session_id)
        if msg:
            self._queue_loaded_skills(msg, _t("cli.skills.loading_skill", name=skill_info['name']), None)
        else:
            ChatConsole().print(f"[bold red]{_t('cli.skills.skill_load_failed', command=base_cmd)}[/]")

    def _expand_slash_prefix(self, cmd_original: str, cmd_lower: str, skill_commands, skill_bundles) -> bool:
        """Unique-prefix expansion against built-in COMMANDS + skill commands/bundles (agrees with tab-completion)."""
        from hermes_cli.commands import COMMANDS
        typed_base = cmd_lower.split()[0]
        all_known = set(COMMANDS) | set(skill_commands) | set(skill_bundles)
        matches = [c for c in all_known if c.startswith(typed_base)]
        if len(matches) > 1:
            if typed_base in matches:
                matches = [typed_base]
            else:
                # Unique shortest match wins: /qui -> /quit (5) over /quint-pipeline (15)
                min_len = min(len(c) for c in matches)
                shortest = [c for c in matches if len(c) == min_len]
                if len(shortest) == 1:
                    matches = shortest
        if len(matches) == 1 and matches[0] != typed_base:
            # Expand to the full name, preserving arguments.
            return self.process_command(matches[0] + cmd_original.strip()[len(typed_base):], redispatch=True)
        if len(matches) > 1:
            _cprint(f"{_ACCENT}{_t('cli.command.ambiguous', command=cmd_lower)}{_RST}")
            _cprint(f"{_DIM}{_t('cli.command.did_you_mean_many', candidates=', '.join(sorted(matches)))}{_RST}")
        else:
            # Exact token with no handler (never re-dispatch the same token: recursion), or no match.
            from hermes_cli.cli_unknown_command import unknown_command_lines
            lead, pointer = unknown_command_lines(cmd_lower, all_known)
            _cprint(f"\033[1;31m{lead}{_RST}")
            _cprint(f"{_DIM}{_ACCENT}{pointer}{_RST}")
        return True

    def _drain_interrupt_queue_to_pending_input(self) -> None:
        """Move stray ``_interrupt_queue`` messages into ``_pending_input`` after every turn.

        Busy-time input lands in ``_interrupt_queue`` and is only drained by the explicit
        interrupt path; a turn that finishes naturally would otherwise strand it and the
        CLI appears to hang. Never raises.

        Called once at the end of every turn from ``process_loop``'s ``finally`` block. Catches and swallows
        ``Exception`` because the drain must never break the main loop. (#20271)
        """
        try:
            while not self._interrupt_queue.empty():
                stray = self._interrupt_queue.get_nowait()
                if stray:
                    self._pending_input.put(stray)
        except Exception:
            pass

    def _on_reasoning(self, reasoning_text: str):
        """Callback for intermediate reasoning display during tool-call loops."""
        if not reasoning_text:
            return
        self._reasoning_preview_buf = getattr(self, "_reasoning_preview_buf", "") + reasoning_text
        self._flush_reasoning_preview(force=False)

    # Inline tokens that bypass the destructive-slash confirmation modal (scripting, or
    # when the modal can't be marshaled onto the app loop).
    # A general escape hatch for non-interactive use (scripting/automation) and for the degraded path where
    # the modal can't be marshaled onto the app loop — lets users self-serve without flipping
    # approvals.destructive_slash_confirm in config. (Native Windows now drives the modal normally — see
    # #33961.)
    _DESTRUCTIVE_SKIP_TOKENS = frozenset({"now", "--yes", "-y"})

    def _tui_process_loop(self):
        """REPL worker thread: drain ``_pending_input``, run idle housekeeping, dispatch each input."""
        while not self._should_exit:
            try:
                try:
                    user_input = self._pending_input.get(timeout=0.1)
                except queue.Empty:
                    if not self._agent_running:
                        self._tui_idle_tick()
                    continue
                self._tui_process_one_input(user_input)
            except Exception as e:
                if isinstance(e, OSError) and e.errno == errno.EIO:
                    self._mark_terminal_io_broken("process_loop")
                    logger.warning("process_loop EIO — freezing UI paints (#81521): %s", e)
                    continue
                logger.warning("process_loop unhandled error (msg may be lost): %s", e)

    def _tui_idle_tick(self):
        """Idle housekeeping between inputs (agent not running)."""
        self._check_config_mcp_changes()  # auto-reload MCP on mcp_servers change
        # Termios drift heal first: a drifted tty makes the CLI look dead while the loop is healthy.
        for step in (
            self._check_termios_drift,
            lambda: self._drain_process_notifications("cli-idle"),
            self._maybe_fire_loop_tick,
        ):
            with suppress(Exception):
                step()

    def _tui_unwrap_input(self, user_input):
        """Unwrap ``_VoiceInputMessage`` / ``_SeededQueryMessage`` -> ``(text_or_tuple, is_voice_input, is_seeded_query)``."""
        # Voice-transcribed messages arrive wrapped in a sentinel so only genuine STT output gets the voice
        # prefix (#65827).
        is_voice_input = isinstance(user_input, _VoiceInputMessage)
        if is_voice_input:
            user_input = user_input.text
        is_seeded_query = isinstance(user_input, _SeededQueryMessage)
        if is_seeded_query:
            user_input = (user_input.text, user_input.images) if user_input.images else user_input.text
        return user_input, is_voice_input, is_seeded_query

    def _tui_process_one_input(self, user_input):
        """Route one submitted input: file drop, /resume pick, ! shell, slash command, or a chat turn."""
        user_input, is_voice_input, is_seeded_query = self._tui_unwrap_input(user_input)
        if not user_input:
            return
        self._status_bar_suppressed_after_resize = False  # input ends post-resize suppression

        submit_images = []
        if isinstance(user_input, tuple):
            user_input, submit_images = user_input

        if isinstance(user_input, str):
            user_input = _strip_leaked_bracketed_paste_wrappers(user_input)
            user_input, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(user_input)
            if _had_mouse_reports:
                self._recover_terminal_input_modes(reason="mouse reports leaked into submitted input")

        # A typed bare stop phrase ends an active voice chat (transcripts are checked earlier).
        if not is_voice_input and self._typed_voice_stop(user_input):
            return

        # File drops are detected before any dispatch; seeded -q prompts are literal text.
        _file_drop = _detect_file_drop(user_input) if isinstance(user_input, str) and not is_seeded_query else None
        if _file_drop:
            _drop_path = _file_drop["path"]
            _remainder = _file_drop["remainder"]
            if _file_drop["is_image"]:
                submit_images.append(_drop_path)
                user_input = _remainder or f"[User attached image: {_drop_path.name}]"
                _cprint(f"  📎 Auto-attached image: {_drop_path.name}")
            else:
                _cprint(f"  📄 Detected file: {_drop_path.name}")
                user_input = f"[User attached file: {_drop_path}]" + (f"\n{_remainder}" if _remainder else "")
        elif isinstance(user_input, str):
            # A bare number right after a bare `/resume` selects that session (never sent to the agent).
            if self._pending_resume_sessions and self._consume_pending_resume_selection(user_input):
                return
            if not is_seeded_query:
                if self.handle_bang_shell(user_input):
                    return
                if _looks_like_slash_command(user_input):
                    user_input = self._tui_run_slash_input(user_input)
                    if user_input is None:
                        return

        if isinstance(user_input, str) and _PASTE_REF_RE.search(user_input):
            user_input = self._expand_paste_references(user_input)
        print()
        self._print_user_message_preview(user_input)

        if submit_images:
            n = len(submit_images)
            _cprint(f"  {_DIM}📎 {n} image{'s' if n > 1 else ''} attached{_RST}")

        self._agent_running = self._interactive_turn = True
        self._pet_turn_error = self._pet_reasoning = False
        self._turn_summary_begin()
        self._app.invalidate()
        try:
            self.chat(user_input, images=submit_images or None, voice_input=is_voice_input)
        finally:
            self._tui_after_turn()

    def _tui_run_slash_input(self, user_input: str):
        """Dispatch a slash command. Returns the pending agent seed to run as a chat turn, else None."""
        _cprint(f"\n⚙️  {user_input}")
        try:
            if not self.process_command(user_input):
                self._should_exit = True
                if self._app.is_running:
                    self._app.exit()
        except KeyboardInterrupt:
            # Ctrl+C during a slow slash command returns to the prompt instead of exiting.
            _cprint("\n[dim]Command interrupted.[/dim]")
            return None
        _seed, self._pending_agent_seed = self._pending_agent_seed, None
        return _seed or None

    def _tui_after_turn(self):
        """Post-turn bookkeeping after chat() returns (normal, error, or interrupt)."""
        self._agent_running = self._pet_reasoning = False
        self._spinner_text = self._last_scrollback_tool = ""
        self._tool_start_time = 0.0
        self._pending_tool_info.clear()
        self._pet_react_turn_end()
        self._turn_summary_emit()
        self._interactive_turn = False
        self._app.invalidate()

        # After an interrupt the renderer may have drifted (leaked CPR text, VT100 parser
        # stalled mid-escape): drain stray bytes and force a clean redraw.
        if self._last_turn_interrupted:
            self._recover_terminal_after_interrupt()

        # Re-queue any messages that arrived in _interrupt_queue while the agent was running and were never
        # claimed by the explicit interrupt path. See _drain_interrupt_queue_to_pending_input for the full
        # rationale. Regression of #17666 / #18760 — the drain block from the original PR #17939 was
        # deferred as "worth its own review" and never re-landed (#20271).
        self._drain_interrupt_queue_to_pending_input()

        # /goal continuation (queued user input still preempts), then /loop tick completion.
        for hook, what in (
            (self._maybe_continue_goal_after_turn, "goal continuation"),
            (self._maybe_complete_loop_tick_after_turn, "loop completion"),
        ):
            try:
                hook()
            except Exception as _exc:
                logging.debug("%s hook failed: %s", what, _exc)

        # Continuous voice: restart recording off-thread (beep + recorder start would block process_loop).
        if self._voice_mode and self._voice_continuous and not self._voice_recording:
            def _restart_recording():
                try:
                    if self._voice_tts:
                        self._voice_tts_done.wait(timeout=60)
                        time.sleep(0.3)
                    # A barge-in capture already owns the mic and submits the interruption itself.
                    if self._voice_barge_capture.is_set():
                        return
                    self._voice_start_recording()
                    self._app.invalidate()
                except Exception as e:
                    _cprint(f"{_DIM}Voice auto-restart failed: {e}{_RST}")
            threading.Thread(target=_restart_recording, daemon=True).start()

        with suppress(Exception):
            self._drain_process_notifications("cli-post-turn")

    def _tui_signal_handler(self, signum, frame):
        """SIGHUP/SIGTERM -> graceful shutdown.

        The agent is hard-interrupted first so its daemon thread can kill the tool's
        setsid subprocess group before the main thread unwinds (else an orphan child).
        ``logger.debug`` is guarded: logging is not reentrant-safe and a shutdown race
        can raise ``KeyError`` inside the handler, bypassing prompt_toolkit's unwind.
        """
        with suppress(Exception):
            logger.debug("Received signal %s, triggering graceful shutdown", signum)
        # Arm the backstop IMMEDIATELY: if the unwind wedges, _run_cleanup never arms its own.
        # Shutdown intent is now unambiguous — arm the exit backstop IMMEDIATELY, before the graceful unwind
        # below. If any step of that unwind wedges (main thread parked in a syscall, prompt_toolkit teardown
        # never returning), _run_cleanup never runs and would never arm its own watchdog — leaving a "dead"
        # CLI alive for minutes (#65998 class).
        # Arm the exit backstop now that shutdown intent is unambiguous — covers wedges in the unwind below
        # that would otherwise leave the process alive with no watchdog (#65998 class).
        _arm_exit_watchdog_on_shutdown_signal()
        if self._agent_running or self._command_running:
            # Slow slash commands such as /compress execute outside the normal conversation turn but can
            # still own an auxiliary provider request. Treat process shutdown during either activity as an
            # explicit hard stop so protected compression does not survive its CLI host.
            _interrupt_agent_for_signal(self.agent, signum)
        # Prefer app.exit() over raising KeyboardInterrupt: a KBI from a signal handler
        # lands in a pt Task ("Unhandled exception in event loop" + "Press ENTER to
        # continue..."); call_soon_threadsafe lets the loop unwind normally.
        try:
            from prompt_toolkit.application.current import get_app_or_none
            _app = get_app_or_none()
            _loop = getattr(_app, "loop", None)
            if _loop is not None:
                _loop.call_soon_threadsafe(_app.exit)
                return  # clean unwind — no traceback, no ENTER pause
        except Exception:
            pass
        raise KeyboardInterrupt()  # fallback for non-prompt_toolkit contexts

    def _tui_print_startup(self):
        """Startup output: light-mode probe, banner, advisories, resume/welcome lines, tips."""
        with suppress(Exception):  # light-mode probe before pt grabs the tty (cached)
            _detect_light_mode()
        # Scroll the cursor to the last row so banner, responses and prompt pin to the bottom.
        with suppress(Exception):
            _term_lines = shutil.get_terminal_size().lines
            if _term_lines > 2:
                print("\n" * (_term_lines - 1), end="", flush=True)

        self.show_banner()
        self._show_security_advisories()
        self._show_browser_backend_notice()

        # First-run: an unconfigured install routes into provider onboarding instead of
        # a chat that spins ~30s and fails with a provider-specific error. TTY only.
        try:
            if sys.stdin.isatty() and not self._runtime_credentials_ready():
                self._offer_first_run_setup()
        except Exception:
            logger.debug("first-run setup offer failed", exc_info=True)

        if self._resumed and self._preload_resumed_session():
            self._display_resumed_history()

        _welcome_skin = None  # stays None when the skin engine failed
        _welcome_text = "Welcome to Hermes Agent! Type your message or /help for commands."
        _welcome_color = "#FFF8DC"
        try:
            from hermes_cli.skin_engine import get_active_skin
            _welcome_skin = get_active_skin()
            _welcome_text = _welcome_skin.get_branding("welcome", _welcome_text)
            _welcome_color = _welcome_skin.get_color("banner_text", _welcome_color)
        except Exception:
            pass
        self._console_print(f"[{_welcome_color}]{_welcome_text}[/]")

        self._tui_startup_prewarm_and_warnings(_welcome_skin)
        self._print_random_tip()

        self._tui_startup_background_maintenance()
        # Before the background preload is folded in (at agent init), show the REQUESTED names.
        _skills_for_line = self.preloaded_skills or list(self._preload_skills_requested or [])
        if _skills_for_line and not self._startup_skills_line_shown:
            self._console_print(f"[bold {_accent_hex()}]Activated skills:[/] {', '.join(_skills_for_line)}")
            self._startup_skills_line_shown = True
        self._console_print()

    def _tui_startup_prewarm_and_warnings(self, _welcome_skin):
        """Idle-window prewarms (picker cache, agent runtime imports) plus the redaction-off and OpenClaw-residue banners."""
        # Warm the /model picker cache off-thread (else its first open blocks ~1-2s).
        with suppress(Exception):
            from hermes_cli.model_switch_providers import prewarm_picker_cache_async
            prewarm_picker_cache_async()

        # Pre-import the agent runtime (~1.5s: run_agent + OpenAI SDK) off-thread; the import
        # lock makes an early submit block on the remaining work rather than redo it.
        # Skipped when Termux defers agent startup on purpose.
        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            def _prewarm_agent_runtime() -> None:
                try:
                    import run_agent  # noqa: F401  (imports model_tools + tool registry)
                    import openai  # noqa: F401
                except Exception:
                    logger.debug("agent runtime pre-import failed", exc_info=True)

            threading.Thread(target=_prewarm_agent_runtime, name="agent-runtime-prewarm", daemon=True).start()

        # Redaction is ON by default; be loud when the operator turned it off.
        with suppress(Exception):
            # The redactor snapshots its state at import time so any toggle now won't affect the running
            # process — we just want the operator to see that they're running without the safety net. See
            # #17691.
            _redact_raw = os.getenv("HERMES_REDACT_SECRETS", "true")
            if _redact_raw.lower() not in {"1", "true", "yes", "on"}:
                self._console_print(
                    "[bold red]⚠  Secret redaction is DISABLED[/] "
                    f"(HERMES_REDACT_SECRETS={_redact_raw}). "
                    "API keys and tokens may appear verbatim in chat output, "
                    "session JSONs, and logs. Set "
                    "[cyan]security.redact_secrets: true[/] in config.yaml "
                    "to re-enable."
                )
        # One-time banner when ~/.openclaw/ is left over from a migration.
        try:
            from agent.onboarding import (
                OPENCLAW_RESIDUE_FLAG, detect_openclaw_residue, is_seen, mark_seen, openclaw_residue_hint_cli,
            )
            if not is_seen(self.config, OPENCLAW_RESIDUE_FLAG) and detect_openclaw_residue():
                try:
                    _resid_color = _welcome_skin.get_color("banner_dim", "#B8860B")
                except Exception:
                    _resid_color = "#B8860B"
                self._console_print(f"[{_resid_color}]{openclaw_residue_hint_cli()}[/]")
                try:
                    from hermes_cli.config import get_config_path as _get_cfg_path_resid
                    mark_seen(_get_cfg_path_resid(), OPENCLAW_RESIDUE_FLAG)
                except Exception:
                    pass  # banner fires again next session
        except Exception:
            pass

    def _tui_startup_background_maintenance(self):
        """Best-effort startup passes: curator skill maintenance, personal + org skill sync."""
        with suppress(Exception):
            from agent.curator import maybe_run_curator
            maybe_run_curator(
                idle_for_seconds=float("inf"),  # CLI startup = fully idle
                on_summary=lambda msg: self._console_print(f"[dim #6b7684]💾 {msg}[/]"),
            )

        # Skill sync (personal, then org-shared): inert unless the access gate is open
        # and a sync base URL is configured. The org pull is gated on a real org role on
        # the token (only issued for multi-member orgs), so a solo account never hits
        # the network here. Both fail-quiet.
        try:
            from tools.skills_sync_client import maybe_pull_skills
            from tools.skills_sync_client_org import maybe_pull_org_skills
        except Exception:
            return
        for pull in (maybe_pull_skills, maybe_pull_org_skills):
            with suppress(Exception):
                pull()

    def _tui_build_application(self, layout, kb, style):
        """Construct the prompt_toolkit Application for the REPL."""
        _cpr_disabled_output = _select_classic_cli_pt_output(sys.stdout)

        # Kitty placeholders encode the image id in exact foreground RGB, so the whole app
        # runs 24-bit there. ColorDepth is imported lazily for tests that stub prompt_toolkit.
        extra_kw = {}
        if pet_render.supports_kitty_placeholders():
            from prompt_toolkit.output import ColorDepth

            extra_kw["color_depth"] = ColorDepth.DEPTH_24_BIT
        if _cpr_disabled_output is not None:
            extra_kw["output"] = _cpr_disabled_output
        if _STEADY_CURSOR is not None:
            extra_kw["cursor"] = _STEADY_CURSOR
        return Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
            # 0 (default) avoids fighting terminal auto-scroll in non-fullscreen mode.
            refresh_interval=float(CLI_CONFIG.get("display", {}).get("cli_refresh_interval", 0)),
            # Erase the bottom chrome on exit instead of freezing a copy into scrollback.
            # Without this, prompt_toolkit's render_as_done teardown repaints the chrome one last time and
            # leaves it stranded above the exit summary — so a dead status bar + empty prompt sit between
            # the conversation transcript and the "Resume this session" block, and stack with the next
            # session's UI on resume (#38252). The actual conversation transcript is printed through
            # patch_stdout into normal scrollback and is unaffected; only the managed chrome is erased.
            # Applies to every exit path (/exit, /quit, EOF, Ctrl+C).
            erase_when_done=True,
            **extra_kw,
        )

    def _tui_install_signal_handlers(self):
        """SIGTERM/SIGHUP -> graceful shutdown; Windows absorbs SIGINT (see body)."""
        try:
            import signal as _signal
            _signal.signal(_signal.SIGTERM, self._tui_signal_handler)
            if hasattr(_signal, 'SIGHUP'):
                _signal.signal(_signal.SIGHUP, self._tui_signal_handler)

            # Windows: absorb SIGINT. Win32 delivers spurious CTRL_C_EVENT when children spawn
            # from background threads, which would unwind app.run() mid-turn. Real Ctrl+C is
            # bound by prompt_toolkit. Never call agent.interrupt() here (fake user message).
            if sys.platform == "win32":
                _signal.signal(_signal.SIGINT, lambda signum, frame: None)
        except Exception:
            pass  # restricted environments

    def _tui_stdin_usable(self) -> bool:
        """Validate fd 0 before prompt_toolkit starts; on macOS fall back to a select() loop when kqueue can't watch it (uv-managed Python)."""
        try:
            os.fstat(0)
        except OSError:
            print(
                "Error: stdin (fd 0) is not available.\n"
                "This can happen with certain Python installations (e.g. uv-managed cPython on macOS).\n"
                "Try reinstalling Python via pyenv or Homebrew, then re-run: hermes setup"
            )
            return False
        if sys.platform == "darwin":
            import selectors as _selectors
            try:
                if hasattr(_selectors, "KqueueSelector"):
                    _kq = _selectors.KqueueSelector()
                    try:
                        _kq.register(0, _selectors.EVENT_READ)
                        _kq.unregister(0)
                    finally:
                        _kq.close()
            except (OSError, ValueError, KeyError):
                import asyncio as _aio_probe

                class _SelectEventLoopPolicy(_aio_probe.DefaultEventLoopPolicy):
                    def new_event_loop(self):
                        return _aio_probe.SelectorEventLoop(_selectors.SelectSelector())

                _aio_probe.set_event_loop_policy(_SelectEventLoopPolicy())
        return True

    def run(self):
        """Run the interactive CLI loop with persistent input at bottom."""
        if not self._claim_active_session("cli"):
            return

        self._tui_print_startup()
        self._tui_init_run_state()
        kb = self._tui_build_key_bindings()
        layout, style = self._tui_build_layout(kb)

        app = self._tui_build_application(layout, kb, style)
        _disable_prompt_toolkit_cpr_warning(app)
        app.after_render += self._pet_flush_kitty_frame
        from hermes_cli.observability.shared_metrics_startup import cli_prompt_ready_handler
        app.after_render += cli_prompt_ready_handler()
        self._app = app

        # Ghost status-bar lines on resize: pt's renderer scrolls the terminal after each
        # paint, pushing chrome into scrollback where a column-shrink reflows it into
        # duplicates. Wrapping _output_screen_diff keeps its reserve-space branch from firing.
        try:
            # Background: prompt_toolkit's renderer (renderer.py L232-242) explicitly moves the cursor to
            # the bottom of the canvas after painting "to make sure the terminal scrolls up, even when the
            # lower lines of the canvas just contain whitespace". In non-fullscreen mode this scrolls chrome
            # content (status bar, input rules) into terminal scrollback on every render. When the terminal
            # column-shrinks, the emulator reflows the previously rendered full-width rows into multiple
            # narrower rows that get pushed up — leaving ghost duplicates AND polluting scrollback. Same
            # issue as pt #29 (open since 2014), #1675, #1933. Surgical fix: wrap _output_screen_diff so
            # that when its internal `if current_height > previous_screen.height` branch fires (the one that
            # does the bottom-cursor-move), we make it fall through by inflating previous_screen.height
            # first.
            import prompt_toolkit.renderer as _pt_renderer
            from prompt_toolkit.renderer import _output_screen_diff as _orig_osd

            if not getattr(_pt_renderer, "_hermes_osd_patched", False):
                _pt_renderer._output_screen_diff = functools.partial(
                    _hermes_call_output_screen_diff, _orig_osd
                )
                _pt_renderer._hermes_osd_patched = True
        except Exception:
            pass

        _apply_bracketed_paste_timeout_patch()

        self._install_resize_recovery(app)

        threading.Thread(target=self._tui_spinner_loop, daemon=True).start()
        threading.Thread(target=self._tui_process_loop, daemon=True).start()
        # Wake word listener off-thread so a first-run engine install never blocks the prompt.
        threading.Thread(target=self._tui_wake_startup, daemon=True, name="wake-startup").start()

        atexit.register(_run_cleanup)
        self._tui_install_signal_handlers()

        if not self._tui_stdin_usable():
            _run_cleanup()
            self._print_exit_summary()
            return

        try:
            with patch_stdout():
                try:
                    # run_in_terminal() may return either: • a coroutine / Future (prompt_toolkit ≥ 3.0) —
                    # must be scheduled via ensure_future so the coroutine is actually awaited; calling it
                    # bare would leave it unawaited and silently drop the output (fixes #23185 Bug A). •
                    # None (some mocks / older PT builds) — just call the inner function directly since PT
                    # already executed it synchronously. Do NOT fall back to a bare _pt_print when
                    # ensure_future raises, because run_in_terminal already invoked the lambda in that case
                    # (the mock path), which would double-print the line.
                    import asyncio as _aio
                    _aio.get_running_loop().set_exception_handler(self._tui_suppress_closed_loop_errors)
                except Exception:
                    pass  # no running loop -- nothing to patch
                # Record that the app enables focus reporting + mouse tracking so _run_cleanup
                # resets them; extended key modes are popped by the same reset.
                # When multiline shortcuts are on, also ask supported terminals (e.g. iTerm2) to report
                # modified keys distinctly (kitty protocol + modifyOtherKeys); the cleanup reset pops both
                # modes. See #36823.
                _mark_tui_input_modes_active()
                if self._tui_multiline_shortcuts:
                    _enable_extended_enter_keys(app.output)
                self._pet_start_anim()
                app.run()
        except (EOFError, KeyboardInterrupt, BrokenPipeError):
            pass
        except (KeyError, OSError) as _stdin_err:
            # Selector registration failures from broken stdin and I/O errors from a
            # broken stdout during interrupt (EIO is suppressed).
            _errno = getattr(_stdin_err, "errno", None) if isinstance(_stdin_err, OSError) else None
            _msg = str(_stdin_err)
            if _errno == errno.EIO:
                pass
            elif _errno in {errno.EINVAL, errno.EBADF} or any(
                s in _msg for s in ("is not registered", "Bad file descriptor", "Invalid argument")
            ):
                print(_t("cli.startup.stdin_unusable", error=str(_stdin_err)))
            else:
                raise
        finally:
            # A resize right before exit leaves its recovery (and the paints it held) unrun.
            _release_paints()
            self._tui_shutdown()

        # /update relaunch happens here, after prompt_toolkit restored terminal modes, on the
        # main thread (the process_loop thread would skip cleanup / only exit itself on Windows).
        if self._pending_relaunch:
            from hermes_cli.relaunch import relaunch
            relaunch(self._pending_relaunch, preserve_inherited=False)


def _build_cli_from_args(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget, verbose, compact, resume, checkpoints, pass_session_id, ignore_rules, skills):
    """Resolve the toolset list (explicit / coding posture / platform default), construct HermesCLI, and start the background skills preload."""
    toolsets_list = None
    if isinstance(toolsets, str) and toolsets:
        toolsets_list = [t.strip() for t in toolsets.split(",")]
    elif isinstance(toolsets, (list, tuple)) and toolsets:
        # Fire may pass multiple --toolsets as a tuple
        toolsets_list = []
        for t in toolsets:
            toolsets_list.extend([x.strip() for x in t.split(",")] if isinstance(t, str) else [str(t)])
    elif not toolsets:
        # Coding posture inside a code workspace, else the shared platform resolver.
        try:
            from agent.coding_context import coding_selection
            toolsets_list = coding_selection(platform="cli", config=CLI_CONFIG)
        except Exception:
            toolsets_list = None
        if toolsets_list is None:
            from hermes_cli.tools_config import _get_platform_tools
            toolsets_list = sorted(_get_platform_tools(CLI_CONFIG, "cli"))

    parsed_skills = _parse_skills_argument(skills)

    try:
        cli = HermesCLI(
            model=model,
            toolsets=toolsets_list,
            provider=provider,
            reasoning=reasoning,
            api_key=api_key,
            base_url=base_url,
            max_turns=max_turns,
            run_budget=run_budget,
            verbose=verbose,
            compact=compact,
            resume=resume,
            checkpoints=checkpoints,
            pass_session_id=pass_session_id,
            ignore_rules=ignore_rules,
        )
    except ImportError as e:
        # Direct `python cli.py` bypasses cmd_chat's partial-update ImportError handler.
        from hermes_constants import emit_partial_update_hint

        if emit_partial_update_hint(e):
            sys.exit(1)
        raise

    # skills.auto_load rides the same background preload as -s; --ignore-rules skips it with
    # the rest of the auto-injected context. Resolved here (not lazily in the agent) so the
    # session id is real for ${HERMES_SESSION_ID} and -s can dedupe against it.
    from agent.skill_commands import build_auto_load_prompt, resolve_auto_load_skills
    auto_load_names = [] if getattr(cli, "ignore_rules", ignore_rules) else resolve_auto_load_skills(CLI_CONFIG)
    if not auto_load_names:
        cli._auto_load_skills_result = ("", [], [])
    if parsed_skills or auto_load_names:
        # Load the skill payloads in the background: skill_view walks the full skills
        # tree per skill (~0.5s for a large library) and the result is only consumed
        # at agent init, not by the banner. finalize_preloaded_skills() joins the
        # thread before any consumer reads cli.system_prompt.
        def _load_preloaded_skills() -> None:
            try:
                if auto_load_names:
                    cli._auto_load_skills_result = build_auto_load_prompt(task_id=cli.session_id, user_config=CLI_CONFIG)
                if parsed_skills:
                    cli._preload_skills_result = build_preloaded_skills_prompt(
                        parsed_skills, task_id=cli.session_id, excluded_loaded_names=set(cli._auto_load_skills_result[1]))
            except Exception as exc:  # surfaced by finalize
                cli._preload_skills_error = exc

        cli._preload_skills_requested = [*auto_load_names, *(s for s in parsed_skills if s not in auto_load_names)]
        cli._preload_skills_thread = threading.Thread(target=_load_preloaded_skills, name="skills-preload", daemon=True)
        cli._preload_skills_thread.start()
    return cli


def _run_legacy_gateway():
    """Legacy `cli.py --gateway` entry: arm the startup watchdog (before importing the gateway graph), then run it."""
    import asyncio
    with suppress(Exception):
        from hermes_startup_watchdog import arm_startup_watchdog
        arm_startup_watchdog()
    from gateway.run import start_gateway
    print(_t("cli.gateway.starting"))
    asyncio.run(start_gateway())


def _start_worktree_setup(list_tools, list_toolsets, worktree, w):
    """Start isolated-worktree creation (+ tool prewarm) in the background.

    Returns a join callable that publishes ``_active_worktree``/TERMINAL_CWD and
    schedules stale-worktree GC, or None when no worktree is wanted.
    """
    if list_tools or list_toolsets or not (worktree or w or CLI_CONFIG.get("worktree", False)):
        return None
    # Overlap tool discovery with the I/O-bound worktree setup so show_banner() hits a warm
    # cache (~0.4s). Only on the -w path: plain `hermes` has no I/O wait to hide.
    def _prewarm_tools() -> None:
        try:
            import model_tools as _mt
            _mt.get_tool_definitions(quiet_mode=True)
        except Exception:
            logger.debug("tool prewarm failed", exc_info=True)

    threading.Thread(target=_prewarm_tools, name="tool-prewarm", daemon=True).start()
    _sync_base = CLI_CONFIG.get("worktree_sync", True)
    _wt_result: dict = {}

    def _create_worktree() -> None:
        try:
            _wt_result["info"] = _setup_worktree(sync_base=_sync_base)
        except Exception:
            logger.debug("worktree setup failed", exc_info=True)
            _wt_result["info"] = None

    _wt_thread = threading.Thread(target=_create_worktree, name="worktree-setup", daemon=True)
    _wt_thread.start()

    def _worktree_maintenance(repo: str) -> None:
        _prune_stale_worktrees(repo)
        _maintain_pack_health(repo)

    def _join_worktree() -> Optional[dict[str, str]]:
        _wt_thread.join(timeout=120)
        info = _wt_result.get("info")
        if not info:
            return info
        global _active_worktree
        _active_worktree = info
        os.environ["TERMINAL_CWD"] = info["path"]
        atexit.register(_cleanup_worktree, info)
        # GC stale worktrees AFTER _setup_worktree so they never race on git's worktree
        # metadata (the new tree is immune: <24h age gate + live pid lock); then repack
        # once refs are final so lookups stay fast on multi-agent boxes.
        _repo = _git_repo_root()
        if _repo:
            threading.Thread(target=_worktree_maintenance, args=(_repo,), name="worktree-prune", daemon=True).start()
        return info

    return _join_worktree


def main(
    query: str | None = None,
    q: str | None = None,
    oneshot: bool = False,
    image: str | None = None,
    toolsets: str | None = None,
    skills: str | list[str] | tuple[str, ...] | None = None,
    model: str | None = None,
    provider: str | None = None,
    reasoning: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_turns: int | None = None,
    run_budget: float | None = None,
    verbose: Optional[bool] = None,
    quiet: bool = False,
    compact: bool = False,
    list_tools: bool = False,
    list_toolsets: bool = False,
    gateway: bool = False,
    resume: str | None = None,
    worktree: bool = False,
    w: bool = False,
    checkpoints: bool = False,
    pass_session_id: bool = False,
    output_format: str = "text",
    ignore_user_config: bool = False,
    ignore_rules: bool = False,
):
    """
    Hermes Agent CLI - Interactive AI Assistant
    
    Args:
        query: Query to run. On a real TTY this seeds an interactive session
            (submitted literally as the first turn); with --oneshot/-Q or a
            non-TTY it answers and exits. Alias: -q
        q: Shorthand for --query
        oneshot: With -q: force the legacy answer-and-exit single-query mode
            even on a TTY.
        image: Optional local image path to attach to a single query
        toolsets: Comma-separated list of toolsets to enable (e.g., "web,terminal")
        skills: Comma-separated or repeated list of skills to preload for the session
        model: Model to use (default: anthropic/claude-opus-4-20250514)
        provider: Inference provider ("auto", "openrouter", "nous", "openai-codex", "zai", "kimi-coding", "minimax", "minimax-cn")
        reasoning: Reasoning effort for this run (none|minimal|low|medium|high|xhigh|max|ultra). Overrides agent.reasoning_effort.
        api_key: API key for authentication
        base_url: Base URL for the API
        max_turns: Maximum tool-calling iterations (default: 60)
        verbose: Enable verbose logging
        compact: Use compact display mode
        list_tools: List available tools and exit
        list_toolsets: List available toolsets and exit
        resume: Resume a previous session by its ID (e.g., 20260225_143052_a1b2c3)
        worktree: Run in an isolated git worktree (for parallel agents). Alias: -w
        w: Shorthand for --worktree
    
    Examples:
        python cli.py                            # Start interactive mode
        python cli.py --toolsets web,terminal    # Use specific toolsets
        python cli.py --skills hermes-agent-dev,github-auth
        python cli.py -q "What is Python?"       # Single query mode
        python cli.py -q "Describe this" --image ~/storage/shared/Pictures/cat.png
        python cli.py --list-tools               # List tools and exit
        python cli.py --resume 20260225_143052_a1b2c3  # Resume session
        python cli.py -w                         # Start in isolated git worktree
        python cli.py -w -q "Fix issue #123"     # Single query in worktree
    """
    # UTF-8 stdio on Windows before any print (Rich box-drawing would UnicodeEncodeError on cp1252).
    with suppress(Exception):
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()

    os.environ["HERMES_INTERACTIVE"] = "1"  # terminal_tool: interactive sudo prompts with timeout

    if gateway:
        _run_legacy_gateway()
        return

    if not (list_tools or list_toolsets):
        from hermes_cli.process_identity import register_self
        from hermes_cli.shared_profile_warning import shared_profile_warning

        register_self("cli")
        warning = shared_profile_warning()
        if warning:
            print(_t("cli.startup.warning", warning=warning), file=sys.stderr)

    _join_worktree = _start_worktree_setup(list_tools, list_toolsets, worktree, w)
    query = query or q
    # ``hermes chat`` already validated this; the direct Fire entry point gets the same contract.
    if output_format == "stream-json":
        if not query:
            raise ValueError("--format stream-json requires -q/--query")
        quiet = True
    cli = _build_cli_from_args(model, toolsets, provider, reasoning, api_key, base_url, max_turns, run_budget,
                               verbose, compact, resume, checkpoints, pass_session_id, ignore_rules, skills)

    # Join the background worktree creation before anything consumes TERMINAL_CWD.
    # A requested worktree whose setup failed aborts: never silently run without isolation.
    wt_info = _join_worktree() if _join_worktree is not None else None
    if _join_worktree is not None and not wt_info:
        return

    # Inject worktree context into agent's system prompt
    if wt_info:
        wt_note = (
            f"\n\n[System note: You are working in an isolated git worktree at "
            f"{wt_info['path']}. Your branch is `{wt_info['branch']}`. "
            f"Changes here do not affect the main working tree or other agents. "
            f"Remember to commit and push your changes, and create a PR if appropriate. "
            f"The original repo is at {wt_info['repo_root']}.]"
        )
        cli.system_prompt = (cli.system_prompt or "") + wt_note

    if list_tools or list_toolsets:
        cli.show_banner()
        (cli.show_tools if list_tools else cli.show_toolsets)()
        sys.exit(0)

    atexit.register(_run_cleanup)  # interactive mode registers again in run() (idempotent)
    _install_single_query_signal_handlers(cli)

    if query or image:
        _run_single_query_mode(cli, query, image, quiet, oneshot, stream_json=output_format == "stream-json")
        return
    cli.run()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
