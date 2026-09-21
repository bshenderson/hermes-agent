# GXTD-647 — Hermes upstream integration candidate

Status: candidate validation in progress; **live deployment held separately**.

## Authority and boundary

Barry authorized preparing, testing, committing and pushing a dedicated integration
branch. This is not authorization to move either fork `main` or installed source,
refresh production dependencies, sync profiles, or restart services.

- Branch: `integration/GXTD-647-upstream` in `bshenderson/hermes-agent`.
- Installed/fork baseline: `2bf7d6ba1135f46330a9745af6910b287bc6c2e7`.
- Frozen upstream `main`: `4c2c20f6e7c3ac9dfc1b1049f68391e08f8dffa1`.
- Upstream reports semantic version `0.21.3`; release tag `v2026.9.14` is older
  than this pinned main commit. This branch is **not** simply that release tag.
- Preservation commit: `06fae659ca`. All 18 deployed overlay blobs were compared
  byte-for-byte by SHA256 against the captured installed source before merging.
- Private recovery bundle on chatter:
  `/home/barry/.cache/homelab-rollbacks/gxtd647-source-1789961107273403618`.
  Contains `source/`, `tracked.patch`, `manifest.json`, and `canonical-matches.json`.
  It contains only inventoried Python source, not profile state or credentials.

The first delegated attempt copied an inert `source/` directory rather than
applying the overlay. Parent review rejected it before publication, retained it
only on a local archive branch, and rebuilt this branch from the installed base.
Neither rejected commit belongs to the candidate's ancestry.

## Preserved capabilities

| Ownership | Retained behavior |
|---|---|
| GXTD-603 | Explicit request-local `/research` policy, native backend pins, no alternative tool bypass, bounded answer-only synthesis, hard-stop precedence |
| GXTD-594 | Opt-in gateway-owned runtime/board/delivery/executor aggregates, separate metrics credential, no profile-mirrored metrics route |
| GXTD-624 | Signed turn-local OWUI location, request/profile binding, expiry, unknown-weather clarification, no location lent to research mode |
| GXTD-590 | Request-owned cancellation, SSE disconnect detection, manual-compression startup/cancellation/recovery, bounded complete-input compression and explicit no-fallback policy |
| GXTD-587 | Clear only the recovered secondary adapter's stale health status |
| Earlier fork commits | Bounded media-result projection, tool-loop/no-progress controls, ordinary explicit toolsets omit Kanban |

The undeployed GXTD-605 compaction fixed-point candidate is excluded.

## Merge decisions

- Keep upstream's extracted CLI runtime mixin; carry command-running signal
  cancellation into `hermes_cli/cli_tui_runtime_mixin.py`, not back into `cli.py`.
- Keep the shared upstream manual-compression API while preserving lazy agent
  initialization on resume, interrupt reset only at a new idle command, and
  hard cancellation on interruption.
- Combine API policy and telemetry with upstream relay metadata, interim-message
  controls, notification suppression, attribution, session-history default-deny,
  unanswered-turn adoption, and `_submit_api_worker` ownership.
- Preserve durable-run `served_runtime` return values while instrumenting the
  actual executor lifetime, not the cancelled async waiter.
- Keep research-mode idempotency fingerprints and the new notification field.
- Preserve bounded synthesis and upstream diagnostic-status classification.
- Keep pinned Firecrawl clients request-local; remove the obsolete upstream import.

## Regression disposition

The initial narrow run exposed stale inherited web-finalization assertions, a
fallback-route fixture missing upstream's new endpoint field, and multiplex tests
reading asynchronously published status before flushing. Repairs change test
contracts/fixtures, not production policy. The expanded compression marathon
fixture separately disables only research finalization so it can exercise eight
compression cycles; the research-budget stop retains dedicated integration tests.
The unmodified upstream compression-refund suite passes on the pinned upstream
baseline, and the adjusted fixture passes on this candidate.

Owned research, synthesis, location and lifecycle tests are ported to real branch
imports. Homelab source-text guards and tests reading live runtime paths are not
copied. Added real loopback HTTP cases exercise signature rejection, location/
research isolation and private metrics authorization with disposable state.

## Reproduction and dependency boundary

From this checkout, select an isolated development interpreter:

```sh
UV_PROJECT_ENVIRONMENT=/path/to/isolated-venv uv sync --frozen \
  --python 3.11 --extra dev --extra messaging --extra firecrawl --extra acp --no-install-project
HERMES_PYTHON=/path/to/isolated-venv/bin/python \
  python3 scripts/homelab/test_gxtd647.py
```

The wrapper requires every explicitly named test file, adds scoped compression,
API-server and web-tool regressions, creates a disposable HOME, and invokes the
canonical hermetic per-file runner with two workers and no failure retries.
The first runs reused the installed **development** interpreter read-only; the
final matrix uses `/home/barry/.cache/gxtd647-venv`, built from the candidate's
frozen lock. Production `venv` and `.venv` are not changed.

Import smoke resolves `run_agent`, `cli`, policies, API adapter and web tools from
the integration worktree, not production. CLI `--help` is exercised without user
credentials or real model requests. A bounded read-only review found no blocking
issues in the integration delta; that is not an audit of every upstream change.

Detailed final results are recorded in the adjacent validation artifact and Jira.

## Artifact alignment and deployment hold

This branch is a candidate, not a new live authority. Root context, governance,
profile SOUL/config, architecture, KB, generated model projections and homelab
sync/apply scripts therefore require no live/current-version edits. Existing
historical reports remain historical. The branch report, regression wrapper and
Jira GXTD-647 carry the preparation result. No broad sync was run.

Separate adoption must address:

1. Review and explicitly accept this exact candidate; choose how it reaches fork
   `main` without dropping subsequent upstream or operator work.
2. Re-inventory installed dirty source and active writers; preserve any changes
   since the source snapshot. Never use `--force` to erase the deployed overlay.
3. Prepare scoped state/schema recovery plus the Python dependency refresh and
   required Node/dashboard/TUI builds. This task does not certify those builds,
   other OS platforms, every optional provider, or the full upstream test suite.
4. Drain gateway and dashboard-owned work and use a genuinely independent control
   shell. Session-lease presence and idle GPU slots are not sufficient alone.
5. Use the canonical homelab runtime transaction after the fork/source disposition
   is safe, then exercise service health, imports, authenticated profile/API and
   browser workflows. No real-model or production/browser acceptance is claimed here.
