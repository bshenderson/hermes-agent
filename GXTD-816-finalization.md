GXTD-816 remaining finalization repairs

Baseline: b44ed77bf22905658dc19d653e43f78e58c51eec. Scope: turn facade ambient exception attribution and independent timer cancellation. No routing/configuration changes or renewed native campaign.

Reproduced with tests/agent/test_gxtd816_finalization.py through canonical runner, isolated HOME and existing development interpreter. Both tests fail at intended assertions: successful nested turn suppresses first stop error and annotates unrelated caller exception; first timer cancellation skips later handles.

Hypotheses: sys.exception reads caller's handled exception (observed); explicit inner failure capture isolates ownership. Sequential unguarded cancel skips remaining handles (observed); aggregate first failure after all attempts preserves cleanup and error identity. Alternative broad teardown rewrite rejected as unnecessary.

Acceptance: propagate own primary exception unchanged; successful turn surfaces stop failure even inside caller except; attempt every timer cancellation including BaseException faults; annotate additional failures; retain existing release/scope reset behavior. Budget: at most three unsuccessful repair interventions, assessment after two. Live source movement requires current active-owner gate; this SSH controller is independent of gateway/dashboard ancestry. Rollback owner: operator through canonical updater, never restore conversation snapshots.

Result: first implementation resolves both discriminators. Explicit inner try/except owns the primary error instead of sys.exception(); join_threads attempts every handle before propagating its first error with subsequent errors attached. Canonical runner passed 52 tests across the new discriminator, cross-process turn lease and turn-facade lease modules. git diff --check passed. No model dispatch, config change, source movement, commit, push or service restart performed.

Alignment: source-only candidate. Existing root context correctly says residuals remain pending in serving runtime; leave it unchanged until adoption. Profiles, model registry, generated projections, architecture and routing contracts unchanged. Regression guards are the behavioral tests. Active CLI owner remains PID 977147, under independent SSH ancestry; live shared-source movement must not occur underneath this active owner without an explicit release/handoff. Gateway/dashboard PIDs observed as 192068/192055. Deployment and publication remain separate pending actions; issue must stay open.
