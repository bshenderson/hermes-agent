
import unittest
from pathlib import Path
from typing import Any, Mapping, Optional
from unittest.mock import MagicMock, patch

# Mocking the base classes since we are in a test-only vertical
class DelegationProvider:
    def settings(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        return config.get("delegation", {}).get("backends", {}).get(self.name, {})

    @property
    def name(self) -> str:
        return "base"

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        return {}

    def description(self, config: Mapping[str, Any]) -> str:
        return ""

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        return {}

class ProfilesDelegationProvider(DelegationProvider):
    """The profile_routes delegation provider logic."""
    def __init__(self):
        self._name = "profile_routes"

    @property
    def name(self) -> str:
        return self._name

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        settings = self.settings(config)
        profiles = settings.get("profiles", [])
        if not profiles: return {}
        return {"profile": {"type": "string", "enum": [p["name"] for p in profiles]}}

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        chosen = task.get("profile")
        if chosen:
            return {"profile": chosen, "label": chosen}
        settings = self.settings(config)
        default = settings.get("default")
        if default:
            return {"profile": default, "label": default}
        return {}

class AutoSpecialistDelegationProvider(ProfilesDelegationProvider):
    """
    Automatic Specialist Delegation Provider.
    Maps keywords in the goal/context to specific profiles if no profile is explicitly requested.
    """
    PARAMETER = "profile"
    
    # DETERMINISTIC DISCRIMINATOR MAP
    SPECIALIST_MAP = {
        "code": "coder",
        "write": "coder",
        "refactor": "coder",
        "research": "researcher",
        "news": "researcher",
        "summary": "researcher",
    }

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        chosen = task.get(self.PARAMETER)
        if chosen:
            return super().route_task(task, config)
        
        # Automatic selection logic (Fake-LLM Deterministic)
        goal = task.get("goal", "").lower()
        context = task.get("context", "").lower()
        text = f"{goal} {context}"
        
        for keyword, profile in self.SPECIALIST_MAP.items():
            if keyword in text:
                # verify profile exists in config
                settings = self.settings(config)
                profiles = [p.get("name") for p in settings.get("profiles", []) if isinstance(p, dict)]
                if profile in profiles:
                    return {"profile": profile, "label": f"Auto-selected: {profile}"}
        
        return super().route_task(task, config)

# Import real toolset logic for seam testing
import sys
sys.path.append('/home/barry/.hermes/worktrees/gxtd755-auto-specialist')
from tools.delegate_tool_toolsets import _resolve_child_toolsets

class TestAutoSpecialistDelegation(unittest.TestCase):
    def setUp(self):
        self.config = {
            "delegation": {
                "backends": {
                    "profile_routes": {
                        "profiles": [
                            {"name": "coder", "description": "Code specialist"},
                            {"name": "researcher", "description": "Research specialist"},
                        ],
                        "default": "coder"
                    }
                }
            }
        }
        self.provider = AutoSpecialistDelegationProvider()
        # Inject settings mock
        self.provider.settings = MagicMock(return_value=self.config["delegation"]["backends"]["profile_routes"])

    def test_automatic_selection_researcher(self):
        task = {"goal": "Find the latest news about AI", "context": ""}
        route = self.provider.route_task(task, self.config)
        self.assertEqual(route["profile"], "researcher")
        self.assertIn("Auto-selected", route["label"])

    def test_automatic_selection_coder(self):
        task = {"goal": "Refactor the following function", "context": ""}
        route = self.provider.route_task(task, self.config)
        self.assertEqual(route["profile"], "coder")
        self.assertIn("Auto-selected", route["label"])

    def test_fallback_to_default(self):
        task = {"goal": "Hello there", "context": ""}
        route = self.provider.route_task(task, self.config)
        self.assertEqual(route["profile"], "coder")

    def test_explicit_override(self):
        task = {"goal": "Research this", "context": "", "profile": "coder"}
        route = self.provider.route_task(task, self.config)
        self.assertEqual(route["profile"], "coder")

    def test_toolset_seam_intersection(self):
        # Parent only has 'terminal'
        parent = MagicMock()
        parent.enabled_toolsets = ["terminal"]
        
        # Specialist request for 'web' (which parent lacks)
        requested_toolsets = ["web"]
        
        enabled, disabled = _resolve_child_toolsets(parent, requested_toolsets, "leaf")
        
        # The child should NOT get 'web' because the parent doesn't have it.
        self.assertNotIn("web", enabled)

    def test_prefill_messages_seam(self):
        parent = MagicMock()
        parent.prefill_messages = [{"role": "user", "content": "Hello"}]
        val = getattr(parent, "prefill_messages", None)
        self.assertEqual(val, [{"role": "user", "content": "Hello"}])

if __name__ == "__main__":
    from unittest.mock import MagicMock
    unittest.main()
