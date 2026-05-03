"""Hermes plugin queue_prefetch hybrid_recall opt-in contracts.

The plugin file imports Hermes runtime interfaces, so this test stubs only the
minimal module surface needed to instantiate NeuralMemoryProvider. It does not
import or start the real Hermes runtime.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _install_hermes_stubs() -> None:
    agent_module = types.ModuleType("agent")
    memory_provider_module = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    memory_provider_module.MemoryProvider = MemoryProvider
    agent_module.memory_provider = memory_provider_module
    sys.modules.setdefault("agent", agent_module)
    sys.modules.setdefault("agent.memory_provider", memory_provider_module)

    tools_module = types.ModuleType("tools")
    registry_module = types.ModuleType("tools.registry")
    registry_module.tool_error = lambda message: {"error": message}
    tools_module.registry = registry_module
    sys.modules.setdefault("tools", tools_module)
    sys.modules.setdefault("tools.registry", registry_module)


def _load_plugin_module():
    _install_hermes_stubs()
    plugin_path = Path(__file__).resolve().parent.parent / "hermes-plugin" / "__init__.py"
    spec = importlib.util.spec_from_file_location("nm_hermes_plugin_for_tests", plugin_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plugin module from {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLUGIN = _load_plugin_module()


class HermesPluginHybridRecallTests(unittest.TestCase):
    def _provider(self):
        provider = PLUGIN.NeuralMemoryProvider()
        provider._config = {"prefetch_limit": 3}
        provider._memory = MagicMock()
        provider._memory.recall.return_value = [
            {"id": 1, "content": "dense result", "similarity": 0.7}
        ]
        provider._memory.hybrid_recall.return_value = [
            {"id": 2, "content": "hybrid result", "similarity": 0.9}
        ]
        return provider

    def _run_prefetch(self, provider) -> str:
        provider.queue_prefetch("panel materials")
        self.assertIsNotNone(provider._prefetch_thread)
        provider._prefetch_thread.join(timeout=2)
        self.assertFalse(provider._prefetch_thread.is_alive())
        return provider._prefetch_result

    def test_default_env_uses_recall_not_hybrid(self) -> None:
        provider = self._provider()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NM_HERMES_HYBRID_RECALL", None)
            result = self._run_prefetch(provider)

        provider._memory.recall.assert_called_once_with("panel materials", k=3)
        provider._memory.hybrid_recall.assert_not_called()
        self.assertIn("dense result", result)

    def test_env_flag_set_routes_to_hybrid_recall(self) -> None:
        provider = self._provider()
        with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": "1"}, clear=False):
            result = self._run_prefetch(provider)

        provider._memory.hybrid_recall.assert_called_once_with(
            "panel materials", k=3, rerank=False)
        provider._memory.recall.assert_not_called()
        self.assertIn("hybrid result", result)

    def test_hybrid_recall_exception_falls_back_to_recall(self) -> None:
        provider = self._provider()
        provider._memory.hybrid_recall.side_effect = RuntimeError("hybrid unavailable")
        with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": "1"}, clear=False):
            result = self._run_prefetch(provider)

        provider._memory.hybrid_recall.assert_called_once_with(
            "panel materials", k=3, rerank=False)
        provider._memory.recall.assert_called_once_with("panel materials", k=3)
        self.assertIn("dense result", result)

    def test_env_flag_other_value_does_not_route(self) -> None:
        for value in ("0", "", "true", "yes"):
            with self.subTest(value=value):
                provider = self._provider()
                with patch.dict(os.environ, {"NM_HERMES_HYBRID_RECALL": value},
                                clear=False):
                    result = self._run_prefetch(provider)

                provider._memory.recall.assert_called_once_with(
                    "panel materials", k=3)
                provider._memory.hybrid_recall.assert_not_called()
                self.assertIn("dense result", result)


if __name__ == "__main__":
    unittest.main()
