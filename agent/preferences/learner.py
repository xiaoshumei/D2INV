"""
User Preference Learner for D2INV Agent — Phase 3.

Tracks user preferences implicitly (from choices/patterns) and explicitly
(from direct statements). Persists across sessions via LongTermMemory.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent.memory.long_term import LongTermMemory


_DEFAULT_PREFERENCES = {
    "visualization": {"chart_library": "echarts", "color_scheme": "default"},
    "language": "zh",
    "verbosity": "normal",  # minimal | normal | detailed
    "auto_save_results": True,
}


class PreferenceLearner:
    """
    Learns and persists user preferences across sessions.

    Preferences are stored as facts in LongTermMemory with type='preference'.
    Can infer preferences from user behaviour patterns.
    """

    def __init__(self, memory: LongTermMemory):
        self._memory = memory
        self._cache: Dict[str, Any] = {}

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific preference value (checks cache first, then memory)."""
        if key in self._cache:
            return self._cache[key]

        val = self._memory.recall_by_fact(f"pref:{key}")
        if val is not None:
            try:
                parsed = json.loads(val)
                self._cache[key] = parsed
                return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return self._try_default(key, default)

    def get_all(self) -> Dict[str, Any]:
        """Return all known preferences as a flat dict."""
        all_prefs = {}
        results = self._memory.recall("preference:", memory_type="fact", top_k=20)
        for r in results:
            key = r["metadata"].get("key", "")
            if key.startswith("pref:"):
                short_key = key[5:]
                try:
                    all_prefs[short_key] = json.loads(r["metadata"].get("value", ""))
                except (json.JSONDecodeError, TypeError):
                    all_prefs[short_key] = r["metadata"].get("value")
        return all_prefs

    def set(self, key: str, value: Any) -> None:
        """Store a preference explicitly (called by user action or system)."""
        self._cache[key] = value
        self._memory.remember_fact(
            key=f"pref:{key}",
            value=json.dumps(value, ensure_ascii=False),
        )

    def observe_chart_choice(self, chart_type: str) -> None:
        """Increment use count for a chart type (implicit preference signal)."""
        current = self.get("chart_usage", {})
        current[chart_type] = current.get(chart_type, 0) + 1
        self.set("chart_usage", current)

        # If a chart type is used 3+ times, auto-promote to preferred default
        if current[chart_type] >= 3:
            self.set("preferred_chart", chart_type)

    def observe_dataset(self, dataset_name: str) -> None:
        """Track which datasets user accesses most."""
        history = self.get("dataset_history", [])
        history.insert(0, dataset_name)
        # Limit to last 20
        self.set("dataset_history", history[:20])

        # Count frequency
        freq = self.get("dataset_frequency", {})
        freq[dataset_name] = freq.get(dataset_name, 0) + 1
        self.set("dataset_frequency", freq)

    def observe_language(self, detected_lang: str) -> None:
        """Observe user's language from message patterns."""
        if detected_lang in ("zh", "en", "ja", "ko", "other"):
            self.set("language", detected_lang)

    def clear(self) -> None:
        """Reset all learned preferences."""
        self._cache.clear()
        self._memory.forget_by_type("fact")  # Only removes facts, not conversations

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _try_default(self, key: str, default: Any) -> Any:
        parts = key.split(".")
        val = _DEFAULT_PREFERENCES
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return default
        return val


_DEFAULT_PREFERENCES = {
    "visualization": {"chart_library": "echarts", "color_scheme": "default"},
    "language": "zh",
    "verbosity": "normal",
    "auto_save_results": True,
}