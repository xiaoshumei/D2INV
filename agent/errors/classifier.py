"""
Error Classifier for D2INV Agent — Phase 3.

Analyses error messages and classifies them into structured categories
for downstream intelligent recovery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class ErrorCategory(Enum):
    """Graded error categories from transient → permanent."""

    TRANSIENT = auto()       # Retry will likely fix (timeout, rate-limit)
    INPUT_INVALID = auto()   # Bad user input / parameter
    MISSING_DEPENDENCY = auto()  # File not found, module not installed
    PERMISSION = auto()       # Access denied, not allowed
    LOGIC = auto()            # Code bug, runtime TypeError, etc.
    RESOURCE_EXHAUSTED = auto()  # Out of memory, disk full
    UNKNOWN = auto()          # Uncategorised


# ---- Pattern-based classification rules ----
_RULES: List[Tuple[str, ErrorCategory]] = [
    # Transient
    (r"time\s*out|timed?\s*out|too\s+many\s+requests|rate\s+limit|503|service\s+unavailable",
     ErrorCategory.TRANSIENT),
    (r"connection\s+(refused|reset|aborted)|network|DNS|resolve\s+host",
     ErrorCategory.TRANSIENT),
    (r"temporarily\s+unavailable|please\s+retry|try\s+again\s+later",
     ErrorCategory.TRANSIENT),

    # Input validation
    (r"invalid\s+(argument|parameter|input|value|syntax)|unexpected\s+(keyword|argument)",
     ErrorCategory.INPUT_INVALID),
    (r"required\s+(argument|parameter|field).*missing",
     ErrorCategory.INPUT_INVALID),
    (r"out\s+of\s+range|must\s+be\s+(between|one\s+of|a?\s?positive|non-negative)",
     ErrorCategory.INPUT_INVALID),

    # Missing dependency
    (r"no\s+such\s+file|file\s+not\s+found|cannot\s+find\s+(the\s+)?file",
     ErrorCategory.MISSING_DEPENDENCY),
    (r"no\s+module\s+named|ModuleNotFoundError|ImportError",
     ErrorCategory.MISSING_DEPENDENCY),
    (r"command\s+not\s+found|not\s+recognized\s+as\s+an?\s+(internal\s+)?command",
     ErrorCategory.MISSING_DEPENDENCY),

    # Permission
    (r"permission\s+(denied|error)|access\s+denied|not\s+allowed|forbidden|401|403",
     ErrorCategory.PERMISSION),

    # Logic / runtime
    (r"TypeError|ValueError|KeyError|IndexError|AttributeError|NameError",
     ErrorCategory.LOGIC),
    (r"AssertionError|division\s+by\s+zero|null\s+pointer|undefined\s+is\s+not\s+a\s+function",
     ErrorCategory.LOGIC),

    # Resource
    (r"out\s+of\s+memory|memory\s+error|disk\s+full|quota\s+exceeded",
     ErrorCategory.RESOURCE_EXHAUSTED),
]


@dataclass
class ErrorDiagnosis:
    """Structured diagnosis of a captured error."""

    category: ErrorCategory
    original_message: str
    is_retryable: bool = False
    suggested_fix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorClassifier:
    """Classify an exception or error string into structured categories."""

    def classify(self, error: Exception | str, context: Optional[Dict[str, Any]] = None) -> ErrorDiagnosis:
        """Main entry point."""
        msg = _extract_message(error)
        category, rule_match = self._match_category(msg)
        retryable = category in (ErrorCategory.TRANSIENT,)
        suggestion = _build_suggestion(category, msg)

        return ErrorDiagnosis(
            category=category,
            original_message=msg,
            is_retryable=retryable,
            suggested_fix=suggestion,
            metadata={
                "matched_rule": rule_match,
                "context": context or {},
            },
        )

    def classify_exception(self, exc: Exception, context: Optional[Dict] = None) -> ErrorDiagnosis:
        return self.classify(exc, context)

    # ---- internal ----------------------------------------------------------

    @staticmethod
    def _match_category(msg: str) -> Tuple[ErrorCategory, str]:
        msg_lower = msg.lower()
        for pattern, category in _RULES:
            if re.search(pattern, msg_lower):
                return category, pattern
        return ErrorCategory.UNKNOWN, ""


def _extract_message(error: Exception | str) -> str:
    if isinstance(error, str):
        return error
    return f"{type(error).__name__}: {error}"


def _build_suggestion(category: ErrorCategory, msg: str) -> str:
    if category == ErrorCategory.TRANSIENT:
        return "Retry the operation after a short delay (exponential backoff)."
    if category == ErrorCategory.INPUT_INVALID:
        pf = _find_parameter_name(msg)
        if pf:
            return f"Check the value of parameter '{pf}' – it appears invalid."
        return "Verify your input arguments."
    if category == ErrorCategory.MISSING_DEPENDENCY:
        pf = _find_file_or_module(msg)
        if pf:
            return f"Ensure '{pf}' exists / is installed."
        return "Check that required files and modules are installed."
    if category == ErrorCategory.PERMISSION:
        return "Check file permissions or authentication credentials."
    if category == ErrorCategory.LOGIC:
        return "There is a logic error in the code / data. Review the traceback."
    if category == ErrorCategory.RESOURCE_EXHAUSTED:
        return "Free up memory / disk space and try again."
    return ""


def _find_parameter_name(msg: str) -> Optional[str]:
    m = re.search(r"'(\w+)'", msg)
    return m.group(1) if m else None


def _find_file_or_module(msg: str) -> Optional[str]:
    m = re.search(r"'([^']+)'", msg)
    return m.group(1) if m else None