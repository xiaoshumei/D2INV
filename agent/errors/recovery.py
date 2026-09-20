"""
Error Recovery Strategies for D2INV Agent — Phase 3.

Provides intelligent, tiered recovery actions based on ErrorDiagnosis.
Integrates with ErrorClassifier and LongTermMemory.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional

from agent.errors.classifier import ErrorCategory, ErrorDiagnosis, ErrorClassifier


class RecoveryAction(Enum):
    RETRY = auto()
    CORRECT_INPUT = auto()
    FALLBACK = auto()
    ASK_USER = auto()
    ABORT = auto()
    LOG_ONLY = auto()


@dataclass
class RecoveryPlan:
    action: RecoveryAction
    diagnosis: ErrorDiagnosis
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    fallback_value: Any = None
    message: str = ""


class RecoveryExecutor:
    """
    Executes recovery plans with retry logic and exponential backoff.
    """

    def __init__(self, classifier: Optional[ErrorClassifier] = None,
                 default_max_retries: int = 3):
        self.classifier = classifier or ErrorClassifier()
        self.default_max_retries = default_max_retries

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def classify_and_plan(self, error: Exception | str,
                          context: Optional[Dict] = None) -> RecoveryPlan:
        diag = self.classifier.classify(error, context)
        return self._build_plan(diag)

    def execute_with_recovery(self, func: Callable, *args,
                              context: Optional[Dict] = None, **kwargs) -> Any:
        """Execute func(*args,**kwargs) with automatic retry and recovery."""
        plan = None
        last_error = None

        for attempt in range(1, self.default_max_retries + 2):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                diagnosis = self.classifier.classify_exception(exc, context)

                if diagnosis.is_retryable and attempt <= self.default_max_retries:
                    delay = 2 ** (attempt - 1)  # exponential backoff: 1, 2, 4...
                    print(f"[Recovery] Retry {attempt}/{self.default_max_retries} "
                          f"after {delay}s (reason: {diagnosis.category.name})")
                    time.sleep(delay)
                    continue

                plan = self._build_plan(diagnosis)
                break

        # If we exhausted retries or error is non-retryable
        if plan is None and last_error is not None:
            diagnosis = self.classifier.classify_exception(last_error, context)
            plan = self._build_plan(diagnosis)
        elif plan is None:
            plan = RecoveryPlan(
                action=RecoveryAction.ABORT,
                diagnosis=ErrorDiagnosis(
                    category=ErrorCategory.UNKNOWN,
                    original_message="Unknown failure",
                ),
                message="Unknown error state",
            )
        raise RecoveryError(plan, last_error)

    def execute_safe(self, func: Callable, *args,
                     default: Any = None, **kwargs) -> Any:
        """Safe wrapper – never raises, returns default on failure."""
        try:
            return self.execute_with_recovery(func, *args, **kwargs)
        except RecoveryError:
            return default

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _build_plan(self, diag: ErrorDiagnosis) -> RecoveryPlan:
        cat = diag.category
        if cat == ErrorCategory.TRANSIENT:
            return RecoveryPlan(action=RecoveryAction.RETRY, diagnosis=diag,
                                max_retries=3, message="Transient error – retrying.")
        if cat == ErrorCategory.INPUT_INVALID:
            return RecoveryPlan(action=RecoveryAction.CORRECT_INPUT, diagnosis=diag,
                                message="Invalid input – check parameters.")
        if cat == ErrorCategory.MISSING_DEPENDENCY:
            return RecoveryPlan(action=RecoveryAction.FALLBACK, diagnosis=diag,
                                message="Missing dependency – attempting fallback.")
        if cat == ErrorCategory.PERMISSION:
            return RecoveryPlan(action=RecoveryAction.ASK_USER, diagnosis=diag,
                                message="Permission denied – user intervention needed.")
        if cat == ErrorCategory.LOGIC:
            return RecoveryPlan(action=RecoveryAction.LOG_ONLY, diagnosis=diag,
                                message="Logic error – logging and proceeding.")
        if cat == ErrorCategory.RESOURCE_EXHAUSTED:
            return RecoveryPlan(action=RecoveryAction.ABORT, diagnosis=diag,
                                message="Resource exhausted – aborting.")
        return RecoveryPlan(action=RecoveryAction.ASK_USER, diagnosis=diag,
                            message="Unknown error – ask user.")


class RecoveryError(Exception):
    """Raised when recovery fails completely."""

    def __init__(self, plan: RecoveryPlan, original_error: Optional[Exception] = None):
        super().__init__(plan.message)
        self.plan = plan
        self.original_error = original_error