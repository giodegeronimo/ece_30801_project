# src/Metrics.py
# THIS CODE WILL HANDLE THE METRIC OBJECTS
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class MetricResult:
    """
    Canonical result object returned by all metrics.

    Attributes
    ----------
    metric : str
        Human-friendly metric name (e.g., "License Check").
    key : str
        Stable identifier/slug for the metric (e.g., "license").
    value : Any
        The primary result produced by the metric (bool, str, dict, etc.).
    latency_ms : float
        How long the metric took to execute (milliseconds).
    details : Optional[Mapping[str, Any]]
        Optional extra information for display or debugging.
    error : Optional[str]
        If the metric failed, put a concise error message here
        and set `value` as appropriate.
    """
    metric: str
    key: str
    value: Any
    latency_ms: float
    details: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None


class Metric(ABC):
    """
    Abstract base class for metrics.

    Subclasses must implement ``compute()`` to perform the actual work.
    """

    @abstractmethod
    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        """
        Compute the metric score from parsed inputs.

        Parameters
        ----------
        inputs : dict[str, Any]
            Parsed inputs required by the metric.
        **kwargs : Any
            Optional per-metric tuning parameters.

        Returns
        -------
        float
            A score between 0.0 and 1.0.
        """
        raise NotImplementedError


class BusFactorMetric(Metric):
    """
    Bus factor proxy.

    Preferred input:
      - commit_count_by_author: Mapping[str, int]   # author -> commit count
        → computes "effective maintainers" via inverse-HHI (1 / Σ p_i^2).

    Fallbacks (in order):
      - commit_authors: list[str]                   # counts unique authors
      - unique_committers: int                      # direct count
      - recent_unique_committers: int               # overrides if present

    kwargs:
      - target_maintainers: int = 5   # maintainers that map to score 1.0
    """
    key: str = "bus_factor"

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        # 1) Recent window override (if provided)
        if "recent_unique_committers" in inputs:
            eff = float(self._as_int(inputs["recent_unique_committers"]))
        else:
            # 2) Best signal: inverse-HHI on commit counts by author
            ccba = inputs.get("commit_count_by_author")
            if isinstance(ccba, Mapping) and ccba:
                eff = self._effective_maints(ccba)
            else:
                # 3) Unique authors list
                authors = inputs.get("commit_authors")
                if isinstance(authors, list):
                    eff = float(
                        len(
                            {
                                a
                                for a in authors
                                if isinstance(a, str) and a}))
                else:
                    # 4) Explicit integer fallback
                    eff = float(
                        self._as_int(inputs.get("unique_committers", 0)))

        if not (eff == eff) or eff < 0.0:  # guard NaN/negative
            eff = 0.0

        target = kwargs.get("target_maintainers", 5)
        try:
            target_i = int(target)
        except Exception:
            target_i = 5
        if target_i <= 1:
            target_i = 1

        score = eff / float(target_i)
        # Clamp [0,1]
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return float(score)

    # ---- helpers ----
    @staticmethod
    def _as_int(x: Any) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    @staticmethod
    def _effective_maints(commit_count_by_author: Mapping[str, int]) -> float:
        """
        Effective maintainers = 1 / sum(p_i^2),
        p_i = commits_i / total_commits.
        One dominant author -> ~1.0; even k-way split -> ~k.
        """
        counts = [max(0, int(v)) for v in commit_count_by_author.values()]
        total = sum(counts)
        if total <= 0:
            return 0.0
        shares = [(c / total) for c in counts if c > 0]
        hhi = sum(p * p for p in shares)
        if hhi <= 0.0:
            return 0.0
        return 1.0 / hhi
