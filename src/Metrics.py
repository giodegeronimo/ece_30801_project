# src/Metrics.py
# THIS CODE WILL HANDLE THE METRIC OBJECTS
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.Client import HFClient


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

    def _process_commits(
            self,
            commit_data: list
            ) -> tuple[dict[str, int], list[str], int, int]:
        """Process commit data to extract author information"""
        commit_count_by_author: dict[str, int] = {}
        commit_authors: list[str] = []

        for commit in commit_data:
            authors = commit.get("authors", [])
            if authors:
                for author_info in authors:
                    user = author_info.get("user", "unknown")
                    commit_authors.append(user)
                    commit_count_by_author[user] = (
                        commit_count_by_author.get(user, 0) + 1
                    )
            else:
                commit_authors.append("unknown")
                commit_count_by_author["unknown"] = (
                    commit_count_by_author.get("unknown", 0) + 1
                )

        num_commits = len(commit_data)
        unique_committers = len(set(commit_authors))

        return (
            commit_count_by_author, commit_authors,
            num_commits, unique_committers)

    def _fetch_all_commits(
            self, client, model_id, branch="main", max_pages=100):
        """Fetch all commit pages for a model repo."""
        all_commits = []
        page = 0
        while True:
            commits = client.request(
                "GET", f"/api/models/{model_id}/commits/{branch}?p={page}")
            if not commits:
                break
            all_commits.extend(commits)
            if len(commits) < 50:
                break
            page += 1
            if page >= max_pages:
                break
        return all_commits

    def _calculate_effective_maintainers(
            self, commit_count_by_author: dict[str, int]) -> float:
        """Calculate effective maintainers using inverse HHI."""
        counts = [max(0, int(v)) for v in commit_count_by_author.values()]
        total = sum(counts)
        if total > 0:
            shares = [(c / total) for c in counts if c > 0]
            hhi = sum(p * p for p in shares)
            eff = 1.0 / hhi if hhi > 0 else 0.0
        else:
            eff = 0.0
        return eff

    def compute(self, inputs: dict[str, Any], **kwargs: Any) -> float:
        if "model_url" not in inputs:
            raise ValueError("Missing required input: model_url")

        start = time.time()
        error = None

        client = HFClient(max_requests=10, token="HF_TOKEN")
        model_id = inputs["model_url"].split("https://huggingface.co/")[-1]

        # Fetch all commit pages from HF
        commit_data = self._fetch_all_commits(client, model_id)

        # Process commits to get author information
        (
            commit_count_by_author,
            commit_authors,
            num_commits,
            unique_committers,
            ) = self._process_commits(commit_data)

        # Calculate effective maintainers (inverse-HHI)
        eff = self._calculate_effective_maintainers(commit_count_by_author)

        # Normalize and clamp the score
        target = kwargs.get("target_maintainers", 5)
        try:
            target_i = int(target)
        except Exception:
            target_i = 5
        if target_i <= 1:
            target_i = 1

        score = max(0.0, min(1.0, eff / float(target_i)))
        latency_ms = (time.time() - start) * 1000

        # put all relevant info into MetricResult object
        self.result = MetricResult(
            metric="Bus Factor",
            key="bus_factor",
            value=score,
            latency_ms=latency_ms,
            details={
                "model_id": model_id,
                "effective_maintainers": eff,
                "target_maintainers": target_i,
                "commit_count_by_author": commit_count_by_author,
                "unique_committers": unique_committers,
                "num_commits": num_commits,
            },
            error=error,
        )

        return score
