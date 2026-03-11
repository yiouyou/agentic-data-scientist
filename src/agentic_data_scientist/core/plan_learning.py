"""Learning-driven plan ranking and replay helpers (advice-first, conservative)."""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Dict, List, Optional


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_STAGE_LINE_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.+)$")
_EN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "then",
    "than",
    "were",
    "have",
    "has",
    "had",
    "will",
    "shall",
    "would",
    "could",
    "should",
    "about",
    "after",
    "before",
    "your",
    "you",
    "our",
    "their",
    "analysis",
    "stage",
    "plan",
    "task",
}


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.findall((text or "").lower()):
        token = match.strip()
        if not token:
            continue
        if token in _EN_STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        tokens.add(token)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def extract_stage_titles_from_plan(plan_text: str) -> List[str]:
    """Extract stage titles heuristically from numbered list lines."""
    titles: List[str] = []
    for line in (plan_text or "").splitlines():
        match = _STAGE_LINE_RE.match(line)
        if not match:
            continue
        title_raw = match.group(2).strip()
        title = re.sub(r"^\*\*(.+?)\*\*.*$", r"\1", title_raw).strip()
        title = title.strip("-: ")
        if title:
            titles.append(title)
    return titles


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _stage_count_score(stage_count: int) -> float:
    # Preferred window from current planning prompt guidance.
    if 3 <= stage_count <= 7:
        return 0.16
    return -min(0.16, 0.04 * abs(stage_count - 5))


def _coverage_score(user_request: str, plan_text: str) -> float:
    return 0.45 * _jaccard(_tokenize(user_request), _tokenize(plan_text))


def _historical_pattern_score(plan_text: str, history_signals: Dict[str, Any]) -> float:
    score = 0.0
    plan_tokens = _tokenize(plan_text)

    topk = history_signals.get("topk_similar_runs", [])
    if isinstance(topk, list):
        similarities = []
        for run in topk[:5]:
            stage_titles = run.get("stage_titles", [])
            joined = " ".join(stage_titles) if isinstance(stage_titles, list) else str(stage_titles or "")
            sim = _jaccard(plan_tokens, _tokenize(joined))
            if sim > 0:
                similarities.append(sim)
        if similarities:
            score += 0.22 * (sum(similarities) / len(similarities))

    workflows = history_signals.get("hot", {}).get("top_workflows", [])
    if isinstance(workflows, list):
        wf_bonus = 0.0
        lower_plan = (plan_text or "").lower()
        for item in workflows[:3]:
            wf_id = str(item.get("workflow_id", "")).strip().lower()
            if not wf_id:
                continue
            if wf_id in lower_plan:
                success = _safe_float(item.get("success_rate"), 0.0)
                wf_bonus += 0.04 + min(0.04, 0.04 * success)
        score += min(0.14, wf_bonus)

    return score


def _dag_validity_score(parsed_stages: List[Dict[str, Any]]) -> float:
    """Check that ``depends_on`` references form a valid DAG (no cycles, no dangling refs).

    Returns 0.10 for a valid DAG, 0.0 for empty/missing dependency data,
    and -0.10 for cycles or invalid references.
    """
    if not parsed_stages:
        return 0.0

    stage_ids = {s.get("stage_id", f"s{s.get('index', i) + 1}") for i, s in enumerate(parsed_stages)}
    if not stage_ids:
        return 0.0

    has_any_deps = False
    for stage in parsed_stages:
        deps = stage.get("depends_on", [])
        if not isinstance(deps, list):
            continue
        for dep in deps:
            has_any_deps = True
            if dep not in stage_ids:
                return -0.10

    if not has_any_deps:
        return 0.0

    # Topological-sort cycle detection via Kahn's algorithm
    adj: Dict[str, List[str]] = {sid: [] for sid in stage_ids}
    in_degree: Dict[str, int] = {sid: 0 for sid in stage_ids}
    for stage in parsed_stages:
        sid = stage.get("stage_id", f"s{stage.get('index', 0) + 1}")
        for dep in stage.get("depends_on", []):
            if dep in adj:
                adj[dep].append(sid)
                in_degree[sid] = in_degree.get(sid, 0) + 1

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbour in adj.get(node, []):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if visited < len(stage_ids):
        return -0.10

    return 0.10


def _dataflow_coverage_score(parsed_stages: List[Dict[str, Any]]) -> float:
    """Check that every ``inputs_required`` is satisfied by some prior stage's ``outputs_produced``.

    Returns 0.10 for full coverage, scaled down proportionally for partial coverage,
    and 0.0 if no dataflow metadata is present.
    """
    if not parsed_stages:
        return 0.0

    available_outputs: set[str] = set()
    total_inputs = 0
    satisfied_inputs = 0

    for stage in parsed_stages:
        inputs_req = stage.get("inputs_required", [])
        if isinstance(inputs_req, list):
            for inp in inputs_req:
                inp_lower = str(inp).strip().lower()
                if not inp_lower:
                    continue
                total_inputs += 1
                if inp_lower in available_outputs:
                    satisfied_inputs += 1

        outputs_prod = stage.get("outputs_produced", [])
        if isinstance(outputs_prod, list):
            for out in outputs_prod:
                out_lower = str(out).strip().lower()
                if out_lower:
                    available_outputs.add(out_lower)

    if total_inputs == 0:
        return 0.0

    coverage = satisfied_inputs / total_inputs
    return 0.10 * coverage


def _granularity_uniformity_score(parsed_stages: List[Dict[str, Any]]) -> float:
    """Penalize plans where stage descriptions vary wildly in length (proxy for uneven granularity).

    Returns 0.06 for uniform stages, decreasing toward 0.0 for high variance.
    Uses coefficient of variation (stdev / mean) of description lengths.
    """
    if len(parsed_stages) < 2:
        return 0.06

    lengths = []
    for stage in parsed_stages:
        desc = stage.get("description", "") or ""
        lengths.append(len(desc))

    if not lengths or all(l == 0 for l in lengths):
        return 0.0

    mean_len = statistics.mean(lengths)
    if mean_len == 0:
        return 0.0

    stdev_len = statistics.stdev(lengths)
    cv = stdev_len / mean_len

    # cv < 0.5 → full bonus; cv > 2.0 → zero bonus
    if cv <= 0.5:
        return 0.06
    if cv >= 2.0:
        return 0.0
    return 0.06 * (1.0 - (cv - 0.5) / 1.5)


def score_plan_candidate(
    *,
    user_request: str,
    candidate_plan: str,
    history_signals: Dict[str, Any],
    is_baseline: bool,
    parsed_stages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score one candidate with conservative priors."""
    stage_titles = extract_stage_titles_from_plan(candidate_plan)
    stage_count = len(stage_titles)

    score = 0.0
    score += _coverage_score(user_request, candidate_plan)
    score += _stage_count_score(stage_count)
    score += _historical_pattern_score(candidate_plan, history_signals)

    structural_scores: Dict[str, float] = {}
    if parsed_stages:
        dag_score = _dag_validity_score(parsed_stages)
        dataflow_score = _dataflow_coverage_score(parsed_stages)
        granularity_score = _granularity_uniformity_score(parsed_stages)
        score += dag_score + dataflow_score + granularity_score
        structural_scores = {
            "dag_validity": dag_score,
            "dataflow_coverage": dataflow_score,
            "granularity_uniformity": granularity_score,
        }

    hot = history_signals.get("hot", {})
    run_count = int(hot.get("run_count", 0) or 0)
    retry_rate = _safe_float(hot.get("stage_retry_rate"), 0.0)
    if run_count < 10:
        score -= 0.03
    score -= min(0.08, 0.12 * max(0.0, retry_rate))

    if is_baseline:
        score += 0.08

    score = max(-1.0, min(1.0, score))

    result: Dict[str, Any] = {
        "score": score,
        "stage_count": stage_count,
        "stage_titles": stage_titles[:6],
    }
    if structural_scores:
        result["structural_scores"] = structural_scores
    return result


def rank_plan_candidates(
    *,
    user_request: str,
    candidates: List[str],
    history_signals: Dict[str, Any],
    baseline_index: int = 0,
    min_switch_margin: float = 0.12,
) -> Dict[str, Any]:
    """
    Rank candidate plans using historical signals.

    The policy is conservative and advice-only:
    - choose highest score candidate
    - keep baseline when improvement margin is below threshold
    """
    if not candidates:
        return {
            "selected_index": 0,
            "baseline_index": baseline_index,
            "candidate_scores": [],
            "switch_applied": False,
            "reason": "no_candidates",
        }

    safe_baseline = min(max(0, int(baseline_index)), len(candidates) - 1)
    scored: List[Dict[str, Any]] = []

    for idx, candidate in enumerate(candidates):
        candidate_text = str(candidate or "").strip()
        if not candidate_text:
            scored.append(
                {
                    "index": idx,
                    "score": -1.0,
                    "stage_count": 0,
                    "stage_titles": [],
                }
            )
            continue

        detail = score_plan_candidate(
            user_request=user_request,
            candidate_plan=candidate_text,
            history_signals=history_signals,
            is_baseline=(idx == safe_baseline),
        )
        scored.append({"index": idx, **detail})

    best = max(scored, key=lambda item: item["score"])
    baseline = scored[safe_baseline]
    margin = float(best["score"]) - float(baseline["score"])

    selected_index = safe_baseline
    switch_applied = False
    reason = "baseline_kept_conservative"
    if best["index"] != safe_baseline and margin >= float(min_switch_margin):
        selected_index = int(best["index"])
        switch_applied = True
        reason = "switched_by_margin"
    elif best["index"] == safe_baseline:
        reason = "baseline_already_best"

    return {
        "selected_index": selected_index,
        "baseline_index": safe_baseline,
        "candidate_scores": scored,
        "switch_applied": switch_applied,
        "margin_vs_baseline": margin,
        "min_switch_margin": float(min_switch_margin),
        "reason": reason,
    }


def replay_selection_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute an offline counterfactual replay summary from recorded selection traces.

    This is an approximate sanity-check report (not causal inference):
    - observed reward: derived from actual run outcomes
    - policy gain proxy: selected score - baseline score
    """
    if not records:
        return {
            "records": 0,
            "switch_rate": 0.0,
            "avg_observed_reward": 0.0,
            "avg_policy_gain_proxy": 0.0,
            "avg_observed_reward_when_switched": 0.0,
            "note": "no_records",
        }

    total = len(records)
    switched = 0
    sum_reward = 0.0
    sum_gain_proxy = 0.0
    switched_rewards: List[float] = []

    for item in records:
        if bool(item.get("switch_applied", False)):
            switched += 1
        reward = _safe_float(item.get("observed_reward"), 0.0)
        gain = _safe_float(item.get("policy_gain_proxy"), 0.0)
        sum_reward += reward
        sum_gain_proxy += gain
        if bool(item.get("switch_applied", False)):
            switched_rewards.append(reward)

    avg_switched_reward = sum(switched_rewards) / len(switched_rewards) if switched_rewards else 0.0
    return {
        "records": total,
        "switch_rate": float(switched) / float(total),
        "avg_observed_reward": sum_reward / float(total),
        "avg_policy_gain_proxy": sum_gain_proxy / float(total),
        "avg_observed_reward_when_switched": avg_switched_reward,
        "note": "counterfactual_is_proxy_not_causal",
    }
