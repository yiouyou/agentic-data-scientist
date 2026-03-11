# Planning Strategy Evolution

Inspired by arXiv:2603.04735v1 (Neuro-Symbolic AI Discovery with Tree Search + Automated Verification).
Adapted for general-purpose deep research with scientific skills.

---

## 1. Current Architecture Snapshot

### 1.1 Planning Pipeline

```
User Query
  → plan_maker (generates natural-language plan)
  → plan_reviewer (critiques plan)
  → plan_review_confirmation (decides exit/retry)
  → [loop up to 10 iterations]
  → plan_candidate_selector (ranks candidates, keeps baseline unless margin > 0.12)
  → plan_parser (structures plan → stages[] + success_criteria[])
  → stage_orchestrator (executes stages one by one)
      per stage:
        → implementation_loop (coding → review → review_confirmation, loop escalates on exit=true)
        → criteria_checker (LLM inspects files, updates met/evidence)
        → stage_reflector (may modify/add stages)
  → summary_agent
```

### 1.2 Key Numerical Limits

| Parameter | Value | Location |
|-----------|-------|----------|
| Planning loop max iterations | 10 | `agent.py:896` |
| Plan candidates max | 8 | `plan_selector.py` |
| Min switch margin (keep baseline) | 0.12 | `ADS_PLAN_RANK_MIN_SWITCH_MARGIN` |
| Orchestrator max iterations | 50 | `stage_orchestrator.py:294` |
| Stage retry (not approved) | unlimited within 50 iterations | orchestrator continues loop |
| Typical stage count | 3–7 | `plan_parser.md:19` guidance, not enforced |
| Event compression threshold | 40 events | multiple locations |
| Coding event hard limit | 100 events | `implementation_loop.py:38` |

### 1.3 Stage Record Schema

```python
{
    "index": int,
    "stage_id": "s1",
    "title": str,
    "description": str,
    "completed": bool,
    "status": "pending|in_progress|approved|retrying|failed",
    "implementation_result": str,
    "depends_on": ["s0"],          # DAG dependencies
    "inputs_required": [...],       # expected input artifacts
    "outputs_produced": [...],      # output artifacts
    "evidence_refs": [...],         # evidence references
    "subtasks": [...],              # sub-decomposition (field exists, unused)
}
```

### 1.4 Success Criterion Record

```python
{
    "index": int,
    "criteria": str,       # human-readable criterion
    "met": bool,           # updated by criteria_checker
    "evidence": str | None # file paths, metrics cited by LLM
}
```

### 1.5 Current Verification Method

criteria_checker is a **pure LLM agent** with file inspection tools (read_file, list_directory, etc.).
It reads files in the working directory and subjectively judges whether each criterion is met.
No programmatic/numerical verification exists.

### 1.6 Current Adaptive Mechanisms

- **stage_reflector**: can modify uncompleted stage descriptions or append new stages after each stage completion.
  Prompt instructs "be conservative", "only add if truly necessary".
- **plan_candidate_selector**: collects plans during planning loop, ranks by jaccard coverage + history + stage count. Conservative: keeps baseline unless challenger exceeds margin.
- **No sub-stage decomposition**: `subtasks` field exists in schema but is never populated or used.
- **No backtracking**: failed/retrying stages are re-attempted linearly, no rollback to alternative approaches.

---

## 2. Identified Improvement Strategies

Source: arXiv:2603.04735v1 methodology, adapted for general deep research.

### Strategy A: Programmatic Verification (P0)

**Insight**: "Everything that can be verified programmatically should be verified programmatically."

The paper's automated numerical feedback caught >80% of errors. Currently criteria_checker relies entirely on LLM judgment — subjective, hallucination-prone, and unable to run code.

**Current gap**:
- criteria_checker uses file tools (read_file, list_directory) but cannot execute code
- Verification is subjective: LLM reads file content and "decides" if criterion is met
- No numerical comparison, no schema validation, no existence checks beyond what LLM chooses to do

**Proposed approach**:
Build a verification layer that runs **before** LLM criteria_checker, providing factual signals:

```
Tier 1 (Programmatic, no LLM):
  - File existence checks: do expected output files exist?
  - Schema validation: CSV has expected columns? JSON parses correctly?
  - Numerical thresholds: AUC > 0.85? p-value < 0.05? (extract from result files)
  - Image validation: figure files are non-zero size, correct format?
  - Code execution success: scripts ran without errors?

Tier 2 (LLM-assisted, with Tier 1 signals):
  - criteria_checker receives Tier 1 results as additional context
  - LLM focuses on qualitative judgments that truly need reasoning
  - "Programmatic checks passed: [file_exists=true, auc=0.91]. Now assess analytical quality."
```

**Integration point**: New callback or agent between implementation_loop and criteria_checker in stage_orchestrator (lines 535–571). Or inject Tier 1 results into criteria_checker prompt context.

**Risks**: Low. Additive, does not replace existing criteria_checker.

---

### Strategy B: Negative Prompting for Plan Diversity (P0)

**Insight**: "Generating genuinely different approaches, not iterative refinements of the same idea."

The paper used negative prompting to force discovery of 6 different solution methods. Current planning loop produces iterative refinements of a single approach — reviewer gives feedback, plan_maker adjusts.

**Current gap**:
- Planning loop (max 10 iterations) exits as soon as reviewer approves
- All iterations refine the SAME plan, not generate alternatives
- plan_candidate_selector collects variants but they are incremental revisions, not diverse strategies

**Proposed approach**:
After reviewer approves Plan A, don't exit immediately. Instead:

```
Round 1: plan_maker → reviewer approves → Candidate A ✓ (store)
Round 2: plan_maker + negative prompt:
         "A valid plan has been approved (summary: {Plan_A_summary}).
          Generate a FUNDAMENTALLY DIFFERENT analytical strategy.
          Do NOT use {key_methods_from_A}.
          Explore alternative methodologies, different analytical frameworks,
          or different decomposition of the problem."
         → reviewer evaluates → Candidate B ✓ (store)
Round 3: (optional, if budget allows) → Candidate C ✓ (store)
Final:   plan_candidate_selector ranks A, B, C → pick best
```

**Why negative prompting > "list 3 strategies in one prompt"**:
- Separate generation rounds with explicit exclusion produce more divergent ideas
- Single-prompt multi-strategy tends to produce superficial variants (same core, different labels)
- Each candidate gets full reviewer scrutiny independently

**Integration point**: Modify planning loop exit logic in agent.py (line 876–897). After first `exit=true`, store candidate and inject negative prompt for next iteration instead of exiting.

**Risks**: Medium. 2–3x planning cost (but planning is ~10% of total cost). Requires careful prompt engineering to ensure negative constraints don't degrade quality.

---

### Strategy C: Adaptive Depth — Dynamic Stage Decomposition (P1)

**Insight**: "Research depth should be non-uniform. Some steps deserve deeper exploration; depth and breadth should emerge from discoveries, not be fixed upfront."

The paper's tree search naturally allocated more exploration to promising branches. Current system has fixed-depth stages: each stage gets one implementation_loop pass, no sub-decomposition.

**Current gap**:
- `subtasks` field exists in stage record but is never used
- All stages are treated equally: one implementation_loop pass each
- No mechanism to say "this stage is unexpectedly complex, decompose it further"
- stage_reflector can modify descriptions and add stages, but not decompose a stage into sub-stages
- No backtracking: if Stage 3 reveals Stage 2's approach was wrong, no mechanism to revisit

**Proposed approach (phased)**:

Phase 1 — **Stage Decomposition on Complexity Signal**:
```
After implementation_loop attempt for a stage:
  IF review says "too complex" or "partially completed" or attempt > 1:
    → Invoke a "stage_decomposer" agent
    → Split current stage into 2-3 sub-stages
    → Insert sub-stages into the stage list (replacing original)
    → Continue orchestration with finer granularity
```

Phase 2 — **Backtracking on Discovery**:
```
After stage_reflector identifies a fundamental issue:
  IF reflector says "Stage N's approach is invalidated by findings":
    → Mark Stage N for re-implementation with new approach
    → Inject negative prompt: "Previous approach was {old_approach}. It failed because {reason}. Use a different method."
    → This combines Strategy B (negative prompting) at the stage level
```

Phase 3 — **Tree-like Exploration** (future, if Phase 1+2 insufficient):
```
For high-stakes stages:
  → Generate 2 implementation approaches in parallel
  → Run lightweight validation on both
  → Continue with the one that shows better intermediate results
```

**Integration point**: 
- Phase 1: Add decomposition logic in stage_orchestrator after retry detection (lines 477–514)
- Phase 2: Extend stage_reflector prompt and callback to support "re-approach" directive
- Phase 3: Requires parallel execution capability (not currently supported)

**Risks**: 
- Phase 1: Low-medium. Well-scoped.
- Phase 2: Medium. Needs careful state management to avoid infinite loops.
- Phase 3: High. Architectural change.

---

### Strategy D: Hierarchical Verification & Model Escalation (P1)

**Insight**: "Use fast/cheap models for exploration, strong models for verification and refinement."

The paper used standard Gemini for tree search, then a more advanced model for deep verification and simplification.

**Current gap**:
- Model routing is role-based (plan_maker, reviewer, coding, etc.), not stage-aware
- All stages of the same type use the same model regardless of complexity
- No "escalation" mechanism: if a stage fails, it retries with the same model

**Proposed approach**:

```
Stage-Aware Model Selection:
  plan_parser annotates stages with complexity_hint: "standard" | "complex" | "critical"

  During execution:
    "standard" stages → default coding model
    "complex" stages → stronger model (e.g., gpt-5.2)
    "critical" stages (final integration, report) → strongest available

  On retry (attempt > 1):
    Automatically escalate to next-tier model
    "If the default model couldn't do it in one try, try a stronger model"
```

```
Deep Verification Pass:
  After all stages complete, before summary_agent:
    → deep_verifier agent (using strongest model)
    → Checks: do results truly satisfy all success criteria?
    → Checks: are there logical inconsistencies across stages?
    → If issues found: inject correction stage and re-enter orchestration
```

**Integration point**: 
- Stage complexity: extend plan_parser output schema with `complexity_hint`
- Model escalation: add retry-aware model selection in stage_orchestrator
- Deep verifier: new agent between stage_orchestrator and summary_agent in agent.py workflow

**Risks**: Low. Additive. Token cost increase is bounded.

---

### Strategy E: Enhanced Plan Scoring (P0)

**Insight**: "Automated scoring with multiple dimensions enables better pruning."

The paper's PUCT scoring balanced exploration and exploitation across 600 nodes. Current plan_selector has a basic scoring formula with limited dimensions.

**Current scoring formula** (plan_learning.py):
```
score = 0.45 * jaccard_coverage
      + stage_count_bonus (0.05 for 3-7 stages)
      + historical_pattern_score
      - 0.03 * cold_start_penalty
      - 0.12 * retry_rate
      + 0.08 * baseline_bonus
```

**Proposed additional dimensions**:

| Dimension | Signal | Cost |
|-----------|--------|------|
| **Dependency graph validity** | Is depends_on a valid DAG? No cycles? All refs exist? | Zero (pure computation) |
| **Data flow coverage** | Do outputs_produced cover downstream inputs_required? | Zero (pure computation) |
| **Skill coverage** | What fraction of needed capabilities are in skill_registry? | Zero (lookup) |
| **Granularity uniformity** | Variance of stage description lengths. Very uneven → penalty | Zero (string length) |
| **Method diversity** | If Strategy B is active: how different is this plan from others? | Low (jaccard on method terms) |

**Integration point**: Extend `rank_plan_candidates()` in plan_learning.py. All new dimensions are computable without LLM calls.

**Risks**: Minimal. Pure scoring extension.

---

## 3. Implementation Priority

| Priority | Strategy | Effort | Risk | Impact |
|----------|----------|--------|------|--------|
| **P0** | A: Programmatic Verification | Small-Medium | Low | High — eliminates hallucination in verification |
| **P0** | B: Negative Prompting | Medium | Medium | High — fundamentally different analysis approaches |
| **P0** | E: Enhanced Plan Scoring | Small | Minimal | Medium — better plan selection with no LLM cost |
| **P1** | C Phase 1: Stage Decomposition | Medium | Low-Medium | High — adaptive depth for complex stages |
| **P1** | D: Model Escalation | Small-Medium | Low | Medium — better resource allocation |
| **P2** | C Phase 2: Backtracking | Medium-Large | Medium | High — but complex to implement safely |
| **P3** | C Phase 3: Tree Exploration | Large | High | Variable — may not justify cost for data science tasks |

### Recommended Implementation Order

```
Sprint 1: E (scoring) + A (verification Tier 1)
  → Quick wins, no LLM cost increase, immediate quality improvement

Sprint 2: B (negative prompting) + A (verification Tier 2)
  → Plan diversity + LLM-assisted verification with programmatic context

Sprint 3: D (model escalation) + C Phase 1 (stage decomposition)
  → Adaptive depth and model routing

Sprint 4+: C Phase 2 (backtracking), C Phase 3 (tree exploration)
  → Only if Sprint 1-3 results are insufficient
```

---

## 4. Key Design Principles

Derived from the paper's methodology, adapted for research tasks:

1. **Verify early, verify often**: Don't wait until the end to check. Programmatic checks at every stage boundary.

2. **Diversity over refinement**: Multiple genuinely different approaches beat polishing one approach. Use negative prompting to force divergence.

3. **Adaptive depth**: Not all stages deserve equal depth. Decompose complex stages; don't over-plan simple ones. Let discoveries during execution drive depth decisions.

4. **Match model to task**: Exploration can use fast/cheap models. Verification and synthesis need the strongest available.

5. **Programmatic > LLM for factual checks**: If it can be checked by running code or inspecting files, don't ask an LLM. Reserve LLM judgment for qualitative assessment.

6. **Backtracking is not failure**: Discovering that an approach doesn't work is valuable information. The system should be able to try a different approach (negative prompting at stage level) rather than repeatedly retrying the same failing strategy.

---

## 5. Open Questions for Discussion

- [ ] Strategy B: How many alternative plans to generate? 2 (minimum diversity) or 3 (more coverage, higher cost)?
- [ ] Strategy A: Should programmatic verification be mandatory (block on failure) or advisory (pass signals to LLM)?
- [ ] Strategy C: What triggers stage decomposition? LLM review saying "too complex"? Attempt count > N? Stage description length?
- [ ] Strategy D: How to annotate stage complexity? LLM during parsing? Heuristic (description length, keyword detection)?
- [ ] Cost budget: Is there a maximum acceptable cost multiplier from these improvements? (Current: 1x → with all strategies: ~3-4x for planning, ~1.2x for execution)
- [ ] Strategy C Phase 2: How to prevent infinite backtracking loops? Max backtrack count? Negative prompt must exclude ALL previously tried approaches?

---

## Appendix: Source Paper Summary

**Paper**: "Solving an Open Problem in Theoretical Physics using AI-Assisted Discovery" (arXiv:2603.04735v1)

**Core system**: Gemini Deep Think + Tree Search (PUCT) + Automated Numerical Feedback

**Key results**:
- Explored ~600 candidate nodes in search tree
- Automated verifier pruned >80% of branches (algebraic errors, numerical divergence)
- Negative prompting forced discovery of 6 fundamentally different solution methods
- Most elegant solution (Gegenbauer method) was the 6th discovered — would be missed by single-path search
- Hierarchical verification: automated search → human-guided deep verification with stronger model

**Transferable principles** (not tied to mathematical domain):
1. Every-step verification with automated feedback
2. Multi-path exploration with diversity enforcement
3. Scoring and pruning to focus resources on promising paths
4. Hierarchical refinement: fast exploration → deep verification
5. Negative prompting to escape local optima in solution space
