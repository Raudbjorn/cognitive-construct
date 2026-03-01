# Volition Technical Specification

> "Between stimulus and response there is a space. In that space is our power
> to choose our response. In our response lies our growth and our freedom."
> — Viktor Frankl

**Version:** 0.1.0-draft
**Status:** Design
**Depends on:** `shared.embeddings`, `shared.fusion`, `shared.events`, `shared.membrane`, `shared.synergies`, `shared.feedback`

---

## 1. Problem Statement

### 1.1 The Reactivity Trap

LLM-based agents are reactive systems. A prompt arrives, the model pattern-matches
against it, and a response fires. There is no intermediate step where the agent
asks itself: *"Do I understand what is being asked? Am I confident enough to act?
What happens if I'm wrong?"*

This is the difference between a reflex arc and executive function. A reflex arc
produces fast, brittle responses. Executive function produces deliberate,
recoverable action.

Volition today is a reflex arc. Its `_classify_intent()` function
(`volition/scripts/volition.py:282-294`) counts keyword hits against four
hard-coded categories and picks the winner. "Refactor the authentication module"
matches `code_edit` because "refactor" is in the keyword list. "Make the auth
module more secure" matches `security` because "security" is in the keyword list
-- and routes to Shodan, which is categorically wrong.

The cost of misclassification is not a bad search result. It is a **wrong action**:
an LLM call when the user wanted a code edit, a Shodan scan when the user wanted
a code review. Volition is the only skill that *does things to the world*. Every
other skill produces information. Volition produces consequences.

### 1.2 What This Specification Addresses

This document specifies the transformation of Volition from a keyword-counting
router into a genuine executive function with:

- **Semantic intent classification** that understands meaning, not keywords
- **Confidence gating** that asks instead of guessing when uncertain
- **Multi-step action plans** for intents that require coordination
- **Pre-flight validation** that catches bad plans before they execute
- **Fallback chains** that recover from backend failures gracefully
- **Outcome learning** that improves routing over time from audit logs

### 1.3 What This Specification Does Not Address

- The internal implementation of backend MCP servers (Serena, cross-llm, etc.)
- The Claude Code skill framework itself (SKILL.md, frontmatter, commands)
- Real-time streaming or interrupt handling during action execution
- Multi-user concurrency (Volition operates in single-user CLI context)

---

## 2. Requirements

### 2.1 User Stories

**US-1: Semantic intent routing**
*As a developer, when I say "make the auth module more secure," I want Volition
to route to code editing (Serena), not to Shodan security scanning.*

EARS acceptance:
> WHEN the user provides a natural-language action description,
> the system SHALL classify intent using embedding similarity fused with
> keyword signals, and SHALL route to the handler whose intent prototype
> has the highest fused score above the confidence threshold.

**US-2: Confidence gating**
*As a developer, when Volition is not sure what I mean, I want it to ask me
instead of guessing wrong.*

EARS acceptance:
> WHEN intent classification produces no candidate above the confidence
> threshold (default 0.65), the system SHALL return a clarification request
> listing the top-2 candidates with their scores, instead of executing an action.

**US-3: Multi-step plans**
*As a developer, when I say "review the auth module and fix any issues," I want
Volition to first consult an LLM for review, then apply code edits based on
the findings -- not pick one handler and lose the other half of my intent.*

EARS acceptance:
> WHEN intent classification identifies multiple action types in a single
> request, the system SHALL construct an ActionPlan DAG with ordered steps
> and data dependencies, and SHALL execute steps in topological order.

**US-4: Pre-flight validation**
*As a developer, I want Volition to catch obviously bad plans (e.g., editing
a symbol that doesn't exist, querying Shodan without confirmation) before
attempting to execute them.*

EARS acceptance:
> BEFORE executing any action plan, the system SHALL run all four pre-flight
> validation passes. IF any pass produces a CRITICAL flag, the system SHALL
> abort execution and report the flags to the user.

**US-5: Fallback chains**
*As a developer, when Serena is unavailable, I want Volition to fall back to
text-based editing rather than failing outright.*

EARS acceptance:
> WHEN a backend handler returns an availability error, the system SHALL
> attempt the next handler in the fallback chain for that capability.
> IF all fallbacks are exhausted, the system SHALL report the failure with
> the original error and all attempted fallbacks.

**US-6: Outcome learning**
*As a developer, I want Volition's routing to improve over time as it learns
which backends produce useful results for which query types.*

EARS acceptance:
> WHEN an action completes, the system SHALL record the outcome in the audit
> log. WHEN classifying future intents, the system SHALL incorporate feedback
> scores from `shared.feedback.FeedbackCollector` to adjust handler weights.

**US-7: Rhetoric pre-check**
*As a developer, when I request a consequential action (security scan, bulk
edit), I want Volition to optionally run a Rhetoric deliberation before acting.*

EARS acceptance:
> WHEN an action plan includes a step with `risk_level >= HIGH`, and the
> `RHETORIC_PREFLIGHT` feature flag is enabled, the system SHALL request
> a Rhetoric deliberation via the synergy bus before proceeding.

### 2.2 Scope

**In scope:**
- Intent classification engine (embedding + keyword fusion)
- Action plan construction and execution
- Pre-flight validation engine (4-pass)
- Fallback chain management
- Feedback-weighted routing
- Integration with existing `shared.*` infrastructure

**Out of scope (this version):**
- Interactive multi-turn plan negotiation
- Parallel step execution within a plan
- User-defined custom handlers
- Natural language plan editing ("actually, skip the review step")

### 2.3 Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-1 | Latency | Intent classification SHALL complete in < 50ms (Model2Vec inference + RRF fusion) |
| NFR-2 | Latency | Pre-flight validation SHALL complete in < 20ms (pure symbolic, no I/O) |
| NFR-3 | Availability | Classification SHALL degrade to keyword-only if embedding model is unavailable |
| NFR-4 | Auditability | Every action plan and its outcome SHALL be recorded in `~/.volition/audit.log` |
| NFR-5 | Determinism | Given identical input, classification scores SHALL be identical (no randomness) |
| NFR-6 | Safety | Security-category actions SHALL always require explicit confirmation (R.22.1) |
| NFR-7 | Memory | Embedding model footprint SHALL not exceed 16MB resident (Model2Vec: ~8MB) |

---

## 3. Design

### 3.1 Architecture Overview

```
                     user prompt
                          |
                          v
                +---------+----------+
                |  Intent Classifier  |
                | (embedding + kw)    |
                +---------+----------+
                          |
               confidence >= threshold?
                    /           \
                  yes            no
                   |              |
                   v              v
          +--------+------+  +---+-------------+
          |  Plan Builder  |  | Clarification   |
          |  (DAG steps)   |  | Request (top-2) |
          +--------+------+  +-----------------+
                   |
                   v
          +--------+----------+
          |  Pre-flight        |
          |  Validator (4-pass)|
          +--------+----------+
                   |
              all passes OK?
                /        \
              yes         no
               |           |
               v           v
      +--------+------+  flags → user
      |  Executor      |
      |  (step by step)|
      +--------+------+
               |
               v
      +--------+----------+
      |  Outcome Logger    |
      |  (audit + feedback)|
      +--------+----------+
               |
               v
          result → user
```

### 3.2 Data Structures

#### 3.2.1 IntentPrototype

Each handler registers a prototype: a short natural-language description of what
it does, plus keyword signals for hybrid scoring.

```json
{
  "handler": "code_edit",
  "prototype_text": "Edit, refactor, modify, or fix source code symbols and files using LSP-powered semantic editing",
  "keywords": ["refactor", "edit", "modify", "add", "remove", "fix", "update", "change", "rename", "extract"],
  "risk_level": "MEDIUM",
  "fallback_chain": ["text_edit"],
  "requires_confirmation": false
}
```

**Design rationale:** The `prototype_text` is embedded once at startup and
cached. At classification time, the user's action is embedded and compared
against all prototypes via cosine similarity. Keywords provide a secondary
signal fused via RRF. This makes classification robust to paraphrasing
("improve security" matches `code_edit` because the prototype embedding is
close) while still fast on exact keyword hits.

#### 3.2.2 ClassificationResult

```json
{
  "action": "make the auth module more secure",
  "candidates": [
    {"handler": "code_edit",   "embedding_score": 0.82, "keyword_score": 0.14, "fused_score": 0.71},
    {"handler": "llm_call",    "embedding_score": 0.45, "keyword_score": 0.00, "fused_score": 0.31},
    {"handler": "security",    "embedding_score": 0.38, "keyword_score": 0.43, "fused_score": 0.29},
    {"handler": "web_search",  "embedding_score": 0.21, "keyword_score": 0.00, "fused_score": 0.14}
  ],
  "selected": "code_edit",
  "confidence": 0.71,
  "above_threshold": true,
  "feedback_adjustment": 0.03
}
```

**Design rationale:** Returning all candidates with decomposed scores serves
three purposes: (1) the user can see *why* a handler was selected in
`--verbose` mode, (2) the clarification request uses top-2 candidates, and
(3) the audit log records the full decision for outcome learning.

#### 3.2.3 ActionPlan

A plan is a directed acyclic graph of steps. Each step has an explicit handler,
input bindings (which may reference outputs of prior steps), and a fallback chain.

```json
{
  "plan_id": "plan-a1b2c3d4",
  "created": "2026-03-01T14:30:00Z",
  "original_action": "review the auth module and fix any issues",
  "steps": [
    {
      "step_id": "step-1",
      "handler": "llm_call",
      "action": "Review the authentication module for security issues",
      "inputs": {},
      "depends_on": [],
      "risk_level": "LOW",
      "fallback_chain": []
    },
    {
      "step_id": "step-2",
      "handler": "code_edit",
      "action": "Apply fixes for issues identified in step-1",
      "inputs": {"review_findings": "step-1.output"},
      "depends_on": ["step-1"],
      "risk_level": "MEDIUM",
      "fallback_chain": ["text_edit"]
    }
  ],
  "validation": {
    "status": "valid",
    "flags": []
  }
}
```

**Design rationale:** The DAG structure handles dependency ordering naturally.
`inputs` with step references create data flow edges. The executor processes
steps in topological order, substituting output references with actual values.
This is deliberately simple -- no conditionals, no loops, no parallel execution.
Those are Phase 2 concerns.

#### 3.2.4 PreflightResult

```json
{
  "plan_id": "plan-a1b2c3d4",
  "status": "flagged",
  "passes": {
    "capability_check": {"status": "pass", "flags": []},
    "input_validation": {"status": "pass", "flags": []},
    "risk_assessment": {"status": "flagged", "flags": [
      {
        "type": "unconfirmed_security_action",
        "severity": "CRITICAL",
        "step_id": "step-3",
        "description": "Security query requires --confirm flag",
        "remediation": "Add --confirm flag or remove the security step"
      }
    ]},
    "dependency_check": {"status": "pass", "flags": []}
  }
}
```

#### 3.2.5 ActionOutcome

```json
{
  "plan_id": "plan-a1b2c3d4",
  "step_id": "step-1",
  "handler": "llm_call",
  "status": "success",
  "duration_ms": 2340,
  "output_summary": "Found 3 issues: missing rate limiting, weak password hashing, no CSRF protection",
  "fallbacks_attempted": [],
  "feedback_recorded": false
}
```

### 3.3 Intent Classification Engine

The classifier replaces the keyword-counting `_classify_intent()` function
(`volition/scripts/volition.py:282-294`) with a two-signal fusion system.

#### Signal 1: Embedding Similarity

Uses `shared.embeddings.rank_by_relevance()` with Model2Vec
(`minishlab/potion-base-8M`, ~8MB, microsecond inference).

At startup, each `IntentPrototype.prototype_text` is embedded and cached.
At classification time, the user's action is embedded once, and cosine similarity
is computed against all prototype embeddings.

```python
from shared.embeddings import rank_by_relevance, ScoredItem

prototypes: list[IntentPrototype] = load_prototypes()
scored: list[ScoredItem] = rank_by_relevance(
    items=prototypes,
    query=user_action,
    key=lambda p: p.prototype_text,
)
# scored[0].score is the highest embedding similarity
```

#### Signal 2: Keyword Scoring

The existing keyword lists are preserved as a secondary signal. Each keyword
hit contributes a fractional score: `1.0 / len(keywords_for_handler)`.

```python
def keyword_score(action: str, prototype: IntentPrototype) -> float:
    action_lower = action.lower()
    hits = sum(1 for kw in prototype.keywords if kw in action_lower)
    return hits / len(prototype.keywords) if prototype.keywords else 0.0
```

#### Fusion

The two signals are fused using `shared.fusion.RRFEngine` with per-signal
weights derived from `FusionStrategy`:

```python
from shared.fusion import RRFEngine, RankedItem

engine = RRFEngine.default()
fused = engine.fuse_sources(
    source_results={
        "embedding": sorted_by_embedding,
        "keyword": sorted_by_keyword,
    },
    source_weights={
        "embedding": 0.7,
        "keyword": 0.3,
    },
    key_fn=lambda p: p.handler,
)
```

**Weight rationale:** Embedding gets 0.7 because it handles paraphrasing and
novel phrasing. Keywords get 0.3 because they provide fast, precise matching
for common verbs ("refactor", "search", "scan"). These weights are adjustable
via `VOLITION_EMBEDDING_WEIGHT` and `VOLITION_KEYWORD_WEIGHT` environment
variables.

#### Feedback Adjustment

After fusion, scores are adjusted by historical feedback from
`shared.feedback.FeedbackCollector`:

```python
source_scores = FeedbackCollector.get_instance().get_source_scores()
for candidate in fused_results:
    handler_score = source_scores.get(candidate.handler, 0.5)
    # Mild adjustment: +-10% based on historical effectiveness
    candidate.fused_score *= 0.9 + (handler_score * 0.2)
```

This creates a learning loop: handlers that historically produce "useful"
outcomes for similar queries get a slight boost.

### 3.4 Pre-flight Validation Engine

Mirrors the 4-pass structure of Rhetoric's validation engine
(`rhetoric/scripts/toulmin/validate.py`), adapted for action plans instead
of argument graphs. Pure functions, no I/O, deterministic.

| Pass | Name | Detects |
|------|------|---------|
| 1 | Capability check | Handler unavailable, missing API keys, disabled feature flags |
| 2 | Input validation | Malformed step inputs, missing required fields, invalid step references |
| 3 | Risk assessment | Unconfirmed security actions, high-risk steps without safeguards, rate limit proximity |
| 4 | Dependency check | Circular dependencies, unreachable steps, broken output references |

#### Pass 1: Capability Check

For each step in the plan, verify the handler is available:

```python
def _pass_capability(plan: ActionPlan) -> list[PreflightFlag]:
    flags = []
    capabilities = cmd_capabilities()  # existing function
    for step in plan.steps:
        cap = capabilities.get(HANDLER_TO_CAPABILITY[step.handler])
        if cap is None or cap["status"] == "unavailable":
            flags.append(PreflightFlag(
                type="handler_unavailable",
                severity="CRITICAL",
                step_id=step.step_id,
                description=f"Handler '{step.handler}' is not available",
                remediation=_suggest_fallback(step),
            ))
    return flags
```

#### Pass 2: Input Validation

Verify step inputs are well-formed and references are resolvable:

- Step references (`"step-1.output"`) point to existing prior steps
- Required fields for each handler type are present
- String inputs are non-empty and within length limits

#### Pass 3: Risk Assessment

Check safety constraints:

- Security-category steps require `--confirm` (R.22.1)
- Shodan rate limit proximity check (warn at 80%, block at 100%)
- Steps with `risk_level >= HIGH` and `RHETORIC_PREFLIGHT` enabled trigger
  a Rhetoric deliberation request

#### Pass 4: Dependency Check

Validate the DAG structure:

- No circular dependencies (topological sort must succeed)
- All `depends_on` references point to existing steps
- All `inputs` step references resolve to steps that appear earlier in
  topological order

### 3.5 Execution Engine

The executor processes validated plans step-by-step in topological order.

```python
async def execute_plan(plan: ActionPlan) -> list[ActionOutcome]:
    outcomes: list[ActionOutcome] = []
    step_outputs: dict[str, Any] = {}

    for step in topological_sort(plan.steps):
        # Resolve input references
        resolved_inputs = resolve_inputs(step.inputs, step_outputs)

        # Attempt primary handler
        result = await dispatch(step.handler, step.action, resolved_inputs)

        # Fallback chain on failure
        if result.is_err() and step.fallback_chain:
            for fallback in step.fallback_chain:
                result = await dispatch(fallback, step.action, resolved_inputs)
                if result.is_ok():
                    break

        # Record outcome
        outcome = ActionOutcome(
            plan_id=plan.plan_id,
            step_id=step.step_id,
            handler=step.handler,
            status="success" if result.is_ok() else "error",
            output_summary=summarize(result),
        )
        outcomes.append(outcome)
        step_outputs[step.step_id] = result

        # Abort on failure (no partial execution)
        if result.is_err():
            break

    return outcomes
```

**Design decision: Abort on failure.** Partial plan execution is worse than
no execution. If step-1 (LLM review) fails, step-2 (apply fixes) has no
input to work with. The executor aborts and reports which step failed and why.

### 3.6 Outcome Learning

After execution, outcomes flow into two systems:

1. **Audit log** (`~/.volition/audit.log`): Append-only JSONL with full plan,
   classification scores, execution results, and timing. This is the ground
   truth for debugging.

2. **Feedback loop** (`shared.feedback.FeedbackCollector`): When the user
   provides explicit feedback (via the feedback event system) or when Volition
   observes implicit signals (handler succeeded, handler failed, handler
   fell back), it records a `FeedbackSignal`. Over time, this shifts handler
   weights in the classifier:

```
                      +-----------+
                      |  Classify  |
                      +-----+-----+
                            |
                   uses feedback weights
                            |
                      +-----+-----+
                      |  Execute   |
                      +-----+-----+
                            |
                   records outcome
                            |
                      +-----+------+
                      |  Feedback   |
                      |  Collector  |
                      +------------+
```

The feedback adjustment is deliberately conservative (+-10% per Section 3.3).
A single bad outcome does not tank a handler's score. The
`FeedbackCollector` uses a sliding window of the last 100 signals per source
to prevent ancient data from dominating.

### 3.7 Information Flow with Other Skills

Volition integrates with the Cognitive Construct through three channels:

**Event bus** (`shared.events`): Volition emits `action.started`,
`action.completed`, `action.failed`, `action.confirmed`, `action.rejected`.
These are defined in `shared/events.py:52-56` and absorbed by Inland Empire
for memory storage.

**Membrane** (`shared.membrane`): Volition's membrane
(`shared/membrane.py:342-365`) absorbs `rhetoric.decision.made` (when
Rhetoric determines action is required) and `encyclopedia.cache.miss` (when
Encyclopedia needs a fallback search). It emits the five action events above.

**Synergy bus** (`shared.synergies`): Volition logs completed actions to
Inland Empire via `volition_log_action()` (`shared/synergies.py:128-169`).
This is fire-and-forget; if Inland Empire is unavailable, Volition
continues without error.

```
  Rhetoric ──decision.made──> [Volition Membrane] ──> Classify → Plan → Execute
                                     │
                                     ├──action.completed──> Inland Empire
                                     ├──action.completed──> Feedback Collector
                                     └──action.failed────> Audit Log
```

---

## 4. Constitution

These rules govern the executive skill and cannot be overridden.

### Rule 1: NEVER execute an action the classifier is not confident about.

If the top candidate's fused score is below the confidence threshold,
Volition MUST ask for clarification. No default fallback to "just call
the LLM." Uncertainty is not a bug -- it is information that the user's
intent is ambiguous.

### Rule 2: NEVER execute a security action without explicit confirmation.

Shodan queries, vulnerability scans, and any action classified as
`security` category require the `--confirm` flag. This is not overridable
by confidence score, feedback adjustment, or plan construction. A security
action with 0.99 confidence still requires confirmation.

### Rule 3: NEVER execute a plan that fails pre-flight validation.

If any pre-flight pass produces a CRITICAL flag, the plan does not execute.
The user sees the flags and decides whether to modify the request. Volition
does not auto-remediate CRITICAL flags.

### Rule 4: NEVER hide classification uncertainty from the user.

When `--verbose` is active, the full `ClassificationResult` with all
candidate scores is shown. When `--verbose` is not active, the selected
handler and confidence score are still included in the JSON output.
The user can always ask "why did you choose that handler?"

### Rule 5: NEVER let feedback adjustment override safety constraints.

Feedback weights can boost or reduce handler scores, but they cannot:
- Push a below-threshold score above threshold (the raw fused score
  must independently exceed threshold)
- Bypass confirmation requirements for security actions
- Skip pre-flight validation passes

### Rule 6: ALWAYS abort on step failure.

If any step in an action plan fails (after exhausting its fallback chain),
the entire plan aborts. No partial execution. The user gets a clear report
of which step failed, why, and what fallbacks were attempted.

### Rule 7: ALWAYS log before acting.

The audit log entry for a plan is written BEFORE execution begins, with
status `"started"`. This ensures that even if Volition crashes mid-execution,
the audit trail shows what was attempted. A second entry with the final
status is written after completion or failure.

---

## 5. Tasks

### Phase 1: Intent Classification Engine

**Goal:** Replace keyword counting with embedding + keyword fusion.

| Task | Description | Files |
|------|-------------|-------|
| 1.1 | Define `IntentPrototype` dataclass and prototype registry | `volition/scripts/classify.py` (new) |
| 1.2 | Write prototype texts for all 4 handlers + 1 "ambiguous" prototype | `volition/scripts/classify.py` |
| 1.3 | Implement `classify_intent()` using `shared.embeddings.rank_by_relevance()` | `volition/scripts/classify.py` |
| 1.4 | Add keyword scoring as secondary signal | `volition/scripts/classify.py` |
| 1.5 | Fuse signals via `shared.fusion.RRFEngine` | `volition/scripts/classify.py` |
| 1.6 | Add feedback adjustment from `shared.feedback.FeedbackCollector` | `volition/scripts/classify.py` |
| 1.7 | Implement confidence gating with clarification request | `volition/scripts/classify.py` |
| 1.8 | Wire `classify_intent()` into `cmd_act()` replacing `_classify_intent()` | `volition/scripts/volition.py` |
| 1.9 | Tests: classification accuracy on 20+ test cases, threshold behavior | `volition/tests/test_classify.py` (new) |

**Depends on:** `shared.embeddings` (exists), `shared.fusion` (exists)

### Phase 2: Action Plan Construction

**Goal:** Build multi-step DAGs for complex intents.

| Task | Description | Files |
|------|-------------|-------|
| 2.1 | Define `ActionPlan`, `PlanStep`, `ActionOutcome` dataclasses | `volition/scripts/planner.py` (new) |
| 2.2 | Implement single-step plan builder (wraps current behavior) | `volition/scripts/planner.py` |
| 2.3 | Implement multi-step plan builder with dependency extraction | `volition/scripts/planner.py` |
| 2.4 | Implement topological sort and input reference resolution | `volition/scripts/planner.py` |
| 2.5 | Tests: single-step, multi-step, circular dependency rejection | `volition/tests/test_planner.py` (new) |

**Depends on:** Phase 1 (classification result feeds plan builder)

### Phase 3: Pre-flight Validation

**Goal:** 4-pass validation mirroring Rhetoric's engine.

| Task | Description | Files |
|------|-------------|-------|
| 3.1 | Define `PreflightFlag`, `PreflightResult` dataclasses | `volition/scripts/preflight.py` (new) |
| 3.2 | Implement Pass 1: Capability check | `volition/scripts/preflight.py` |
| 3.3 | Implement Pass 2: Input validation | `volition/scripts/preflight.py` |
| 3.4 | Implement Pass 3: Risk assessment (including R.22 constraints) | `volition/scripts/preflight.py` |
| 3.5 | Implement Pass 4: Dependency check | `volition/scripts/preflight.py` |
| 3.6 | Wire preflight into plan execution pipeline | `volition/scripts/volition.py` |
| 3.7 | Tests: each pass individually + full pipeline | `volition/tests/test_preflight.py` (new) |

**Depends on:** Phase 2 (validates ActionPlan structures)

### Phase 4: Execution & Fallback Chains

**Goal:** Step-by-step plan execution with fallback recovery.

| Task | Description | Files |
|------|-------------|-------|
| 4.1 | Define fallback chain configuration per handler | `volition/scripts/handlers.py` (new) |
| 4.2 | Implement `execute_plan()` with topological step execution | `volition/scripts/executor.py` (new) |
| 4.3 | Implement fallback dispatch on handler failure | `volition/scripts/executor.py` |
| 4.4 | Implement abort-on-failure semantics | `volition/scripts/executor.py` |
| 4.5 | Wire executor into main pipeline (classify -> plan -> validate -> execute) | `volition/scripts/volition.py` |
| 4.6 | Tests: success path, fallback path, abort path | `volition/tests/test_executor.py` (new) |

**Depends on:** Phase 3 (only validated plans reach executor)

### Phase 5: Outcome Learning

**Goal:** Close the feedback loop for adaptive routing.

| Task | Description | Files |
|------|-------------|-------|
| 5.1 | Emit `ActionOutcome` events via `shared.events.create_event()` | `volition/scripts/executor.py` |
| 5.2 | Record implicit feedback signals (success/failure/fallback) | `volition/scripts/executor.py` |
| 5.3 | Read feedback scores in classifier for weight adjustment | `volition/scripts/classify.py` |
| 5.4 | Add `VOLITION_LEARNING_RATE` env var for adjustment sensitivity | `volition/scripts/classify.py` |
| 5.5 | Tests: feedback adjustment shifts scores, safety constraints are not bypassed | `volition/tests/test_learning.py` (new) |

**Depends on:** Phase 4 (outcomes come from execution), `shared.feedback` (exists)

### Phase 6: Integration & Polish

**Goal:** Wire everything together, update SKILL.md, add verbose mode.

| Task | Description | Files |
|------|-------------|-------|
| 6.1 | Update SKILL.md with new architecture and commands | `volition/SKILL.md` |
| 6.2 | Add `--verbose` flag showing full classification breakdown | `volition/scripts/volition.py` |
| 6.3 | Add `--dry-run` flag showing plan without executing | `volition/scripts/volition.py` |
| 6.4 | Update membrane definition in `shared/membrane.py` if new event types needed | `shared/membrane.py` |
| 6.5 | Write constitution.md | `volition/constitution.md` (new) |
| 6.6 | End-to-end integration tests | `volition/tests/test_integration.py` (new) |

**Depends on:** All prior phases

---

## 6. Open Questions

### Q1: The Bootstrap Problem

The embedding model needs prototype texts to classify against. But prototype
texts are written by humans based on intuition about what each handler should
handle. How do we validate that the prototype texts actually capture the right
semantic space?

**Proposed answer:** Ship Phase 1 with hand-written prototypes. After 100+
real uses, analyze the audit log to find misclassifications. Refine prototype
texts based on actual query distribution. The feedback loop (Phase 5) provides
the automated signal; prototype refinement remains manual.

### Q2: Confidence Threshold Calibration

The default threshold of 0.65 is arbitrary. Too low: Volition guesses wrong.
Too high: Volition asks for clarification on obvious requests.

**Proposed answer:** Start at 0.65. Log all classifications with scores. After
initial deployment, compute the empirical distribution of correct-classification
scores vs. misclassification scores. Set threshold at the intersection point.
Expose as `VOLITION_CONFIDENCE_THRESHOLD` env var for manual override.

### Q3: Multi-Step Plan Construction

How does Volition know that "review and fix" is two steps, not one? The
current LLM-based decomposition approach (sending the action to an LLM to
extract sub-intents) introduces non-determinism and latency.

**Proposed answer:** Use simple conjunction detection ("and", "then", "after
that", "followed by") to split compound actions. Classify each sub-action
independently. This is imperfect but deterministic. Phase 2 of plan
construction can add LLM-assisted decomposition behind a feature flag.

### Q4: Feedback Weighting Stability

If a backend is temporarily degraded (e.g., OpenAI has an outage), the
feedback collector will record many "not_useful" signals, tanking the
handler's score. When the backend recovers, it takes 100 subsequent
positive signals to restore the score.

**Proposed answer:** Use exponential decay weighting so recent signals
matter more than old ones. Add a `VOLITION_FEEDBACK_DECAY` parameter
(default 0.95) that multiplies each signal's weight by its age in decay
periods. Alternatively, detect backend recovery via the membrane's
`source.restored` event and reset the handler's feedback window.

### Q5: Rhetoric Integration Overhead

If `RHETORIC_PREFLIGHT` is enabled for high-risk actions, and Rhetoric
requires an LLM call (via Mercury), this adds seconds of latency to
every consequential action.

**Proposed answer:** Rhetoric's `validate` command is pure symbolic (no
LLM call). Only `plan` requires Mercury. For pre-flight, we use `validate`
on a synthetic argument graph: "This action should be taken because [plan
rationale]." This keeps pre-flight fast and deterministic.

---

## 7. Success Criteria

### 7.1 Classification Accuracy

Given a test set of 50 natural-language action descriptions with human-labeled
correct handlers:

- **Precision >= 0.90** for each handler category
- **Recall >= 0.85** for each handler category
- **Ambiguous detection rate >= 0.80**: actions intentionally designed to be
  ambiguous should trigger clarification requests, not wrong classifications

### 7.2 Confidence Gating Effectiveness

- **False confidence rate < 5%**: actions classified above threshold that
  were routed to the wrong handler
- **Over-caution rate < 15%**: actions that triggered clarification but had
  an obvious correct handler

### 7.3 Latency Targets

| Component | Target | Measurement |
|-----------|--------|-------------|
| Intent classification | < 50ms | `time.perf_counter()` around `classify_intent()` |
| Pre-flight validation | < 20ms | `time.perf_counter()` around `preflight_validate()` |
| Plan construction | < 10ms | `time.perf_counter()` around `build_plan()` |
| Full pipeline (classify + plan + validate) | < 100ms | End-to-end before handler dispatch |

### 7.4 Safety Invariants

- **Zero unconfirmed security actions**: no security-category action executes
  without `--confirm` flag, regardless of confidence score
- **Zero executions of invalid plans**: no plan with CRITICAL pre-flight flags
  reaches the executor
- **Zero silent failures**: every error produces a structured JSON response
  with error code and human-readable message

### 7.5 Learning Loop

After 200 actions with feedback:
- Classification accuracy improves by >= 3 percentage points over the
  no-feedback baseline
- Handler selection for repeated query patterns converges within 10 examples

---

## 8. Implementation Notes

### 8.1 Language and Dependencies

- **Python >= 3.12** (consistent with Rhetoric skill)
- **model2vec**: Static embeddings for intent classification (already in
  `shared.embeddings`, no additional dependency)
- **pydantic**: Data validation for plan structures
- **No new external dependencies**: all infrastructure exists in `shared/`

### 8.2 Testing Strategy

**Unit tests** (per-module):
- `test_classify.py`: 20+ cases covering exact keywords, paraphrases,
  ambiguous inputs, embedding-unavailable fallback
- `test_planner.py`: single-step, multi-step, dependency ordering, invalid DAGs
- `test_preflight.py`: each pass independently, combined pipeline,
  CRITICAL flag abort
- `test_executor.py`: success path, fallback chain, abort-on-failure

**Integration tests** (`test_integration.py`):
- Full pipeline: user action -> classification -> plan -> validation -> execution
- Synergy integration: action completion -> Inland Empire memory logging
- Feedback loop: record outcome -> subsequent classification weight shift

**Property tests** (if time permits):
- For any valid ActionPlan, topological sort produces a valid execution order
- Pre-flight validation is deterministic: same plan -> same flags

### 8.3 File Organization

```
volition/
  SPEC.md              <- this document
  SKILL.md             <- user-facing skill specification (updated Phase 6)
  constitution.md      <- inviolable rules (created Phase 6)
  README.md            <- usage and installation
  scripts/
    volition.py        <- CLI entrypoint (modified)
    classify.py        <- intent classification engine (new, Phase 1)
    planner.py         <- action plan construction (new, Phase 2)
    preflight.py       <- pre-flight validation engine (new, Phase 3)
    executor.py        <- step-by-step plan executor (new, Phase 4)
    handlers.py        <- handler registry and fallback config (new, Phase 4)
    cross-llm-mcp/     <- existing backend
    mcp-shodan/        <- existing backend
    mcp-server-openai/ <- existing backend
    openai-websearch-mcp/ <- existing backend
    serena/            <- existing backend
  tests/
    test_classify.py   <- Phase 1 tests
    test_planner.py    <- Phase 2 tests
    test_preflight.py  <- Phase 3 tests
    test_executor.py   <- Phase 4 tests
    test_learning.py   <- Phase 5 tests
    test_integration.py <- Phase 6 tests
```

### 8.4 Migration Path

The new classification engine is a drop-in replacement for `_classify_intent()`.
Phase 1 can be deployed by changing a single call site in `cmd_act()`. All
existing CLI commands (`act`, `edit`, `query`, `capabilities`) continue to work
unchanged. The plan construction, pre-flight, and execution machinery wrap
around the existing handler functions without modifying them.

This means each phase can be shipped, tested, and observed independently.
There is no big-bang cutover.
