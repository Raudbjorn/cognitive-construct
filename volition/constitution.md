# Volition Constitution

These rules govern the executive skill and cannot be overridden.

## Rule 1: NEVER execute an action the classifier is not confident about.

If the top candidate's fused score is below the confidence threshold,
Volition MUST ask for clarification. No default fallback to "just call
the LLM." Uncertainty is not a bug -- it is information that the user's
intent is ambiguous.

## Rule 2: NEVER execute a security action without explicit confirmation.

Shodan queries, vulnerability scans, and any action classified in the
`security` category require the `--confirm` flag. This is not overridable
by confidence score, feedback adjustment, or plan construction. A security
action with 0.99 confidence still requires confirmation.

## Rule 3: NEVER execute a plan that fails pre-flight validation.

If any pre-flight pass produces a CRITICAL flag, the plan does not execute.
The user sees the flags and decides whether to modify the request. Volition
does not auto-remediate CRITICAL flags.

## Rule 4: NEVER hide classification uncertainty from the user.

When `--verbose` is active, the full ClassificationResult with all
candidate scores is shown. When `--verbose` is not active, the selected
handler and confidence score are still included in the JSON output.
The user can always ask "why did you choose that handler?"

## Rule 5: NEVER let feedback adjustment override safety constraints.

Feedback weights can boost or reduce handler scores, but they cannot:
- Push a below-threshold score above threshold (the `raw_fused_score`
  is preserved before feedback adjustment and must independently exceed
  threshold)
- Bypass confirmation requirements for security actions
- Skip pre-flight validation passes

## Rule 6: ALWAYS abort on step failure.

If any step in an action plan fails (after exhausting its fallback chain),
the entire plan aborts. No partial execution. The user gets a clear report
of which step failed, why, and what fallbacks were attempted.

## Rule 7: ALWAYS log before acting.

The audit log entry for a plan is written BEFORE execution begins, with
status `"started"`. This ensures that even if Volition crashes mid-execution,
the audit trail shows what was attempted. A second entry with the final
status is written after completion or failure.
