## Results Evaluation Plan

### Aim

Evaluate whether the scheduling system:

1. books only feasible appointments;
2. produces better allocations than simple baselines;
3. remains effective under multi-patient contention;
4. treats stricter and more flexible patient profiles fairly;
5. scales acceptably as problem size grows.

### Main Results Section

#### 1. Constraint Satisfaction and Safety

Purpose:
Show that the scheduler respects hard scheduling constraints before discussing quality.

Metrics:
- hard-constraint violations
- bookings outside horizon
- bookings on excluded days or periods
- excluded modality violations
- disallowed routing modality violations

Expected outcome:
Zero violations. Any non-zero result is a defect, not a tradeoff.

Implementation status:
- keep, but treat as a correctness section rather than a headline experiment

#### 2. Single-Patient Allocation Quality vs Baselines

Purpose:
Show that the scoring and tie-breaking logic improves allocation quality beyond trivial booking policies.

Policies to compare:
- current scheduler
- earliest feasible slot
- random feasible slot
- earliest feasible preferred-modality slot

Metrics:
- booking success rate
- mean utility
- median utility
- p95 utility
- preferred day-period hit rate
- preferred modality hit rate
- 95% bootstrap confidence intervals

Interpretation:
If the current scheduler only matches a naive baseline, the utility model is not doing enough useful work.

Implementation status:
- keep the existing allocation-quality benchmark only if it is rewritten to compare against baselines

#### 3. Multi-Patient Contention and Fairness

Purpose:
Evaluate the scheduler in the more realistic setting where multiple patients compete for the same shared slot pool.

Policies to compare:
- FCFS/input order
- random patient order
- scarcity-first order

Metrics:
- total patients booked
- booking rate
- mean utility per patient
- strict-group booking rate
- flexible-group booking rate
- utility gap between strict and flexible groups
- per-request-type success rate
- unscheduled patient count

Interpretation:
This should become the main evaluation because it tests whether the system still works when one patient's booking changes another patient's options.

Implementation status:
- new experiment required

#### 4. Runtime and Scaling

Purpose:
Quantify computational cost and relate it to the number of patients and candidate slots.

Experiments:
- fixed patient count, increasing slots
- fixed slot count, increasing patients
- increasing both together

Metrics:
- mean runtime
- median runtime
- p95 runtime
- maximum runtime
- empirical growth trend

Interpretation:
Runtime results should be discussed in the context of whether the method is suitable for interactive appointment support.

Implementation status:
- keep runtime scaling, but extend it to multi-patient runs

#### 5. Ablation Study

Purpose:
Show which design choices actually matter.

Ablations:
- no scarcity ordering
- no preference weighting
- no day-period synergy
- no relaxation
- no soonest component

Metrics:
- booking rate
- mean utility
- fairness gap

Interpretation:
This demonstrates understanding of the model rather than treating it as a black box.

Implementation status:
- new experiment recommended

#### 6. Optional User Evaluation

Purpose:
Only include this if the report claims that the system is usable, understandable, or trustworthy for users.

Suggested study:
- 8 to 15 participants
- short task-based evaluation
- compare plain result output vs explanation plus relaxation prompts

Measures:
- task completion
- time on task
- confidence/trust Likert items
- ease-of-use questionnaire such as SUS

Interpretation:
Keep this brief unless usability is a central project claim.

### Existing Experiments: Keep, Rewrite, or Demote

Keep:
- runtime scaling
- negotiation evaluation if negotiation is a claimed contribution
- LLM vs form only if intake/extraction is a central project aim

Rewrite:
- allocation quality and fairness
  - add baselines
  - extend to shared-slot multi-patient contention

Demote to appendix or sanity checks:
- determinism
- tie-breaking

### Quantified Outcomes to Report

The final report should include:
- exact booking success rates
- exact utility summaries
- fairness gaps between patient groups
- confidence intervals where appropriate
- runtime summaries and scaling interpretation

### Future Work

Strong future-work ideas:
- compare the heuristic against exact optimal assignment on small instances
- add clinician-side balancing and service capacity constraints
- replace synthetic slot pools with de-identified real booking patterns
- learn preference weights from observed user choices
- evaluate on real cohorts and demographic fairness criteria
- add prospective user testing in a realistic booking workflow
