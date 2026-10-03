# Paper 2 – Controllable Consequence-Aware Underwater Image Enhancement

## Research Direction

Paper 2 investigates whether underwater image enhancement can be made
controllable and self-correcting by learning from the consequences
produced by its own enhancement process.

## Research Question

Can underwater image enhancement be made controllable and self-correcting
by learning from the consequences produced by its own enhancement process?

## Core Concept

The proposed framework follows a closed-loop enhancement process:

X → E(X, α) → Y → R(Y) → X̂
→ Consequence Analysis → Feedback → α′
→ E(X, α′)

where:

- X = input underwater image
- E = enhancement model
- α = enhancement control parameter
- Y = enhanced image
- R = re-degradation / reconstruction operator
- X̂ = reconstructed/degraded estimate
- Consequence Analysis = analysis of the effects produced by enhancement
- Feedback = learned correction signal
- α′ = updated enhancement control

## Main Research Idea

The central idea is not simply to perform enhancement followed by
re-degradation.

The research focus is on learning from the consequences of enhancement
and using those consequences to control a subsequent enhancement step.

## Controllability

The enhancement process will include a continuous control variable:

α ∈ [0, 1]

Initial interpretation:

- α = 0 → conservative enhancement
- α = 0.5 → balanced enhancement
- α = 1 → aggressive enhancement

These interpretations are hypotheses and must be validated experimentally.

## Closed-Loop Principle

The system should be able to:

1. Enhance an underwater image using a selected control level.
2. Analyze the consequences of that enhancement.
3. Estimate whether the enhancement should be modified.
4. Produce a feedback signal.
5. Adjust the enhancement control.
6. Perform a subsequent enhancement step.

## Distinction from Paper 1

Paper 2 must not claim re-degradation/reconstruction itself as the
primary novelty.

Re-degradation is used as part of the consequence-analysis mechanism.

The intended novelty is the learned consequence-aware feedback/control
mechanism.

## Development Milestones

M0 – Research specification
M1 – Dataset pipeline
M2 – Controllable enhancement
M3 – Re-degradation
M4 – Consequence representation
M5 – Feedback controller
M6 – One-step closed loop
M7 – Loss design
M8 – Full closed loop
M9 – Evaluation
M10 – Ablation studies
M11 – Robustness and generalization
M12 – Final analysis

## Experimental Principle

The architecture and losses should be developed incrementally.

Each major component must be experimentally validated before the next
component is added.

Paper 1 code, architecture, checkpoints, results, and research files
must remain untouched.