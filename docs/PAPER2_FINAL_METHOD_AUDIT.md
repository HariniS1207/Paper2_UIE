# Paper 2 final method and reproducibility audit

Audit date: 2026-10-06. Scope: documentation and evidence audit of the current repository and saved artifacts. No experiment was rerun for this audit. The selected research decision is **B — ONE-STEP ONLY**.

## A. Research question

Can a frozen, continuously controllable underwater image enhancer be adjusted once using only the input image, its current enhanced output, and the current control value, to improve paired-target L1 quality on held-out EUVP states?

The evidence here is limited to one EUVP test partition and mean absolute pixel error (L1). It does not establish perceptual quality or generalization beyond this split.

## B. Final proposed method

The final method is the frozen endpoint enhancer with the verified **Baseline B** state-conditioned directional controller, applied for one correction. At inference, the controller consumes image-statistic descriptors and does not consume the target image or oracle values. The endpoint enhancer is held fixed during controller inference.

The originally trained Baseline B weights were not saved by the Local Probe Controller run. `results/reconstructed_baseline_b_seed42_20261006/controller.pth` is an explicitly identified reconstruction from the original cached training features and recipe, not recovery of the original weights. Its reported predictions reproduce the persisted original validation and test predictions exactly at class/action level, with maximum probability difference `1.1921e-7`. The two-step stability study used this reconstructed artifact. The one-step action-effectiveness study replayed persisted original predictions and thresholds.

## C. Mathematical formulation

For input image `X` and control `alpha` in `[0,1]`:

```text
Y_alpha = E(X, alpha)

z_alpha = [alpha, phi(X), phi(Y_alpha), phi(Y_alpha) - phi(X)]

a = f_theta(z_alpha)

alpha' = clip(alpha + 0.1*a, 0, 1)
```

Here `phi` is the 14-statistic RGB/luminance/gradient descriptor implemented by `experiments/diagnose_feedback_failure.py`; the concatenated Baseline B vector has 43 entries (one alpha plus three 14-value descriptors). `a` is a directional class decision mapped to an adjustment of `-0.1` or `+0.1`; the saved policy uses validation-selected thresholding of the conditional increase probability. At the boundary, clipping/feasibility can reduce the applied change. The one-step endpoint output is `E(X, alpha')`.

## D. Architecture components and implementation classification

The classifications below describe the role supported by current code and artifacts, not whether a component is novel.

| Component / files | Classification | Audit finding |
|---|---|---|
| EUVP pair loading: `src/data/euvp_dataset.py`, `src/data/feedback_state_dataset.py`, `configs/dataset.yaml` | DEVELOPMENT/UTILITY | Pair loading and manifest-backed RGB resize/tensor conversion; not a learned method component. |
| Fixed pair split: `results/metrics/feedback_target_split.csv` | CORE FINAL METHOD | Fixed train/validation/held-out test manifest used to sample controller states and report final evaluations. Counts: 8004 / 1715 / 1716 pairs. |
| Controllable enhancer: `src/models/controllable_enhancer.py` | CORE FINAL METHOD | Three 3×3 convolution layers (3→32→32→3 with ReLU between), residual `X + alpha*residual`, clamp to `[0,1]`. |
| Endpoint training: `experiments/run_feedback_phase.py`, `src/training/train_step.py`, `src/losses/alpha.py`; checkpoint `checkpoints/feedback_dev_seed42_20261004_retry1/endpoint_enhancer.pth` | CORE FINAL METHOD | Development-run frozen endpoint model; 3 epochs, seed 42, random initialization; identity L1 at alpha 0 and endpoint L1 at alpha 1. It did not load the Paper 1 checkpoint. This is the endpoint used by the local-probe work. |
| Image-statistics representation: `experiments/diagnose_feedback_failure.py` (`image_stats`), `experiments/run_local_probe_controller.py` | CORE FINAL METHOD | 14 statistics per RGB image; Baseline B uses alpha, input descriptor, current-output descriptor and their descriptor difference (43D). |
| Oracle: `src/training/feedback_oracle.py`; materialized labels in `results/metrics/feedback_dev_seed42_20261004_margin_corrected/local_action_oracles.csv` | EVALUATION-ONLY | Uses target L1 at current and feasible `alpha±0.1` candidate settings. Not available to the controller at inference. |
| Baseline B controller: `experiments/run_local_probe_controller.py`, reconstruction recipe `experiments/reconstruct_baseline_b.py`, artifact `results/reconstructed_baseline_b_seed42_20261006/controller.pth` | CORE FINAL METHOD | `Linear(43,3)`; decrease/hold/increase. The core one-step policy is Baseline B. Artifact is reconstructed, not original-run weights. |
| Alpha-only reference: `experiments/run_local_probe_controller.py`; saved results under `results/local_probe_controller_seed42_20261004/` and `results/action_effectiveness_seed42_20261005/` | ABLATION | Control-value-only comparator. |
| Candidate C full local response and C1/C2/C3 feature variants: `experiments/run_local_probe_controller.py`; `results/local_probe_controller_seed42_20261004/report.md` | REJECTED APPROACH / ABLATION | Candidate C failed its predefined requirement to improve both decrease AUROC and decrease recall over B. C1–C3 are representation ablations. Kept as research history. |
| Earlier `AlphaOnlyController`, `ImageStateController`, `FeedbackController`, `ConsequenceEncoder`: `src/models/feedback_controller.py`, `src/models/consequence_encoder.py` and staged code in `experiments/run_feedback_phase.py` | EXPERIMENTAL BASELINE | Earlier regression/convolutional controller candidates; not the final Baseline B classifier. |
| Consequence/re-degradation components: `src/models/redegradation.py`, `src/models/consequence_encoder.py`, `src/losses/consequence.py`, `src/losses/control.py`, `src/losses/reconstruction.py`, `src/losses/total.py`; staged `experiments/run_feedback_phase.py` | REJECTED APPROACH / DEVELOPMENT/UTILITY | Present in the earlier broad research framework. They are not used in the final one-step Baseline B method or supported as final method components by the chosen results. No deletion is implied. |
| Action-effectiveness evaluation: `experiments/evaluate_action_effectiveness.py`, `results/action_effectiveness_seed42_20261005/` | EVALUATION-ONLY | Replays saved held-out predictions and thresholds; computes realized target-L1 change. |
| Closed-loop stability evaluation: `src/models/closed_loop.py`, `experiments/evaluate_closed_loop_stability.py`, `results/closed_loop_stability_seed42_20261006/` | EVALUATION-ONLY | Two-step diagnostic only; recomputes enhancer output and descriptors before step 2. It is not adopted as the final method. |
| Diagnostics and summaries: `experiments/analyze_feedback_targets.py`, `experiments/analyze_consequence_features.py`, `experiments/diagnose_feedback_failure.py`, `experiments/diagnose_consequence_representation.py`, `experiments/validate_alpha_behavior.py`, `src/evaluation/test_controllability.py` | DEVELOPMENT/UTILITY | Analysis, representation/alpha behavior checks, and evaluation helpers. Their outputs are historical diagnostic evidence, not additional final method modules. |
| Generic training and config: `src/training/config.py`, `src/training/loss_utils.py`, `src/training/smoke_test.py`, `src/training/train.py`, `configs/training.yaml`, `configs/environment.yaml` | DEVELOPMENT/UTILITY | Generic prototype/training infrastructure; not a claim that the full configured 20-epoch run produced the audited frozen endpoint. |

## E. Training procedure

The audited endpoint checkpoint metadata and `results/metrics/feedback_dev_seed42_20261004_retry1/run_config.json` identify the actual enhancer as randomly initialized (no `paper2_epoch_001.pth`/Paper 1 weights loaded), seed 42, 3 epochs, batch size 8, Adam learning rate `1e-4`, weight decay `1e-5`. The development run uses 1,024 selected training pairs for endpoint development. Images are RGB, bilinearly resized to 256×256, float32 `[0,1]`. Training loss is endpoint L1 plus identity L1; for this residual architecture the zero-control identity is structurally produced. The final experiment freezes this endpoint checkpoint.

`configs/training.yaml` is generic project configuration (20 epochs, batch size 4, etc.) and must not be cited as the actual recipe for this 3-epoch frozen checkpoint. The two-step evaluation performs no training.

## F. Controller training procedure

The saved reconstruction metadata (`results/reconstructed_baseline_b_seed42_20261006/config.json`) records the original helper recipe: seed 42 immediately before `nn.Linear` construction; full-batch weighted cross-entropy; inverse-frequency class weights `[2.4444444, 5.1764708, 0.4170616]`; Adam at `0.03`, weight decay `0.001`, 300 epochs; training state counts decrease/hold/increase = 144/68/844. Features are standardized with training-feature means and population standard deviations (`unbiased=False`, clamped to at least `1e-6`). Class order is decrease, hold, increase. A threshold of `0.46343794465065` on conditional increase probability was selected by maximum validation balanced accuracy for the binary directional interface.

The reconstruction artifact embeds model weights and required preprocessing/schema details. Replayed predictions match original persisted predictions on all 1,747 validation and 1,748 test states; the artifact’s report identifies its source hashes. This agreement supports recipe/prediction reproducibility, but does not restore the lost original state dict.

## G. Oracle definition

For current quality `q_current` and feasible candidate qualities at clipped `alpha-0.1` and `alpha+0.1`, quality is target-image mean L1. The oracle selects the feasible direction with the greatest positive gain `q_current - q_candidate`, only when that gain exceeds `0.0001` and, when both directions are feasible, beats the alternative by more than `0.0001`; otherwise it holds. The implementation is `src/training/feedback_oracle.py`. Oracle decisions use the paired target and are strictly training/evaluation labels or upper-reference measurements, never controller inputs at inference.

## H. Evaluation protocol

The split manifest is `results/metrics/feedback_target_split.csv` (8,004 controller-train, 1,715 validation, 1,716 held-out test pairs). The fixed local-probe state artifacts contain 1,056 training states, 1,747 validation states and 1,748 held-out test states; the extra validation/test states reflect explicit alpha-boundary samples. Preprocessing is RGB bilinear 256×256 float32 `[0,1]`. Quality `Q` is paired-target mean pixelwise L1; improvement is `Q_before - Q_after`, so positive values are better. No perceptual metrics are reported.

The one-step effectiveness evaluation replays persisted predictions and the validation threshold from the original local-probe run, with no test refitting. The two-step stability evaluation uses the reconstructed Baseline B checkpoint, frozen enhancer and the same held-out state list. At step 2 it recomputes `Y_alpha1`, `phi(X)`, `phi(Y_alpha1)` and `f_theta`; no third step was run. Oracle values are reported only as an evaluation reference.

## I. One-step results

Source: `results/action_effectiveness_seed42_20261005/report.md`, `metrics.json`, and `per_state_predictions.csv`; local-probe classifier discrimination results: `results/local_probe_controller_seed42_20261004/report.md` and `model_metrics.csv`.

On 1,748 held-out states (from 1,716 pairs), Baseline B mean L1 improvement was `0.0037084` (median `0.0052407`), with 75.11% improved and 24.89% worsened. Paired 95% bootstrap CI for mean improvement was `[0.0034377, 0.0039680]`. Alpha-only mean improvement was `0.0029457`. Candidate C was effectively tied with B: paired C−B mean `−0.00000556`, 95% CI `[−0.0001493, 0.0001449]`. The local-probe predefined discrimination criterion also failed: C's decrease AUROC was 0.8530 vs B's 0.8522, but decrease recall was unchanged at 0.7220.

## J. Two-step stability results

Source: `results/closed_loop_stability_seed42_20261006/report.md`, `metrics.json`, `per_state_trajectory.csv`, `alpha_bin_metrics.csv`, and `action_transition_metrics.csv`.

With reconstructed Baseline B, step 1 mean gain was `0.0037084` (75.11% improved); step 2 incremental mean gain was `0.0031161` (72.31% improved, 27.69% worsened); total two-step gain was `0.0068245` (67.05% net improved vs initial state). Immediate applied-action reversals occurred in 262/1,748 states (14.99%). For initial alpha `[0.8,1.0]`, mean gains were negative at step 1 (`−0.0004660`) and step 2 (`−0.0011310`); 56.73% of this bin worsened on step 2. Thus the additional step has positive average incremental L1 gain but does not establish stable iterative correction. The documented research decision is **B — ONE-STEP ONLY**.

The step-1 replay check matched saved actions on all 1,748 rows and per-state L1 improvements exactly. The report also notes max absolute discrepancy `0.000407994` between recomputed initial features and cached initial feature rows; it did not change initial actions or step-1 improvements. This discrepancy remains a reproducibility detail to investigate before relying on cached features for a different analysis.

## K. Rejected approaches

- Candidate C's added local-response/probe representation did not meet its stated validation criterion and showed no measurable held-out realized-L1 advantage over Baseline B. Preserve its code and artifacts as research history.
- The earlier consequence-encoder/regression-controller direction in `experiments/run_feedback_phase.py` is not the selected final controller or supported novelty in this audit. Its saved staged metrics remain exploratory.
- Iterative use beyond one correction is not adopted. The two-step experiment is a stability diagnostic whose observed failures motivate the one-step limit, not evidence for an iterative method.
- The broad re-degradation/reconstruction concept in README and associated model/loss modules is not part of the final method evidenced here.

## L. Limitations

- The original Baseline B state dict is missing. The later reconstruction matches persisted predictions to reported numerical tolerance, but is not the original weights.
- The action-effectiveness run depends on saved predictions because the original controller checkpoint did not exist; it is not independently replayable from that original run’s weights.
- Evidence is from one fixed EUVP held-out partition and paired pixel L1 only; no perceptual/human evaluation or external-dataset generalization is established.
- The endpoint enhancer is a small development-scale, randomly initialized model trained for three epochs on the selected development data; results do not establish performance of a pretrained Paper 1 model or a production enhancer.
- The two-step stability results expose alpha-dependent failures and reversals; no claim of stable iterative self-correction is supported.
- Cached versus recomputed initial feature values differ by up to `0.000407994` in one replay audit, despite identical initial actions and gains.

## M. Final scientific claims supported by evidence

1. On this fixed EUVP held-out state set, the Baseline B one-step policy produced a positive mean paired-L1 improvement, with a paired bootstrap interval above zero, and improved 75.11% of states.
2. The saved Baseline B training recipe can be replayed from cached features to produce a reconstruction matching all persisted original validation/test decisions and nearly identical probabilities.
3. Candidate C did not show a measurable realized-L1 improvement over Baseline B and failed its predefined joint AUROC/recall criterion.
4. Applying the same controller a second time produced positive mean incremental L1 gain in aggregate but worsened 27.69% of states, with worse results concentrated at higher initial alpha; these data do not justify an iterative stability claim.

## N. Claims that must NOT be made

- Do not claim stable, convergent, or reliably beneficial multi-step/iterative self-correction.
- Do not claim the original Baseline B weights were recovered; cite the reconstructed checkpoint as a reconstruction, with its provenance.
- Do not claim Candidate C or local probes improve on Baseline B.
- Do not claim perceptual enhancement improvement, human preference, or generalization to other datasets from these L1 results.
- Do not claim use or validation of the Paper 1 pretrained checkpoint; the audited enhancer was randomly initialized.
- Do not present the oracle as an inference-time controller or claim it is available without target images.
- Do not present re-degradation, reconstruction, the consequence encoder, or the two-step diagnostic as components of the final selected method.

## Audit coverage and changes

The source audit covered the repository’s `src/`, `experiments/`, `configs/`, `tests/`, checkpoint and results trees, plus `README.md` and the split manifest. Result directories reviewed include `local_probe_controller_seed42_20261004`, `reconstructed_baseline_b_seed42_20261006`, `action_effectiveness_seed42_20261005`, `closed_loop_stability_seed42_20261006`, and the `results/metrics/feedback_dev_seed42_20261004*` development/oracle artifacts. The only file created for this task is this document. No scientific code, checkpoints, datasets, splits, or results were modified; no experiments were rerun.
