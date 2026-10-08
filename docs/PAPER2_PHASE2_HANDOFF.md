# Paper 2 Phase 2 handoff

## Frozen enhancer

- Model: `src/models/final_controllable_enhancer.py` (`FinalControllableEnhancer`)
- Checkpoint: `checkpoints/paper2_final_enhancer_seed42_20261006/best_endpoint.pth`
- Selected validation checkpoint: epoch 7, seed 42, 298,947 parameters
- SHA256: `D56DB81583DF7CD0056896B6ABFC46D82072E0606705981525B8AF588F6E43E9`
- The checkpoint was loaded strictly, evaluated in inference mode, and its hash was unchanged after the test evaluation. Do not retrain or replace it.

## Phase 2 evaluation

The frozen checkpoint was evaluated on the persisted EUVP paired split in manifest order: 8,004 train, 1,715 validation, and 1,716 held-out test pairs. The exact test rows are saved in `results/phase2_final_enhancer_evaluation_seed42_20261006/test_split_manifest_used.csv`; the source manifest SHA256 is `536C2956E65A65BF43E7BF753247F029455F1298B24165EEF0775011C0E613C8`.

At alpha=1.0, original versus enhanced test means were:

| Metric | Original | Enhanced | Paired difference |
|---|---:|---:|---:|
| L1 | 0.115542 | 0.058939 | +0.056603 improvement |
| PSNR (dB) | 17.102814 | 22.739205 | +5.636391 improvement |
| SSIM | 0.786184 | 0.861900 | +0.075716 improvement |
| UIQM | 2.530537 | -0.209480 | -2.740017 descriptive change |
| UCIQE | 26.884241 | 27.573231 | +0.688990 descriptive change |

L1, PSNR, and SSIM compare output with paired targets. UIQM and UCIQE are no-reference image-quality measures; their changes do not measure target similarity. Paired 95% percentile bootstrap intervals (10,000 samples, seed 42) are in `bootstrap_results.json` in the evaluation directory.

The alpha sweep at 0, 0.25, 0.50, 0.75, and 1.00 shows monotonically increasing mean L1-from-input (0, 0.024904, 0.049805, 0.074700, 0.099552). Mean target L1 decreases and PSNR/SSIM increase over those five sampled alpha values. UIQM first decreases and then rises; UCIQE changes only slightly before rising at alpha=1. These five-point trends do not establish general monotonic quality behavior. Alpha=0 is an exact identity by model contract and test coverage.

Failures are retained and listed in `failure_cases/failure_case_ids.json`: L1 worsened for 137 pairs, PSNR decreased for 138, and SSIM decreased for 134. Representative qualitative comparisons, including a failure case, are in `qualitative_examples/` and `failure_cases/`.

Metric definitions, implementation variants, color conventions, numerical handling, and references are frozen in `results/phase2_metric_audit_seed42_20261006/metric_config.json` and documented in `metric_audit.md`. Evaluation details and reproducibility artifacts are in `results/phase2_final_enhancer_evaluation_seed42_20261006/`.

## Phase boundary

**Controller training was NOT performed in Phase 2.** The old Baseline B controller was trained against a previous enhancer and is not the final controller for this frozen enhancer. Phase 3 must retrain Baseline B against this exact checkpoint. No Phase 3 work is authorized by this handoff.
