# Paper 2 final experimental summary

## Validated evidence chain

- **Phase 1 — enhancer development:** Selected the epoch-7, seed-42 enhancer checkpoint using validation performance. It has 298,947 parameters and remains frozen.
- **Phase 2 — final enhancer evaluation:** Evaluated on the fixed 1,716-pair EUVP test set with a frozen, documented metric configuration. Full per-pair metrics, alpha sweep, bootstrap intervals, failures and qualitative examples remain in the Phase 2 results directory.
- **Phase 3 — controller training:** Trained Baseline B from scratch against that exact enhancer using the 43D `[alpha, phi(X), phi(Y), phi(Y)-phi(X)]` state and three directional actions.
- **Phase 3A — statistical audit:** Replayed predictions, labels and both checkpoint hashes; compared Baseline B with alpha-only using paired state bootstrap and signed-rank analysis. Decision: moderate support, small practical effect.
- **Phase 4 — final evaluation:** Consolidated frozen Phase-2 and Phase-3 results, recomputed primary paired comparisons, generated publication tables and figures, and made no model changes.

## Final result

At alpha=1, mean L1 changed from 0.115542 to 0.058939; mean PSNR from 17.102814 to 22.739205; mean SSIM from 0.786184 to 0.861900. UIQM/UCIQE remain no-reference descriptive measures.
One-step mean L1 improvement was 0.004048 for Baseline B and 0.003658 for alpha-only. Their paired mean difference was 0.00038999, 95% CI [0.00018776, 0.00059375].
At high initial alpha [0.8,1.0], Baseline B mean improvement was -0.000277; this limitation remains visible.

The valid claim is limited to one-step directional correction on this fixed test-state sample. Multi-step closed-loop stability is unresolved. The requested state bootstrap resamples 1,748 states; 32 explicit boundary states share 16 pair IDs with uniform-alpha states, so within-pair dependence may make intervals optimistic.

Frozen checkpoint hashes: enhancer `D56DB81583DF7CD0056896B6ABFC46D82072E0606705981525B8AF588F6E43E9`; controller `C9E44B5D20C555572C09C6768B4BCEBC5CB06E4CD0D8704024FFBAE07DF1BDFC`. Phase 4 environment: Python 3.12.0, PyTorch 2.13.0+cu130, CUDA 13.0, GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU.
