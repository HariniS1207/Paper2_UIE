# Paper 2 frozen web demo

The local browser demo serves a small FastAPI backend and a static responsive interface. It loads the final enhancer and Baseline-B controller once at startup. It does not train, fine-tune, quantize, or write either model.

## Frozen checkpoints

| Model | File | SHA256 |
|---|---|---|
| Phase 2 controllable enhancer | `checkpoints/paper2_final_enhancer_seed42_20261006/best_endpoint.pth` | `D56DB81583DF7CD0056896B6ABFC46D82072E0606705981525B8AF588F6E43E9` |
| Phase 3 Baseline-B controller | `checkpoints/paper2_final_controller_baseline_b_seed42_20261006/controller.pth` | `C9E44B5D20C555572C09C6768B4BCEBC5CB06E4CD0D8704024FFBAE07DF1BDFC` |

Both files are SHA256-verified before deserialization. A missing or mismatched file fails application startup with the expected and observed hash. The enhancer loads strictly as `FinalControllableEnhancer(channels=3, features=64, blocks=4)` (298,947 parameters). Baseline B loads strictly as `Linear(43,3)` with the saved training-only feature mean and scale. Both models are in evaluation mode with gradients disabled. CUDA is used when available; CPU is the fallback.

## Setup and run

Use the project environment that contains PyTorch. In PowerShell from the repository root:

```powershell
C:\Users\Harini\TestTorchEnv\Scripts\python.exe -m pip install -r requirements-web-demo.txt
C:\Users\Harini\TestTorchEnv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The FastAPI app serves the HTML, CSS, and JavaScript at `/`; no Node installation or separate frontend server is required. The environment must also have the repository's PyTorch dependencies (`requirements.txt`).

## Interface modes

### Manual alpha

Upload a PNG, JPG/JPEG, or WEBP image and select any alpha in `[0,1]` with the slider, or use presets `0.00`, `0.25`, `0.50`, `0.75`, and `1.00`. The interface shows the original beside the selected output. **Compare five alpha levels** returns outputs at all five presets. Images can be clicked to view them larger.

At alpha zero the enhancer is the identity path, and the demo returns the oriented RGB source image unchanged as the displayed output.

### One-step controller

Choose an initial alpha (default `0.50`) and request a recommendation. The backend computes `Y_alpha`, forms the Phase 3 state

`z = [alpha, phi(X), phi(Y_alpha), phi(Y_alpha) - phi(X)]`,

normalizes it with the frozen controller's saved training mean and scale, and evaluates the three action logits. The feature descriptor `phi` exactly follows the Phase 3 14-value-per-image implementation and order. The selected action moves alpha by `h = 0.10`, clamped to `[0,1]`.

For selection, the Phase 3 rule masks the decrease score at alpha `0` and the increase score at alpha `1` before argmax. Displayed probabilities are the controller's original three-way softmax and sum to one; only the action-selection scores are masked. The interface labels the result **Controller recommendation** and shows current alpha, action, suggested alpha, all probabilities, and current/recommended outputs. It does not call the recommendation correct and does not claim an improvement for an arbitrary upload.

Automatic repeated correction is intentionally not implemented. Multi-step closed-loop stability was not established in the paper evaluation.

## API

All image endpoints accept multipart form data with an `image` field. Supported decoded formats are PNG, JPEG, and WEBP. The two alpha endpoints also accept an `alpha` form value.

| Endpoint | Request | Response |
|---|---|---|
| `GET /api/health` | none | Readiness, device, and verified checkpoint hashes |
| `POST /api/enhance` | `image`, `alpha` | Base64 PNG original/enhanced images, alpha, and inference metadata |
| `POST /api/compare-alphas` | `image` | Base64 PNG original and outputs keyed by the five alpha presets |
| `POST /api/controller` | `image`, optional `alpha` (default `0.5`) | Original/current/recommended images, action, current/suggested alpha, probabilities, metadata |

Invalid or corrupt images receive a JSON error response. Uploads are capped at 10 MB, 20 megapixels, and 12,000 pixels on either side. The app does not save uploads to application-managed files or retain source images after a request; Starlette's multipart parser may use a temporary spool while parsing a request. A bounded in-memory cache retains up to 32 derived 256×256 model outputs, keyed by upload SHA256 and alpha; it is cleared when the process exits.

## Preprocessing and image handling

The checkpoints were trained/evaluated with RGB images resized directly to 256×256 using PIL bilinear interpolation, converted to `float32`, divided by 255 to `[0,1]`, and converted from HWC to BCHW tensors. The demo reuses that canonical inference preprocessing and explicitly warns that the square resize can alter aspect ratio. Outputs are resized back to the EXIF-oriented uploaded dimensions for display. PNG/JPEG/WEBP files are decoded with Pillow; EXIF orientation is applied; transparent images are composited on white before RGB conversion. No crop, extra normalization, enhancement postprocess, or no-reference metric is applied.

Inference time and device are shown in the UI. Repeated identical requests can use the in-memory output cache.

## Reference images and limitations

This demo has no EUVP paired-reference selector. It does not calculate L1, PSNR, SSIM, UIQM, or UCIQE for arbitrary uploads: a target is unavailable, and no-reference scores were not added for this demo. A future paired-reference view must be a separate mode that only computes target-based measures when a valid EUVP target is explicitly paired.

The recommendation is a one-step prediction from the frozen research controller. It is not a guarantee of better perceived or reference-based quality. The final evaluation found modest one-step gains on its fixed EUVP test states and weaker behavior at high initial alpha. Arbitrary user images may differ from that evaluation distribution. Multi-step stability remains unresolved.

## Tests

Run the new demo checks:

```powershell
C:\Users\Harini\TestTorchEnv\Scripts\python.exe -m pytest tests\test_web_demo.py -v
```

Run the complete repository suite:

```powershell
C:\Users\Harini\TestTorchEnv\Scripts\python.exe -m pytest -v
```

Tests cover startup hash checks, frozen model loading, exact alpha-zero identity, output range, the alpha preset list, three-class controller shape, Phase 3 feasible-action masking and boundary behavior, invalid image rejection, API fields, determinism, normalized probabilities, and checkpoint immutability after inference.
