from __future__ import annotations

import base64
import hashlib
import io
import math
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from src.models.final_controllable_enhancer import FinalControllableEnhancer

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"
FAVICON = ROOT / "frontend" / "favicon.svg"
ENHANCER_PATH = ROOT / "checkpoints/paper2_final_enhancer_seed42_20261006/best_endpoint.pth"
CONTROLLER_PATH = ROOT / "checkpoints/paper2_final_controller_baseline_b_seed42_20261006/controller.pth"
ENHANCER_SHA256 = "D56DB81583DF7CD0056896B6ABFC46D82072E0606705981525B8AF588F6E43E9"
CONTROLLER_SHA256 = "C9E44B5D20C555572C09C6768B4BCEBC5CB06E4CD0D8704024FFBAE07DF1BDFC"
CLASS_ORDER = ("decrease", "hold", "increase")
ACTION_INDEX = {name: i for i, name in enumerate(CLASS_ORDER)}
PHI_SUFFIXES = (
    "mean_r", "std_r", "mean_g", "std_g", "mean_b", "std_b",
    "channel_mean_spread", "mean_luminance", "std_luminance", "luma_p05",
    "luma_p95", "gradient_mean", "near_zero", "near_one",
)
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
IMAGE_SIZE = 256
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_SIDE = 12_000
ALPHA_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
CACHE_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_checkpoint_hashes() -> dict[str, str]:
    found = {}
    for name, path, expected in (
        ("enhancer", ENHANCER_PATH, ENHANCER_SHA256),
        ("controller", CONTROLLER_PATH, CONTROLLER_SHA256),
    ):
        if not path.is_file():
            raise RuntimeError(f"Frozen {name} checkpoint is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen {name} checkpoint SHA256 mismatch: expected {expected}, got {actual}. "
                "The Paper 2 demo will not start."
            )
        found[name] = actual
    return found


def _load_models(device: torch.device):
    hashes = verify_checkpoint_hashes()
    # These local checkpoints are authenticated by the required SHA256 above; trusted
    # checkpoint metadata includes NumPy normalization arrays from the frozen run.
    enhancer_obj = torch.load(ENHANCER_PATH, map_location="cpu", weights_only=False)
    enhancer_state = enhancer_obj.get("model_state_dict", enhancer_obj.get("model_state", enhancer_obj))
    enhancer = FinalControllableEnhancer(channels=3, features=64, blocks=4)
    enhancer.load_state_dict(enhancer_state, strict=True)
    enhancer.eval().requires_grad_(False).to(device)

    controller_obj = torch.load(CONTROLLER_PATH, map_location="cpu", weights_only=False)
    if controller_obj.get("input_dim") != 43 or controller_obj.get("class_order") != list(CLASS_ORDER):
        raise RuntimeError("Frozen controller metadata is not the expected Linear(43,3) Baseline B")
    controller = torch.nn.Linear(43, 3)
    controller.load_state_dict(controller_obj["model_state_dict"], strict=True)
    controller.eval().requires_grad_(False).to(device)
    mean = torch.as_tensor(controller_obj["feature_mean"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(controller_obj["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    if mean.shape != (43,) or scale.shape != (43,):
        raise RuntimeError("Frozen controller normalization must contain 43 features")
    return enhancer, controller, mean, scale, hashes


def _parse_image(data: bytes):
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image uploads must be 10 MB or smaller.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format not in {"PNG", "JPEG", "WEBP"}:
                raise HTTPException(status_code=415, detail="Upload a PNG, JPG/JPEG, or WEBP image.")
            if opened.width < 1 or opened.height < 1:
                raise HTTPException(status_code=400, detail="The image has invalid dimensions.")
            if opened.width > MAX_SIDE or opened.height > MAX_SIDE or opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Images are limited to 12,000 px per side and 20 megapixels.")
            opened.load()
            image = ImageOps.exif_transpose(opened)
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                rgba = image.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(white, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            width, height = image.size
            # Training used PIL bilinear resize to 256x256; retain that exact canonical inference geometry.
            canonical = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            arr = np.asarray(canonical, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)
            key = hashlib.sha256(data).hexdigest()
            return image, tensor, key, (width, height)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The upload is not a valid, readable image.") from exc


def _features(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    def phi(image):
        means = image.mean(axis=(0, 1))
        stds = image.std(axis=(0, 1))
        luminance = image @ LUMA
        gy, gx = np.gradient(luminance)
        gradient = np.sqrt(gx * gx + gy * gy)
        return np.asarray([
            means[0], stds[0], means[1], stds[1], means[2], stds[2],
            means.max() - means.min(), luminance.mean(), luminance.std(),
            np.percentile(luminance, 5), np.percentile(luminance, 95), gradient.mean(),
            (image <= 0.01).mean(), (image >= 0.99).mean(),
        ], dtype=np.float32)
    x_phi, y_phi = phi(x), phi(y)
    return np.concatenate((np.asarray([alpha], dtype=np.float32), x_phi, y_phi, y_phi - x_phi))


def _encode_image(rgb: np.ndarray, output_size: tuple[int, int], identity_source: Image.Image | None = None) -> str:
    if identity_source is not None:
        image = identity_source
    else:
        arr = np.clip(rgb * 255.0, 0, 255).round().astype(np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        if image.size != output_size:
            image = image.resize(output_size, Image.Resampling.BILINEAR)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=False)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise HTTPException(status_code=422, detail="alpha must be a finite value in [0, 1].")
    return alpha


def select_feasible_action(probabilities, alpha: float) -> str:
    """Match Phase 3: mask infeasible probability scores only for argmax."""
    if len(probabilities) != 3:
        raise ValueError("The frozen controller must return three class probabilities")
    scores = np.asarray(probabilities, dtype=np.float64).copy()
    if alpha == 0.0:
        scores[ACTION_INDEX["decrease"]] = -np.inf
    if alpha == 1.0:
        scores[ACTION_INDEX["increase"]] = -np.inf
    return CLASS_ORDER[int(np.argmax(scores))]


class InferenceService:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enhancer, self.controller, self.feature_mean, self.feature_scale, self.hashes = _load_models(self.device)
        self.lock = RLock()
        self.cache: OrderedDict[tuple, np.ndarray] = OrderedDict()

    def _enhance(self, key: str, tensor: torch.Tensor, alpha: float) -> np.ndarray:
        cache_key = (key, round(alpha, 8))
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key].copy()
        with torch.inference_mode():
            y = self.enhancer(tensor.to(self.device), torch.tensor([alpha], dtype=torch.float32, device=self.device))
        result = y[0].detach().cpu().permute(1, 2, 0).numpy().copy()
        self.cache[cache_key] = result
        self.cache.move_to_end(cache_key)
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)
        return result.copy()

    def enhance(self, data: bytes, alpha: float) -> dict:
        alpha = _validate_alpha(alpha)
        source, tensor, key, dims = _parse_image(data)
        x = tensor[0].permute(1, 2, 0).numpy()
        start = time.perf_counter()
        with self.lock:
            y = self._enhance(key, tensor, alpha)
        elapsed = (time.perf_counter() - start) * 1000
        image = _encode_image(y, dims, source if alpha == 0.0 else None)
        return {"original_image": _encode_image(x, dims, source), "enhanced_image": image,
                "alpha": alpha, "metadata": self._metadata(dims, elapsed)}

    def compare(self, data: bytes) -> dict:
        source, tensor, key, dims = _parse_image(data)
        outputs = {}
        start = time.perf_counter()
        with self.lock:
            for alpha in ALPHA_LEVELS:
                outputs[f"{alpha:.2f}"] = _encode_image(
                    self._enhance(key, tensor, alpha), dims, source if alpha == 0 else None
                )
        elapsed = (time.perf_counter() - start) * 1000
        return {"original_image": _encode_image(np.empty((1, 1, 3)), dims, source), "outputs": outputs,
                "alphas": list(ALPHA_LEVELS), "metadata": self._metadata(dims, elapsed)}

    def control(self, data: bytes, alpha: float) -> dict:
        alpha = _validate_alpha(alpha)
        source, tensor, key, dims = _parse_image(data)
        x = tensor[0].permute(1, 2, 0).numpy()
        start = time.perf_counter()
        with self.lock:
            current = self._enhance(key, tensor, alpha)
            feature = torch.from_numpy(_features(x, current, alpha)).to(self.device)
            with torch.inference_mode():
                logits = self.controller(((feature - self.feature_mean) / self.feature_scale)[None, :])
                probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            action = select_feasible_action(probabilities, alpha)
            suggested = min(1.0, max(0.0, alpha + 0.1 * {"decrease": -1, "hold": 0, "increase": 1}[action]))
            recommended = self._enhance(key, tensor, suggested)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "original_image": _encode_image(np.empty((1, 1, 3)), dims, source),
            "current_image": _encode_image(current, dims, source if alpha == 0 else None),
            "recommended_image": _encode_image(recommended, dims, source if suggested == 0 else None),
            "action": action,
            "current_alpha": alpha,
            "suggested_alpha": suggested,
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_ORDER)},
            "recommendation_label": "Controller recommendation",
            "metadata": self._metadata(dims, elapsed),
        }

    def _metadata(self, dims, elapsed):
        return {"device": str(self.device), "inference_time_ms": round(elapsed, 2),
                "input_width": dims[0], "input_height": dims[1], "inference_width": IMAGE_SIZE,
                "inference_height": IMAGE_SIZE, "reference_available": False,
                "preprocessing": "EXIF orientation applied; RGB conversion; PIL bilinear resize to 256x256; float32 / 255; CHW tensor."}

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference = InferenceService()
    yield


app = FastAPI(title="Paper 2 Frozen Enhancement Demo", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(FRONTEND, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Browsers commonly probe the legacy path even when an SVG icon is linked.
    return Response(status_code=204)


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg():
    return FileResponse(FAVICON, media_type="image/svg+xml")


@app.get("/api/health")
def health():
    service = getattr(app.state, "inference", None)
    if service is None:
        return {"ready": False, "detail": "Models are not loaded."}
    return {"ready": True, "device": str(service.device), "checkpoint_sha256": service.hashes}


@app.post("/api/enhance")
async def enhance(image: UploadFile = File(...), alpha: float = Form(...)):
    service: InferenceService = getattr(app.state, "inference", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Frozen models are not loaded.")
    return service.enhance(await image.read(MAX_UPLOAD_BYTES + 1), alpha)


@app.post("/api/compare-alphas")
async def compare_alphas(image: UploadFile = File(...)):
    service: InferenceService = getattr(app.state, "inference", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Frozen models are not loaded.")
    return service.compare(await image.read(MAX_UPLOAD_BYTES + 1))


@app.post("/api/controller")
async def controller(image: UploadFile = File(...), alpha: float = Form(0.5)):
    service: InferenceService = getattr(app.state, "inference", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Frozen models are not loaded.")
    return service.control(await image.read(MAX_UPLOAD_BYTES + 1), alpha)
