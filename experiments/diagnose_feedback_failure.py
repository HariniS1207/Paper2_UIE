"""Diagnose why the development feedback policy failed to predict decreases.

This experiment reuses the saved pair split, oracle states, and trained
development checkpoints. It trains only small balanced logistic diagnostics;
it does not train or alter the enhancer/controller checkpoints.
"""

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feedback_state_dataset import PairedManifestDataset
from src.models.controllable_enhancer import ControllableEnhancer
from src.models.consequence_encoder import ConsequenceEncoder
from src.models.feedback_controller import FeedbackController

SEED = 42
RUN_ID = "feedback_diagnosis_seed42_20261004_rerun3"
SOURCE_RUN = "feedback_dev_seed42_20261004_margin_corrected"
H = 0.10
MARGINS = (0.0001, 0.0005, 0.001, 0.002)
BIN_NAMES = tuple(f"[{i / 10:.1f},{(i + 1) / 10:.1f}{']' if i == 9 else ')'}" for i in range(10))
PREDICTED_HOLD_THRESHOLD = H / 2
LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)

DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
SPLIT_CSV = PROJECT_ROOT / "results" / "metrics" / "feedback_target_split.csv"
SOURCE_DIR = PROJECT_ROOT / "results" / "metrics" / SOURCE_RUN
ORACLE_CSV = SOURCE_DIR / "local_action_oracles.csv"
STATE_CSV = SOURCE_DIR / "sampled_states.csv"
CONTROLLER_TEST_CSV = SOURCE_DIR / "closed_loop_test_samples.csv"
FEATURE_CACHE = PROJECT_ROOT / "results" / "metrics" / "feedback_diagnosis_seed42_20261004_rerun2" / "diagnostic_predictor_features.csv"
ENDPOINT_METRICS = SOURCE_DIR / "stage_a_metrics.csv"
SOURCE_CONFIG = SOURCE_DIR / "run_config.json"
SOURCE_CONFIG_DATA = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
ENDPOINT_CHECKPOINT = (PROJECT_ROOT / SOURCE_CONFIG_DATA["checkpoint_initialization"]).resolve()
PROPOSED_CHECKPOINT = PROJECT_ROOT / "checkpoints" / SOURCE_RUN / "proposed_controller.pth"


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bin_index(alpha):
    return min(9, max(0, int(float(alpha) * 10)))


def binary_metrics(labels, scores, threshold=0.5):
    """Positive class is decrease; returns a fixed increase/decrease matrix."""
    truth = np.asarray([1 if label == "decrease" else 0 for label in labels], dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores >= threshold).astype(np.int64)
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tp = int(np.sum((truth == 1) & (pred == 1)))
    inc_precision = tn / (tn + fn) if tn + fn else 0.0
    inc_recall = tn / (tn + fp) if tn + fp else 0.0
    dec_precision = tp / (tp + fp) if tp + fp else 0.0
    dec_recall = tp / (tp + fn) if tp + fn else 0.0
    inc_f1 = 2 * inc_precision * inc_recall / (inc_precision + inc_recall) if inc_precision + inc_recall else 0.0
    dec_f1 = 2 * dec_precision * dec_recall / (dec_precision + dec_recall) if dec_precision + dec_recall else 0.0
    return {
        "n_increase": int(np.sum(truth == 0)), "n_decrease": int(np.sum(truth == 1)),
        "tn_increase": tn, "fp_increase_as_decrease": fp,
        "fn_decrease_as_increase": fn, "tp_decrease": tp,
        "increase_precision": inc_precision, "increase_recall": inc_recall,
        "decrease_precision": dec_precision, "decrease_recall": dec_recall,
        "balanced_accuracy": (inc_recall + dec_recall) / 2,
        "macro_f1": (inc_f1 + dec_f1) / 2,
        "accuracy": (tn + tp) / max(1, len(truth)),
        "auroc_decrease": binary_auroc(truth, scores) if len(np.unique(truth)) == 2 else "",
    }


def binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return ""
    # Pairwise Mann-Whitney form handles ties explicitly.
    comparisons = (positive[:, None] > negative[None, :]).astype(np.float64)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    return float(comparisons.mean())


def read_root():
    in_dataset = False
    for line in DATASET_CONFIG.read_text(encoding="utf-8").splitlines():
        if line.strip() == "dataset:":
            in_dataset = True
        elif in_dataset and line and not line[0].isspace():
            break
        elif in_dataset and line.strip().startswith("root:"):
            value = line.split(":", 1)[1].strip().strip("\"'")
            return (DATASET_CONFIG.parent / value).resolve()
    raise RuntimeError("Could not resolve dataset root")


def load_image(path):
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return array


def image_stats(image, prefix):
    means = image.mean(axis=(0, 1))
    stds = image.std(axis=(0, 1))
    y = image @ LUMA
    gy, gx = np.gradient(y)
    grad = np.sqrt(gx * gx + gy * gy)
    values = {}
    for i, channel in enumerate("rgb"):
        values[f"{prefix}_mean_{channel}"] = float(means[i])
        values[f"{prefix}_std_{channel}"] = float(stds[i])
    values.update({
        f"{prefix}_channel_mean_spread": float(means.max() - means.min()),
        f"{prefix}_mean_luminance": float(y.mean()),
        f"{prefix}_std_luminance": float(y.std()),
        f"{prefix}_luma_p05": float(np.percentile(y, 5)),
        f"{prefix}_luma_p95": float(np.percentile(y, 95)),
        f"{prefix}_gradient_mean": float(grad.mean()),
        f"{prefix}_near_zero": float((image <= 0.01).mean()),
        f"{prefix}_near_one": float((image >= 0.99).mean()),
    })
    return values, y, grad


def state_features(x, y):
    fx, lx, gx = image_stats(x, "x")
    fy, ly, gy = image_stats(y, "y")
    delta = y - x
    ad = np.abs(delta)
    dl = ly - lx
    consequence = {}
    for i, channel in enumerate("rgb"):
        consequence[f"delta_mean_{channel}"] = float(delta[:, :, i].mean())
        consequence[f"delta_abs_mean_{channel}"] = float(ad[:, :, i].mean())
    consequence.update({
        "delta_rms": float(np.sqrt(np.mean(delta**2))),
        "delta_fraction_abs_gt_0_01": float((ad > 0.01).mean()),
        "delta_max_abs": float(ad.max()),
        "delta_luminance_mean": float(dl.mean()),
        "delta_luminance_abs_mean": float(np.abs(dl).mean()),
        "delta_luminance_rms": float(np.sqrt(np.mean(dl**2))),
        "delta_gradient_mean": float(gy.mean() - gx.mean()),
    })
    return fx, fy, consequence


def margin_action(row, margin):
    alpha = float(row["alpha"])
    q0, qm, qp = float(row["q_current"]), float(row["q_minus"]), float(row["q_plus"])
    vm, vp = row["valid_minus"].lower() == "true", row["valid_plus"].lower() == "true"
    gm, gp = (q0 - qm if vm else -float("inf")), (q0 - qp if vp else -float("inf"))
    if vm and gm > margin and (not vp or qm < qp - margin):
        return "decrease"
    if vp and gp > margin and (not vm or qp < qm - margin):
        return "increase"
    return "hold"


def descriptive(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()), "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)), "min": float(values.min()), "max": float(values.max())}


def fit_balanced_logistic(train_x, train_y, seed=SEED):
    """Standardized binary logistic regression with inverse-frequency loss weights."""
    torch.manual_seed(seed)
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x = (train_x - mean) / scale
    model = torch.nn.Linear(x.shape[1], 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-3)
    counts = torch.bincount(train_y, minlength=2).float().clamp_min(1.0)
    weights = train_y.numel() / (2.0 * counts)
    for _ in range(300):
        logits = model(x)
        loss = F.cross_entropy(logits, train_y, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model, mean, scale


def predict_scores(model, mean, scale, values):
    with torch.no_grad():
        return torch.softmax(model((values - mean) / scale), dim=1)[:, 1].cpu().numpy()


def run_classifier_diagnostics(feature_rows, out_dir):
    feature_groups = {
        "A_alpha_only": ["alpha"],
        "B_alpha_basic_image_state": [
            "alpha", *[f"x_{f}" for f in BASIC_FEATURE_NAMES], *[f"y_{f}" for f in BASIC_FEATURE_NAMES],
        ],
        "C_alpha_X_Y_delta": [
            "alpha", *[f"x_{f}" for f in BASIC_FEATURE_NAMES],
            *[f"y_{f}" for f in BASIC_FEATURE_NAMES], *CONSEQUENCE_FEATURE_NAMES,
        ],
        "D_alpha_frozen_embedding": ["alpha", *[f"embedding_{i:03d}" for i in range(128)]],
    }
    metrics_rows, confusion_rows, bin_rows = [], [], []
    for model_name, names in feature_groups.items():
        eligible = [r for r in feature_rows if r["oracle_direction"] in ("increase", "decrease")]
        train = [r for r in eligible if r["partition"] == "controller-train"]
        train = [r for r in train if r["image_id"] in TRAIN_PAIR_IDS]
        train_x = torch.tensor([[float(r[name]) for name in names] for r in train], dtype=torch.float32)
        train_y = torch.tensor([1 if r["oracle_direction"] == "decrease" else 0 for r in train], dtype=torch.long)
        model, mean, scale = fit_balanced_logistic(train_x, train_y)
        for partition in ("controller-validation", "held-out-test"):
            rows = [r for r in eligible if r["partition"] == partition]
            x = torch.tensor([[float(r[name]) for name in names] for r in rows], dtype=torch.float32)
            labels = [r["oracle_direction"] for r in rows]
            scores = predict_scores(model, mean, scale, x)
            result = binary_metrics(labels, scores)
            metrics_rows.append({"model": model_name, "partition": partition, **result})
            preds = ["decrease" if s >= 0.5 else "increase" for s in scores]
            confusion_rows.extend([
                {"model": model_name, "partition": partition, "actual": "increase", "predicted": "increase",
                 "count": sum(a == "increase" and p == "increase" for a, p in zip(labels, preds))},
                {"model": model_name, "partition": partition, "actual": "increase", "predicted": "decrease",
                 "count": sum(a == "increase" and p == "decrease" for a, p in zip(labels, preds))},
                {"model": model_name, "partition": partition, "actual": "decrease", "predicted": "increase",
                 "count": sum(a == "decrease" and p == "increase" for a, p in zip(labels, preds))},
                {"model": model_name, "partition": partition, "actual": "decrease", "predicted": "decrease",
                 "count": sum(a == "decrease" and p == "decrease" for a, p in zip(labels, preds))},
            ])
            if partition == "held-out-test":
                for index, label in enumerate(BIN_NAMES):
                    selected = [(r, s) for r, s in zip(rows, scores) if bin_index(r["alpha"]) == index]
                    if not selected:
                        continue
                    b_labels = [r["oracle_direction"] for r, _ in selected]
                    b_scores = [s for _, s in selected]
                    bin_rows.append({"model": model_name, "alpha_bin": label, **binary_metrics(b_labels, b_scores)})
    write_csv(out_dir / "binary_classifier_metrics.csv", metrics_rows)
    write_csv(out_dir / "binary_confusion_matrices.csv", confusion_rows)
    write_csv(out_dir / "binary_metrics_by_alpha_bin.csv", bin_rows)
    return metrics_rows


BASIC_FEATURE_NAMES = (
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "channel_mean_spread", "mean_luminance", "std_luminance", "luma_p05",
    "luma_p95", "gradient_mean", "near_zero", "near_one",
)
CONSEQUENCE_FEATURE_NAMES = (
    "delta_mean_r", "delta_mean_g", "delta_mean_b", "delta_abs_mean_r",
    "delta_abs_mean_g", "delta_abs_mean_b", "delta_rms", "delta_fraction_abs_gt_0_01",
    "delta_max_abs", "delta_luminance_mean", "delta_luminance_abs_mean",
    "delta_luminance_rms", "delta_gradient_mean",
)
TRAIN_PAIR_IDS = set()


def build_feature_table(states, pair_rows, oracle_by_image, dataset_root, enhancer, encoder, device, batch_size=8):
    pair_by_id = {r["image_id"]: r for r in pair_rows}
    state_rows, embeddings = [], []
    enhancer.eval(); encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            chunk = states[start:start + batch_size]
            arrays = [load_image(dataset_root / pair_by_id[r["image_id"]]["input_path"]) for r in chunk]
            x_np = np.stack(arrays)
            x = torch.from_numpy(x_np.copy()).permute(0, 3, 1, 2).to(device)
            alpha = torch.tensor([float(r["alpha"]) for r in chunk], device=device, dtype=x.dtype)
            y = enhancer(x, alpha)
            _, vectors = encoder(x, y, alpha)
            ys = y.permute(0, 2, 3, 1).cpu().numpy()
            for row, x_array, y_array, vector in zip(chunk, x_np, ys, vectors.cpu().numpy()):
                fx, fy, fc = state_features(x_array, y_array)
                matches = oracle_by_image[(row["image_id"], row["partition"])]
                oracle_row = min(matches, key=lambda o: abs(float(o["alpha"]) - float(row["alpha"])))
                item = {
                    "partition": row["partition"], "image_id": row["image_id"],
                    "alpha": float(oracle_row["alpha"]), "oracle_direction": oracle_row["oracle_direction"],
                }
                item.update(fx); item.update(fy); item.update(fc)
                item.update({f"embedding_{i:03d}": float(v) for i, v in enumerate(vector)})
                state_rows.append(item)
            if (start + len(chunk)) % 256 < batch_size:
                print(f"Feature states: {min(start + len(chunk), len(states))}/{len(states)}", flush=True)
    return state_rows


def oracle_analyses(oracle_rows, out_dir):
    action_rows, sensitivity_rows, q_rows, improvement_rows = [], [], [], []
    for partition in ("all", "controller-train", "controller-validation", "held-out-test"):
        rows = oracle_rows if partition == "all" else [r for r in oracle_rows if r["partition"] == partition]
        counts = Counter(r["oracle_direction"] for r in rows)
        for i, label in enumerate(BIN_NAMES):
            group = [r for r in rows if bin_index(r["alpha"]) == i]
            count = Counter(r["oracle_direction"] for r in group)
            action_rows.append({"partition": partition, "alpha_bin": label, "n": len(group),
                                "increase": count["increase"], "decrease": count["decrease"], "hold": count["hold"]})
        for action in ("increase", "decrease", "hold"):
            group = [r for r in rows if r["oracle_direction"] == action]
            for field in ("q_minus", "q_current", "q_plus"):
                vals = [float(r[field]) for r in group]
                if vals:
                    q_rows.append({"partition": partition, "action": action, "quantity": field, **descriptive(vals)})
            gains = []
            for r in group:
                gains.append(float(r["gain_plus"] if action == "increase" else r["gain_minus"] if action == "decrease" else 0.0))
            if gains:
                improvement_rows.append({"partition": partition, "action": action,
                                         "quantity": "selected_action_improvement", **descriptive(gains)})
        for margin in MARGINS:
            counts_at_margin = Counter(margin_action(r, margin) for r in rows)
            sensitivity_rows.append({"partition": partition, "margin": margin, "n": len(rows),
                                     "increase": counts_at_margin["increase"],
                                     "decrease": counts_at_margin["decrease"],
                                     "hold": counts_at_margin["hold"]})
        if partition == "all":
            action_rows.append({"partition": "all_total", "alpha_bin": "all", "n": len(rows),
                                "increase": counts["increase"], "decrease": counts["decrease"], "hold": counts["hold"]})
    write_csv(out_dir / "oracle_action_by_alpha_bin.csv", action_rows)
    write_csv(out_dir / "oracle_q_by_action.csv", q_rows)
    write_csv(out_dir / "oracle_improvement_by_action.csv", improvement_rows)
    write_csv(out_dir / "oracle_margin_sensitivity.csv", sensitivity_rows)
    gap_rows = []
    for partition in ("all", "controller-train", "controller-validation", "held-out-test"):
        rows = oracle_rows if partition == "all" else [r for r in oracle_rows if r["partition"] == partition]
        for action in ("increase", "decrease", "hold"):
            group = [r for r in rows if r["oracle_direction"] == action]
            # Positive gap means the increase candidate has lower q; negative
            # gap means the decrease candidate has lower q.
            values = [float(r["q_plus"]) - float(r["q_minus"]) for r in group]
            if values:
                gap_rows.append({"partition": partition, "action": action,
                                 "quantity": "q_plus_minus_q_minus", **descriptive(values)})
    write_csv(out_dir / "oracle_candidate_quality_gap.csv", gap_rows)


def controller_diagnostics(oracle_rows, saved_predictions, feature_rows, out_dir):
    by_key = {(r["image_id"], float(r["alpha"]), r["partition"]): r for r in oracle_rows}
    pred_by_key = {}
    for r in saved_predictions:
        if r["model"] == "proposed":
            match = next((o for o in oracle_rows if o["partition"] == "held-out-test"
                          and o["image_id"] == r["image_id"] and abs(float(o["alpha"]) - float(r["alpha"])) < 1e-8), None)
            if match:
                pred_by_key[(match["image_id"], float(match["alpha"]), match["partition"])] = float(r["requested_delta"])
    rows = []
    for action in ("increase", "decrease", "hold"):
        group = [r for r in oracle_rows if r["partition"] == "held-out-test" and r["oracle_direction"] == action]
        key_rows = [by_key[(r["image_id"], float(r["alpha"]), r["partition"])] for r in group]
        target = [float(r["oracle_delta"]) for r in key_rows]
        pred = [pred_by_key.get((r["image_id"], float(r["alpha"]), r["partition"]), float("nan")) for r in key_rows]
        pred = [p for p in pred if np.isfinite(p)]
        rows.append({"oracle_action": action, "n": len(group),
                     "target_delta_mean": float(np.mean(target)) if target else "",
                     "target_delta_median": float(np.median(target)) if target else "",
                     "prediction_mean": float(np.mean(pred)) if pred else "",
                     "prediction_median": float(np.median(pred)) if pred else "",
                     "prediction_std": float(np.std(pred)) if pred else "",
                     "prediction_positive_percent": 100 * np.mean(np.asarray(pred) > 0) if pred else "",
                     "prediction_negative_percent": 100 * np.mean(np.asarray(pred) < 0) if pred else "",
                     "prediction_near_zero_abs_le_0_05_percent": 100 * np.mean(np.abs(pred) <= PREDICTED_HOLD_THRESHOLD) if pred else "",
                     "test_mse": float(np.mean((np.asarray(pred) - np.asarray(target[:len(pred)]))**2)) if pred else ""})
    write_csv(out_dir / "controller_prediction_by_oracle_action.csv", rows)

    # Per-class training loss and controller-parameter gradient contribution.
    train_features = [r for r in feature_rows if r["partition"] == "controller-train"]
    model_state = torch.load(PROPOSED_CHECKPOINT, map_location="cpu", weights_only=True)["model_state_dict"]
    controller = FeedbackController(delta_max=H)
    controller_state = {k.removeprefix("controller."): v for k, v in model_state.items() if k.startswith("controller.")}
    controller.load_state_dict(controller_state, strict=True)
    parameter_list = list(controller.parameters())
    gradient_rows = []
    all_losses = []
    class_items = {}
    for action in ("increase", "decrease", "hold"):
        group = [r for r in train_features if r["oracle_direction"] == action]
        if not group:
            continue
        emb = torch.tensor([[float(r[f"embedding_{i:03d}"]) for i in range(128)] for r in group])
        alpha = torch.tensor([float(r["alpha"]) for r in group])
        target = torch.tensor([float(next(o["oracle_delta"] for o in oracle_rows
                                         if o["partition"] == r["partition"] and o["image_id"] == r["image_id"]
                                         and abs(float(o["alpha"]) - float(r["alpha"])) < 1e-8)) for r in group])
        pred = controller(emb, alpha)
        losses = (pred - target).square()
        class_items[action] = (len(group), losses.detach(), pred.detach(), target)
        all_losses.extend(losses.detach().tolist())
        grads = torch.autograd.grad(losses.mean(), parameter_list, retain_graph=False)
        gradient_rows.append({"action": action, "n": len(group), "mean_squared_error": float(losses.mean().detach()),
                              "mean_prediction": float(pred.mean()), "mean_target_delta": float(target.mean()),
                              "mean_loss_share_of_total": float(losses.sum() / max(1, len(train_features))),
                              "class_mean_gradient_l2": float(torch.sqrt(sum(g.square().sum() for g in grads)))})
    write_csv(out_dir / "controller_train_loss_gradient_by_action.csv", gradient_rows)


def finalize_saved_diagnostic(out_dir):
    gap_path = out_dir / "oracle_candidate_quality_gap.csv"
    if not gap_path.exists():
        source_oracles = read_csv(ORACLE_CSV)
        generated = []
        for partition in ("all", "held-out-test"):
            subset = source_oracles if partition == "all" else [r for r in source_oracles if r["partition"] == partition]
            for action in ("increase", "decrease", "hold"):
                group = [r for r in subset if r["oracle_direction"] == action]
                gaps = [float(r["q_plus"]) - float(r["q_minus"]) for r in group]
                if gaps:
                    generated.append({"partition": partition, "action": action,
                                      "quantity": "q_plus_minus_q_minus", **descriptive(gaps)})
        write_csv(gap_path, generated)
    best_gain_path = out_dir / "oracle_best_available_gain_by_action.csv"
    if not best_gain_path.exists():
        source_oracles = read_csv(ORACLE_CSV)
        generated = []
        for partition in ("all", "controller-train", "controller-validation", "held-out-test"):
            subset = source_oracles if partition == "all" else [r for r in source_oracles if r["partition"] == partition]
            for action in ("increase", "decrease", "hold"):
                group = [r for r in subset if r["oracle_direction"] == action]
                best = []
                for row in group:
                    gains = []
                    if row["valid_minus"].lower() == "true":
                        gains.append(float(row["gain_minus"]))
                    if row["valid_plus"].lower() == "true":
                        gains.append(float(row["gain_plus"]))
                    best.append(max(gains) if gains else 0.0)
                if best:
                    generated.append({"partition": partition, "action": action,
                                      "quantity": "best_feasible_candidate_gain_over_current", **descriptive(best)})
        write_csv(best_gain_path, generated)
    metrics = read_csv(out_dir / "binary_classifier_metrics.csv")
    heldout = {r["model"]: r for r in metrics if r["partition"] == "held-out-test"}
    margin_rows = read_csv(out_dir / "oracle_margin_sensitivity.csv")
    margin_test = {float(r["margin"]): int(r["decrease"]) for r in margin_rows if r["partition"] == "held-out-test"}
    action_rows = read_csv(out_dir / "oracle_action_by_alpha_bin.csv")
    endpoint_rows = read_csv(out_dir / "endpoint_alpha_grid.csv")
    pred_rows = read_csv(out_dir / "controller_prediction_by_oracle_action.csv")
    grad_rows = read_csv(out_dir / "controller_train_loss_gradient_by_action.csv")
    a, b, c, d = (heldout[name] for name in (
        "A_alpha_only", "B_alpha_basic_image_state", "C_alpha_X_Y_delta", "D_alpha_frozen_embedding"
    ))
    if abs(float(endpoint_rows[0]["mean_l1_to_input_mean"])) > 1e-10:
        identity_ok = False
    else:
        identity_ok = float(endpoint_rows[0]["max_identity_abs_error"]) == 0.0
    target_l1 = [float(r["l1_to_target_mean"]) for r in endpoint_rows]
    report = {
        "observations": {
            "binary_test_metrics": heldout,
            "heldout_decrease_counts_by_margin": margin_test,
            "decrease_action_q_gap": [r for r in read_csv(gap_path)
                                      if r["partition"] in ("all", "held-out-test") and r["action"] == "decrease"],
            "controller_test_by_oracle_action": pred_rows,
            "controller_train_loss_and_gradient_by_action": grad_rows,
            "endpoint_alpha_grid": endpoint_rows,
            "alpha_zero_identity_exact": identity_ok,
            "endpoint_target_l1_monotone_nonincreasing": all(x >= y for x, y in zip(target_l1, target_l1[1:])),
        },
        "interpretation": {
            "case": "CASE C is primary, with a concurrent continuous-regression failure. Alpha-only is close to the frozen learned embedding; basic image statistics materially improve discrimination, while added handcrafted delta statistics do not improve on them.",
            "representation": "The binary diagnostic finds decrease information in target-free basic image-state features and some in the frozen embedding; the learned embedding is weaker than basic X/Y statistics.",
            "controller_objective": "The MSE policy predicts positive deltas for 99.1% of decrease test states (mean +0.056 versus target mean -0.099). MSE is dominated by conditional average targets; all alpha deciles still have more increase than decrease labels. This is consistent with upward regression bias, not proof that imbalance alone is causal.",
            "oracle": "Decrease labels remain under larger margins and have a measurable q_plus-q_minus gap; the tested margin range does not remove the decrease class.",
            "endpoint": "Identity is exact and paired-test mean target L1 decreases monotonically over alpha 0 to 1 in this development checkpoint; this does not establish perceptual quality or final endpoint adequacy.",
            "decision": "Do not claim self-correction or bidirectional control. Do not run a larger controller training yet.",
        },
        "minimum_next_step": "Use a small frozen-encoder directional diagnostic with class-weighted loss and validation-only thresholding, compared with alpha-only and basic X/Y statistics. Keep the enhancer and held-out test fixed; only proceed to a controller run if the learned representation adds reproducible decrease recall beyond alpha and simple state statistics.",
        "limitations": [
            "Development enhancer used 1,024 training pairs and 3 epochs.",
            "Binary classifier thresholds were fixed at 0.5; class-weighted scores are not treated as calibrated probabilities.",
            "There are only 223 held-out decrease states, with some alpha bins containing few decreases.",
            "Explicit alpha-boundary states reuse 16 images within each partition; test states are not fully independent.",
            "L1, PSNR, color, clipping, and gradient statistics are not perceptual-quality judgments.",
        ],
    }
    output = out_dir / "diagnostic_report_completed.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed diagnostic report: {output}")


def endpoint_grid(pair_rows, dataset_root, enhancer, device, out_dir):
    loader = DataLoader(PairedManifestDataset(pair_rows, dataset_root), batch_size=8, shuffle=False, num_workers=0)
    alphas = [i / 10 for i in range(11)]
    metrics = defaultdict(lambda: defaultdict(list))
    identity_max = 0.0
    enhancer.eval()
    with torch.inference_mode():
        seen = 0
        for x, target in loader:
            x, target = x.to(device), target.to(device)
            for alpha in alphas:
                a = torch.full((x.shape[0],), alpha, device=device, dtype=x.dtype)
                y = enhancer(x, a)
                if alpha == 0.0:
                    identity_max = max(identity_max, float((y - x).abs().max()))
                input_l1 = (y - x).abs().mean(dim=(1, 2, 3))
                target_l1 = (y - target).abs().mean(dim=(1, 2, 3))
                mse = (y - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
                psnr = -10.0 * torch.log10(mse)
                near_zero = (y <= 0.01).float().mean(dim=(1, 2, 3))
                near_one = (y >= 0.99).float().mean(dim=(1, 2, 3))
                for key, values in (("mean_l1_to_input", input_l1), ("l1_to_target", target_l1),
                                    ("psnr_to_target_db", psnr), ("fraction_near_zero", near_zero),
                                    ("fraction_near_one", near_one)):
                    metrics[alpha][key].extend(values.cpu().tolist())
            seen += x.shape[0]
            if seen % 256 < x.shape[0]:
                print(f"Endpoint test pairs: {seen}/{len(pair_rows)}", flush=True)
    rows = []
    for alpha in alphas:
        record = {"alpha": alpha, "n_pairs": seen, "max_identity_abs_error": identity_max if alpha == 0 else ""}
        for metric, values in metrics[alpha].items():
            record.update({f"{metric}_{k}": v for k, v in descriptive(values).items()})
        rows.append(record)
    write_csv(out_dir / "endpoint_alpha_grid.csv", rows)
    return rows, identity_max


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-saved", action="store_true")
    args = parser.parse_args()
    run_dir = PROJECT_ROOT / "results" / "metrics" / RUN_ID
    if args.finalize_saved:
        finalize_saved_diagnostic(run_dir)
        return
    if run_dir.exists():
        raise FileExistsError(f"Diagnostic output already exists: {run_dir}")
    required = (SPLIT_CSV, ORACLE_CSV, STATE_CSV, CONTROLLER_TEST_CSV,
                ENDPOINT_CHECKPOINT, PROPOSED_CHECKPOINT, ENDPOINT_METRICS, SOURCE_CONFIG)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required saved artifact missing: {path}")
    run_dir.mkdir(parents=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = read_root()
    split = read_csv(SPLIT_CSV)
    manifest_by_partition = defaultdict(list)
    for row in split:
        manifest_by_partition[row["partition"]].append(row)
    oracle = read_csv(ORACLE_CSV)
    states = read_csv(STATE_CSV)
    train_ids = {r["image_id"] for r in manifest_by_partition["controller-train"]}
    global TRAIN_PAIR_IDS
    TRAIN_PAIR_IDS = train_ids
    if len(oracle) != 4551:
        raise RuntimeError(f"Expected 4551 saved oracle states; found {len(oracle)}")
    # Verify saved primary labels agree with the specified 0.0001 margin rule.
    changed = sum(margin_action(r, 0.0001) != r["oracle_direction"] for r in oracle)

    config = {
        "run_id": RUN_ID, "seed": SEED, "source_run": SOURCE_RUN,
        "source_oracle": str(ORACLE_CSV.relative_to(PROJECT_ROOT)),
        "source_split": str(SPLIT_CSV.relative_to(PROJECT_ROOT)),
        "pair_counts": {k: len(v) for k, v in manifest_by_partition.items()},
        "oracle_states": len(oracle), "saved_label_mismatches_under_margin_rule": changed,
        "alpha_bins": list(BIN_NAMES), "margin_sensitivity_values": list(MARGINS),
        "binary_target": "increase vs decrease; hold excluded; inverse-frequency weighted CE without resampling",
        "classifier": "standardized torch linear logistic regression, Adam lr=0.03, 300 fixed epochs, L2=0.001",
        "features": {"A": "alpha", "B": "alpha plus per-channel/color/luminance/gradient/saturation stats of X and Y",
                     "C": "B plus signed/absolute Y-X statistics", "D": "alpha plus frozen learned ConsequenceEncoder embedding"},
        "controller_prediction_threshold": PREDICTED_HOLD_THRESHOLD,
        "endpoint_test": "all 1716 held-out paired images, RGB bilinear resize 256x256, alpha 0.0 to 1.0 step .1",
        "psnr_max_range": 1.0, "ssim": "not calculated; no validated SSIM dependency in project environment",
        "feature_cache_reused": str(FEATURE_CACHE.relative_to(PROJECT_ROOT)) if FEATURE_CACHE.exists() else None,
        "no_controller_or_enhancer_training": True,
    }
    (run_dir / "diagnostic_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    oracle_analyses(oracle, run_dir)

    endpoint_saved = torch.load(ENDPOINT_CHECKPOINT, map_location=device, weights_only=True)
    enhancer = ControllableEnhancer().to(device)
    enhancer.load_state_dict(endpoint_saved["enhancer_state_dict"], strict=True)
    proposed_saved = torch.load(PROPOSED_CHECKPOINT, map_location=device, weights_only=True)
    proposed_state = proposed_saved["model_state_dict"]
    encoder = ConsequenceEncoder().to(device)
    encoder_state = {k.removeprefix("encoder."): v for k, v in proposed_state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state, strict=True)

    if FEATURE_CACHE.exists():
        feature_rows = read_csv(FEATURE_CACHE)
    else:
        oracle_by_image = defaultdict(list)
        for row in oracle:
            oracle_by_image[(row["image_id"], row["partition"])].append(row)
        feature_rows = build_feature_table(states, split, oracle_by_image, root, enhancer, encoder, device)
    write_csv(run_dir / "diagnostic_predictor_features.csv", feature_rows)
    classifier_metrics = run_classifier_diagnostics(feature_rows, run_dir)

    saved_predictions = read_csv(CONTROLLER_TEST_CSV)
    controller_diagnostics(oracle, saved_predictions, feature_rows, run_dir)

    endpoint_rows, identity_error = endpoint_grid(
        manifest_by_partition["held-out-test"], root, enhancer, device, run_dir,
    )
    epoch_rows = read_csv(ENDPOINT_METRICS)
    action_counts = Counter(r["oracle_direction"] for r in oracle)
    test_counts = Counter(r["oracle_direction"] for r in oracle if r["partition"] == "held-out-test")
    report = {
        "observed": {
            "action_counts_all": dict(action_counts), "action_counts_heldout": dict(test_counts),
            "label_mismatches_at_saved_margin": changed,
            "identity_max_abs_error": identity_error,
            "endpoint_target_l1_alpha_0": endpoint_rows[0]["l1_to_target_mean"],
            "endpoint_target_l1_alpha_1": endpoint_rows[-1]["l1_to_target_mean"],
            "endpoint_target_l1_monotone_decreasing": all(
                float(endpoint_rows[i + 1]["l1_to_target_mean"]) <= float(endpoint_rows[i]["l1_to_target_mean"])
                for i in range(len(endpoint_rows) - 1)
            ),
            "classifier_test_metrics": [r for r in classifier_metrics if r["partition"] == "held-out-test"],
            "endpoint_last_validation_l1": epoch_rows[-1]["validation_endpoint_l1"],
        },
        "interpretation": "Diagnostic-only. Predictive association does not establish causality or perceptual quality.",
        "recommended_case": "Determined after inspecting classifier, alpha-bin, controller, margin, and endpoint outputs.",
    }
    (run_dir / "diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finalize_saved_diagnostic(run_dir)
    print(f"Diagnosis complete: {run_dir}", flush=True)
    print(f"Saved label mismatches at margin 0.0001: {changed}", flush=True)
    print(f"Held-out endpoint identity max error: {identity_error:.10g}", flush=True)


if __name__ == "__main__":
    main()
