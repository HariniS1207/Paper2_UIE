"""Development-scale staged experiment for consequence-based feedback control.

The frozen Stage-A enhancer creates train/validation/test local-action oracles.
Three Stage-C regressors are compared without class balancing: alpha-only,
image-state (X,Y), and proposed target-free [X,Y,Y-X,alpha].
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feedback_state_dataset import OracleStateDataset, PairedManifestDataset
from src.losses.alpha import EndpointLoss, IdentityLoss
from src.models.closed_loop import OneStepClosedLoop
from src.models.controllable_enhancer import ControllableEnhancer
from src.models.consequence_encoder import ConsequenceEncoder
from src.models.feedback_controller import (
    AlphaOnlyController,
    FeedbackController,
    ImageStateController,
)
from src.training.feedback_oracle import choose_local_action
from src.training.train_step import train_step


SEED = 42
H = 0.10
MARGIN = 1e-4
TRAIN_PAIRS = 1024
BOUNDARY_PAIRS_PER_PARTITION = 16
ENHANCER_EPOCHS = 3
CONTROLLER_EPOCHS = 5
ENHANCER_BATCH_SIZE = 8
CONTROLLER_BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DELTA_MAX = H
PREDICTED_HOLD_THRESHOLD = H / 2

DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
SPLIT_MANIFEST = PROJECT_ROOT / "results" / "metrics" / "feedback_target_split.csv"


class ProposedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConsequenceEncoder()
        self.controller = FeedbackController(delta_max=DELTA_MAX)

    def forward(self, x, y, alpha):
        _, vector = self.encoder(x, y, alpha)
        return self.controller(vector, alpha)


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


def dataset_root():
    in_dataset = False
    for line in DATASET_CONFIG.read_text(encoding="utf-8").splitlines():
        if line.strip() == "dataset:":
            in_dataset = True
        elif in_dataset and line and not line[0].isspace():
            break
        elif in_dataset and line.strip().startswith("root:"):
            value = line.split(":", 1)[1].strip().strip("\"'")
            return (DATASET_CONFIG.parent / value).resolve()
    raise RuntimeError(f"Could not read dataset.root from {DATASET_CONFIG}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_rgb(path):
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def pair_batch(rows, root, device):
    x = torch.stack([load_rgb(root / row["input_path"]) for row in rows]).to(device)
    target = torch.stack([load_rgb(root / row["target_path"]) for row in rows]).to(device)
    return x, target


def sample_states(partitions, train_limit, seed):
    rng = random.Random(seed)
    states = []
    chosen_train = sorted(partitions["controller-train"], key=lambda r: r["image_id"])
    rng.shuffle(chosen_train)
    chosen_train = chosen_train[: min(train_limit, len(chosen_train))]
    selected = {
        "controller-train": chosen_train,
        "controller-validation": partitions["controller-validation"],
        "held-out-test": partitions["held-out-test"],
    }
    boundary_ids = {}
    for partition_index, partition in enumerate(selected):
        rows = sorted(selected[partition], key=lambda r: r["image_id"])
        alpha_rng = random.Random(seed + 100 + partition_index)
        for row in rows:
            states.append({**row, "partition": partition,
                           "alpha": alpha_rng.uniform(0.0, 1.0),
                           "alpha_source": "uniform_per_pair"})
        boundaries = rows[: min(BOUNDARY_PAIRS_PER_PARTITION, len(rows))]
        boundary_ids[partition] = [r["image_id"] for r in boundaries]
        for row in boundaries:
            for alpha in (0.0, 1.0):
                states.append({**row, "partition": partition, "alpha": alpha,
                               "alpha_source": "explicit_boundary"})
    return states, selected, boundary_ids


def train_endpoint(model, train_rows, validation_rows, root, device, epochs, batch_size):
    train_loader = DataLoader(
        PairedManifestDataset(train_rows, root), batch_size=batch_size,
        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        PairedManifestDataset(validation_rows, root), batch_size=batch_size,
        shuffle=False, num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    log = []
    for epoch in range(1, epochs + 1):
        totals = Counter()
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            losses = train_step(model, optimizer, x, target, alpha=None)
            for key, value in losses.items():
                totals[key] += float(value.item())
        model.eval()
        val_identity, val_endpoint, batches = 0.0, 0.0, 0
        with torch.inference_mode():
            for x, target in val_loader:
                x, target = x.to(device), target.to(device)
                zero = torch.zeros(x.shape[0], device=device)
                one = torch.ones(x.shape[0], device=device)
                val_identity += float(IdentityLoss()(model(x, zero), x).item())
                val_endpoint += float(EndpointLoss()(model(x, one), target).item())
                batches += 1
        count = len(train_loader)
        log.append({
            "epoch": epoch,
            "train_total_loss": totals["loss"] / count,
            "train_identity_loss": totals["identity_loss"] / count,
            "train_endpoint_loss": totals["endpoint_loss"] / count,
            "validation_identity_l1": val_identity / batches,
            "validation_endpoint_l1": val_endpoint / batches,
        })
        print(f"Stage A epoch {epoch}/{epochs}: train endpoint L1="
              f"{log[-1]['train_endpoint_loss']:.6f}; validation endpoint L1="
              f"{log[-1]['validation_endpoint_l1']:.6f}", flush=True)
    return log


def generate_oracles(model, states, root, device, batch_size):
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            batch = states[start:start + batch_size]
            x, target = pair_batch(batch, root, device)
            alpha = torch.tensor([r["alpha"] for r in batch], device=device, dtype=x.dtype)
            minus = (alpha - H).clamp(0.0, 1.0)
            plus = (alpha + H).clamp(0.0, 1.0)
            current_y = model(x, alpha)
            minus_y = model(x, minus)
            plus_y = model(x, plus)
            q0 = (current_y - target).abs().mean(dim=(1, 2, 3)).cpu().tolist()
            qm = (minus_y - target).abs().mean(dim=(1, 2, 3)).cpu().tolist()
            qp = (plus_y - target).abs().mean(dim=(1, 2, 3)).cpu().tolist()
            for state, a, q_current, q_minus, q_plus in zip(batch, alpha.tolist(), q0, qm, qp):
                decision = choose_local_action(a, q_current, q_minus, q_plus, H, MARGIN)
                rows.append({
                    "partition": state["partition"], "image_id": state["image_id"],
                    "input_path": state["input_path"], "target_path": state["target_path"],
                    "alpha": a, "alpha_minus": max(0.0, a - H),
                    "alpha_plus": min(1.0, a + H), "alpha_source": state["alpha_source"],
                    "q_current": q_current, "q_minus": q_minus, "q_plus": q_plus,
                    "gain_minus": decision.gain_minus, "gain_plus": decision.gain_plus,
                    "valid_minus": decision.valid_minus, "valid_plus": decision.valid_plus,
                    "oracle_direction": decision.direction, "oracle_delta": decision.delta,
                })
    return rows


def build_policy(kind, device):
    if kind == "alpha_only":
        return AlphaOnlyController(delta_max=DELTA_MAX).to(device)
    if kind == "image_state":
        return ImageStateController(delta_max=DELTA_MAX).to(device)
    return ProposedPolicy().to(device)


def predict_batch(policy, kind, enhancer, x, alpha, device):
    if kind == "alpha_only":
        return policy(alpha)
    with torch.no_grad():
        y = enhancer(x, alpha)
    if kind == "image_state":
        return policy(x, y, alpha)
    return policy(x, y, alpha)


def train_controller(kind, rows, validation_rows, root, device, epochs, batch_size):
    policy = build_policy(kind, device)
    enhancer = ControllableEnhancer().to(device)
    enhancer.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)["enhancer_state_dict"])
    enhancer.eval()
    for parameter in enhancer.parameters():
        parameter.requires_grad_(False)
    train_loader = DataLoader(OracleStateDataset(rows, root), batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(SEED + 9))
    val_loader = DataLoader(OracleStateDataset(validation_rows, root), batch_size=batch_size,
                            shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    log = []
    for epoch in range(1, epochs + 1):
        policy.train()
        total, batches = 0.0, 0
        for x, alpha, delta in train_loader:
            x, alpha, delta = x.to(device), alpha.to(device), delta.to(device)
            with torch.no_grad():
                y = enhancer(x, alpha)
            if kind == "alpha_only":
                prediction = policy(alpha)
            else:
                prediction = policy(x, y, alpha)
            loss = torch.nn.functional.mse_loss(prediction, delta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        policy.eval()
        val_total, val_batches = 0.0, 0
        with torch.inference_mode():
            for x, alpha, delta in val_loader:
                x, alpha, delta = x.to(device), alpha.to(device), delta.to(device)
                prediction = predict_batch(policy, kind, enhancer, x, alpha, device)
                val_total += float(torch.nn.functional.mse_loss(prediction, delta).item())
                val_batches += 1
        log.append({"epoch": epoch, "train_delta_mse": total / batches,
                    "validation_delta_mse": val_total / val_batches})
        print(f"Stage C {kind} epoch {epoch}/{epochs}: validation delta MSE="
              f"{log[-1]['validation_delta_mse']:.6f}", flush=True)
    return policy, log


def direction_from_delta(delta):
    if delta > PREDICTED_HOLD_THRESHOLD:
        return "increase"
    if delta < -PREDICTED_HOLD_THRESHOLD:
        return "decrease"
    return "hold"


def macro_scores(truth, predicted):
    labels = ("increase", "decrease", "hold")
    f1s, recalls = [], []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        support = sum(t == label for t in truth)
        if support:
            recalls.append(tp / support)
            f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return (float(np.mean(recalls)) if recalls else 0.0,
            float(np.mean(f1s)) if f1s else 0.0)


def evaluate_closed_loop(kind, policy, test_rows, root, device, batch_size):
    enhancer = ControllableEnhancer().to(device)
    enhancer.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)["enhancer_state_dict"])
    enhancer.eval()
    policy.eval()
    samples = []
    with torch.inference_mode():
        for start in range(0, len(test_rows), batch_size):
            batch = test_rows[start:start + batch_size]
            x, target = pair_batch(batch, root, device)
            alpha = torch.tensor([float(r["alpha"]) for r in batch], device=device)
            y0 = enhancer(x, alpha)
            if kind == "alpha_only":
                requested = policy(alpha)
            else:
                requested = policy(x, y0, alpha)
            updated = (alpha + requested).clamp(0.0, 1.0)
            y1 = enhancer(x, updated)
            q0 = (y0 - target).abs().mean(dim=(1, 2, 3)).cpu().tolist()
            q1 = (y1 - target).abs().mean(dim=(1, 2, 3)).cpu().tolist()
            for row, a, request, new_alpha, before, after in zip(
                batch, alpha.cpu().tolist(), requested.cpu().tolist(), updated.cpu().tolist(), q0, q1
            ):
                samples.append({
                    "model": kind, "image_id": row["image_id"], "alpha": a,
                    "requested_delta": request, "applied_delta": new_alpha - a,
                    "alpha_after": new_alpha, "oracle_direction": row["oracle_direction"],
                    "predicted_direction": direction_from_delta(request),
                    "q_before": before, "q_after": after, "q_change": after - before,
                    "improved": after < before,
                })
    truth = [r["oracle_direction"] for r in samples]
    predicted = [r["predicted_direction"] for r in samples]
    balanced_accuracy, macro_f1 = macro_scores(truth, predicted)
    summary = [{
        "model": kind,
        "test_states": len(samples),
        "mean_q_change": float(np.mean([r["q_change"] for r in samples])),
        "median_q_change": float(np.median([r["q_change"] for r in samples])),
        "success_rate_percent": 100.0 * np.mean([r["improved"] for r in samples]),
        "direction_accuracy_percent": 100.0 * np.mean([a == b for a, b in zip(truth, predicted)]),
        "balanced_direction_accuracy_percent": 100.0 * balanced_accuracy,
        "macro_direction_f1": macro_f1,
        "mean_abs_requested_delta": float(np.mean([abs(r["requested_delta"]) for r in samples])),
        "mean_abs_applied_delta": float(np.mean([abs(r["applied_delta"]) for r in samples])),
        "increase_percent": 100.0 * predicted.count("increase") / len(predicted),
        "decrease_percent": 100.0 * predicted.count("decrease") / len(predicted),
        "hold_percent": 100.0 * predicted.count("hold") / len(predicted),
    }]
    bins = ((0.0, 0.2, "[0.0,0.2)"), (0.2, 0.4, "[0.2,0.4)"),
            (0.4, 0.6, "[0.4,0.6)"), (0.6, 0.8, "[0.6,0.8)"),
            (0.8, 1.000001, "[0.8,1.0]"))
    for low, high, label in bins:
        group = [r for r in samples if low <= r["alpha"] < high]
        if group:
            summary.append({
                "model": kind, "test_states": len(group), "alpha_bin": label,
                "mean_q_change": float(np.mean([r["q_change"] for r in group])),
                "median_q_change": float(np.median([r["q_change"] for r in group])),
                "success_rate_percent": 100.0 * np.mean([r["improved"] for r in group]),
                "increase_percent": 100.0 * sum(r["predicted_direction"] == "increase" for r in group) / len(group),
                "decrease_percent": 100.0 * sum(r["predicted_direction"] == "decrease" for r in group) / len(group),
                "hold_percent": 100.0 * sum(r["predicted_direction"] == "hold" for r in group) / len(group),
            })
    for boundary, label in ((0.0, "alpha_zero"), (1.0, "alpha_one")):
        group = [r for r in samples if r["alpha"] == boundary]
        if group:
            summary.append({
                "model": kind, "test_states": len(group), "alpha_bin": label,
                "mean_q_change": float(np.mean([r["q_change"] for r in group])),
                "median_q_change": float(np.median([r["q_change"] for r in group])),
                "success_rate_percent": 100.0 * np.mean([r["improved"] for r in group]),
                "mean_abs_applied_delta": float(np.mean([abs(r["applied_delta"]) for r in group])),
            })
    return samples, summary


def summarize_saved_run(run_dir):
    """Complete aggregate reporting from saved held-out predictions only."""
    path = run_dir / "closed_loop_test_samples.csv"
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"No saved test predictions in {path}")
    result_rows = []
    for kind in ("alpha_only", "image_state", "proposed"):
        model_rows = [r for r in rows if r["model"] == kind]
        if not model_rows:
            raise RuntimeError(f"Missing saved predictions for {kind}")
        subsets = [("overall", model_rows)]
        for low, high, label in (
            (0.0, 0.2, "[0.0,0.2)"), (0.2, 0.4, "[0.2,0.4)"),
            (0.4, 0.6, "[0.4,0.6)"), (0.6, 0.8, "[0.6,0.8)"),
            (0.8, 1.000001, "[0.8,1.0]"),
        ):
            subsets.append((label, [r for r in model_rows if low <= float(r["alpha"]) < high]))
        for boundary, label in ((0.0, "alpha_zero"), (1.0, "alpha_one")):
            subsets.append((label, [r for r in model_rows if float(r["alpha"]) == boundary]))
        for label, group in subsets:
            if not group:
                continue
            truth = [r["oracle_direction"] for r in group]
            predicted = [r["predicted_direction"] for r in group]
            balanced, macro_f1 = macro_scores(truth, predicted)
            row = {
                "model": kind, "subset": label, "test_states": len(group),
                "mean_q_change": float(np.mean([float(r["q_change"]) for r in group])),
                "median_q_change": float(np.median([float(r["q_change"]) for r in group])),
                "success_rate_percent": 100.0 * np.mean([r["improved"].lower() == "true" for r in group]),
                "direction_accuracy_percent": 100.0 * np.mean([a == b for a, b in zip(truth, predicted)]),
                "balanced_direction_accuracy_percent": 100.0 * balanced,
                "macro_direction_f1": macro_f1,
                "mean_abs_requested_delta": float(np.mean([abs(float(r["requested_delta"])) for r in group])),
                "mean_abs_applied_delta": float(np.mean([abs(float(r["applied_delta"])) for r in group])),
                "increase_percent": 100.0 * sum(p == "increase" for p in predicted) / len(group),
                "decrease_percent": 100.0 * sum(p == "decrease" for p in predicted) / len(group),
                "hold_percent": 100.0 * sum(p == "hold" for p in predicted) / len(group),
                "oracle_increase_percent": 100.0 * sum(p == "increase" for p in truth) / len(group),
                "oracle_decrease_percent": 100.0 * sum(p == "decrease" for p in truth) / len(group),
                "oracle_hold_percent": 100.0 * sum(p == "hold" for p in truth) / len(group),
            }
            result_rows.append(row)
    summary_path = run_dir / "closed_loop_test_summary_completed.csv"
    write_csv(summary_path, result_rows)
    overall = [r for r in result_rows if r["subset"] == "overall"]
    report = {
        "source": str(path.relative_to(PROJECT_ROOT)),
        "completion_note": "Aggregated from the already saved held-out predictions; no model was retrained and no predictions were changed.",
        "overall": overall,
    }
    report_path = run_dir / "final_report_completed.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed summary: {summary_path}")
    print(f"Completed report: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="feedback_dev_seed42_20261004")
    parser.add_argument("--train-pairs", type=int, default=TRAIN_PAIRS)
    parser.add_argument("--enhancer-epochs", type=int, default=ENHANCER_EPOCHS)
    parser.add_argument("--controller-epochs", type=int, default=CONTROLLER_EPOCHS)
    parser.add_argument("--reuse-enhancer", help="Reuse a previously trained endpoint_enhancer.pth from this same staged experiment.")
    parser.add_argument("--summarize-saved", help="Summarize an existing run's saved closed-loop test predictions without rerunning models.")
    args = parser.parse_args()

    if args.summarize_saved:
        summarize_saved_run(PROJECT_ROOT / "results" / "metrics" / args.summarize_saved)
        return

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = dataset_root()
    run_dir = PROJECT_ROOT / "results" / "metrics" / args.run_id
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.run_id
    if run_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Run output already exists; choose a new --run-id: {args.run_id}")
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    manifest = read_csv(SPLIT_MANIFEST)
    by_partition = defaultdict(list)
    seen_ids = set()
    for row in manifest:
        if row["image_id"] in seen_ids:
            raise RuntimeError(f"Duplicate image pair in split manifest: {row['image_id']}")
        seen_ids.add(row["image_id"])
        by_partition[row["partition"]].append(row)
    expected = {"controller-train": 8004, "controller-validation": 1715, "held-out-test": 1716}
    for name, count in expected.items():
        if len(by_partition[name]) != count:
            raise RuntimeError(f"Expected {count} {name} pairs, found {len(by_partition[name])}")

    states, selected_pairs, boundary_ids = sample_states(by_partition, args.train_pairs, SEED)
    train_states = [r for r in states if r["partition"] == "controller-train"]
    validation_states = [r for r in states if r["partition"] == "controller-validation"]
    test_states = [r for r in states if r["partition"] == "held-out-test"]
    write_csv(run_dir / "development_train_pairs.csv", selected_pairs["controller-train"])

    config = {
        "experiment": "staged_consequence_feedback_development",
        "run_id": args.run_id, "seed": SEED, "device": str(device),
        "checkpoint_initialization": "random initialization; paper2_epoch_001.pth not loaded",
        "dataset_root": str(root), "source_split_manifest": str(SPLIT_MANIFEST.relative_to(PROJECT_ROOT)),
        "source_pair_counts": expected,
        "development_train_pair_count": len(selected_pairs["controller-train"]),
        "state_counts": {"train": len(train_states), "validation": len(validation_states), "test": len(test_states)},
        "alpha_sampling": "one seed-42 Uniform[0,1) alpha per pair plus alpha=0 and alpha=1 on first 16 sorted pairs per partition",
        "preprocessing": "RGB, bilinear resize to 256x256, float32 [0,1]",
        "h": H, "margin_l1": MARGIN,
        "margin_rationale": "fixed above the observed Step-1 near-numerical gain tail; applied to improvement over current q",
        "enhancer_epochs": args.enhancer_epochs, "controller_epochs": args.controller_epochs,
        "enhancer_batch_size": ENHANCER_BATCH_SIZE, "controller_batch_size": CONTROLLER_BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "loss": "endpoint L1 at alpha=1 plus identity L1 at alpha=0 (structurally zero for this enhancer)",
        "controller_target": "best feasible local alpha change if q_current - q_candidate > margin, otherwise zero",
        "delta_max": DELTA_MAX, "class_balancing": False,
        "predicted_direction_hold_threshold": PREDICTED_HOLD_THRESHOLD,
        "boundary_pair_ids": boundary_ids,
        "oracle_inputs_to_policy": False,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_csv(run_dir / "sampled_states.csv", states)

    enhancer = ControllableEnhancer().to(device)
    initial_endpoint = []
    if args.reuse_enhancer:
        enhancer_checkpoint = (PROJECT_ROOT / args.reuse_enhancer).resolve()
        saved = torch.load(enhancer_checkpoint, map_location=device, weights_only=True)
        enhancer.load_state_dict(saved["enhancer_state_dict"], strict=True)
        source_run_id = enhancer_checkpoint.parent.name
        endpoint_log = read_csv(PROJECT_ROOT / "results" / "metrics" / source_run_id / "stage_a_metrics.csv")
        config["checkpoint_initialization"] = str(enhancer_checkpoint.relative_to(PROJECT_ROOT))
        (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    else:
        val_loader = DataLoader(PairedManifestDataset(selected_pairs["controller-validation"], root),
                                batch_size=ENHANCER_BATCH_SIZE, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for x, target in val_loader:
                x, target = x.to(device), target.to(device)
                initial_endpoint.append(float((enhancer(x, torch.ones(x.shape[0], device=device)) - target).abs().mean().item()))
        endpoint_log = train_endpoint(
            enhancer, selected_pairs["controller-train"], selected_pairs["controller-validation"],
            root, device, args.enhancer_epochs, ENHANCER_BATCH_SIZE,
        )
        enhancer_checkpoint = checkpoint_dir / "endpoint_enhancer.pth"
        torch.save({"epoch": args.enhancer_epochs, "seed": SEED,
                    "enhancer_state_dict": enhancer.state_dict(),
                    "experiment": "endpoint_enhancer_development"}, enhancer_checkpoint)
    global CHECKPOINT_PATH
    CHECKPOINT_PATH = enhancer_checkpoint
    enhancer.eval()
    write_csv(run_dir / "stage_a_metrics.csv", endpoint_log)

    oracle_train = generate_oracles(enhancer, train_states, root, device, ENHANCER_BATCH_SIZE)
    oracle_validation = generate_oracles(enhancer, validation_states, root, device, ENHANCER_BATCH_SIZE)
    oracle_test = generate_oracles(enhancer, test_states, root, device, ENHANCER_BATCH_SIZE)
    oracle_rows = oracle_train + oracle_validation + oracle_test
    write_csv(run_dir / "local_action_oracles.csv", oracle_rows)
    print("Oracle action counts:", dict(Counter(r["oracle_direction"] for r in oracle_rows)), flush=True)

    train_oracles = [r for r in oracle_rows if r["partition"] == "controller-train"]
    validation_oracles = [r for r in oracle_rows if r["partition"] == "controller-validation"]
    test_oracles = [r for r in oracle_rows if r["partition"] == "held-out-test"]
    model_samples, model_summaries = [], []
    for kind in ("alpha_only", "image_state", "proposed"):
        seed_everything(SEED)
        policy, log = train_controller(
            kind, train_oracles, validation_oracles, root, device,
            args.controller_epochs, CONTROLLER_BATCH_SIZE,
        )
        torch.save({"model_state_dict": policy.state_dict(), "model": kind,
                    "seed": SEED, "epochs": args.controller_epochs},
                   checkpoint_dir / f"{kind}_controller.pth")
        write_csv(run_dir / f"stage_c_{kind}_metrics.csv", log)
        samples, summary = evaluate_closed_loop(kind, policy, test_oracles, root, device, CONTROLLER_BATCH_SIZE)
        model_samples.extend(samples)
        model_summaries.extend(summary)
        print(f"Stage D {kind}: mean Q change={summary[0]['mean_q_change']:.6f}; "
              f"success={summary[0]['success_rate_percent']:.2f}%", flush=True)

    write_csv(run_dir / "closed_loop_test_samples.csv", model_samples)
    write_csv(run_dir / "closed_loop_test_summary.csv", model_summaries)
    report = {
        "device": str(device), "run_dir": str(run_dir), "checkpoint_dir": str(checkpoint_dir),
        "initial_random_endpoint_l1_mean_over_validation_batches": float(np.mean(initial_endpoint)) if initial_endpoint else None,
        "trained_endpoint_l1_last_epoch": endpoint_log[-1]["validation_endpoint_l1"],
        "identity_l1_last_epoch": endpoint_log[-1]["validation_identity_l1"],
        "oracle_action_counts": dict(Counter(r["oracle_direction"] for r in oracle_rows)),
        "controller_test_overall": [r for r in model_summaries if "alpha_bin" not in r],
        "test_split_used_once_for_final_evaluation": True,
    }
    (run_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed. Metrics: {run_dir}", flush=True)
    print(f"New trained checkpoints: {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    CHECKPOINT_PATH = None
    main()
