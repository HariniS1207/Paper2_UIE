"""Measure local paired-target L1 gains for the existing Paper 2 enhancer.

This is a diagnostic only: it does not train models or modify dataset files.
Run from any directory with the project's Python environment.
"""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.controllable_enhancer import ControllableEnhancer


DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "paper2_epoch_001.pth"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
SAMPLES_CSV = METRICS_DIR / "feedback_target_samples.csv"
SUMMARY_CSV = METRICS_DIR / "feedback_target_summary.csv"
SPLIT_CSV = METRICS_DIR / "feedback_target_split.csv"
SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
STATE_COUNT = 600
ALPHA_LOW = 0.05
ALPHA_HIGH = 0.95
STEP = 0.10
ALPHA_BINS = (
    (0.05, 0.20, "[0.05,0.20)"),
    (0.20, 0.40, "[0.20,0.40)"),
    (0.40, 0.60, "[0.40,0.60)"),
    (0.60, 0.80, "[0.60,0.80)"),
    (0.80, 0.950000000001, "[0.80,0.95]"),
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def load_pairs(dataset_root: Path):
    pairs = []
    missing = []
    for collection in ("underwater_dark", "underwater_imagenet", "underwater_scenes"):
        collection_root = dataset_root / "Paired" / collection
        input_dir, target_dir = collection_root / "trainA", collection_root / "trainB"
        if not input_dir.is_dir() or not target_dir.is_dir():
            missing.append(f"{collection}: expected {input_dir} and {target_dir}")
            continue
        targets = {
            p.stem: p for p in target_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        }
        inputs = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for input_path in inputs:
            target_path = targets.get(input_path.stem)
            if target_path is None:
                raise RuntimeError(f"No paired target for {input_path}")
            pairs.append((collection, input_path, target_path))
        unmatched = set(targets) - {p.stem for p in inputs}
        if unmatched:
            raise RuntimeError(f"Unmatched targets in {target_dir}: {sorted(unmatched)[:5]}")
    if missing:
        raise FileNotFoundError("Missing paired EUVP directories:\n" + "\n".join(missing))
    if not pairs:
        raise RuntimeError(
            "No paired training images found in EUVP trainA/trainB. "
            "Populate the configured dataset before running this diagnostic."
        )
    return pairs


def make_split(pairs):
    rng = random.Random(SEED)
    ordered = sorted(pairs, key=lambda p: (p[0], p[1].name.lower()))
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = int(n * TRAIN_FRACTION)
    n_validation = int(n * VALIDATION_FRACTION)
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_validation = max(1, min(n - n_train - 1, n_validation))
    partitions = {
        "controller-train": ordered[:n_train],
        "controller-validation": ordered[n_train:n_train + n_validation],
        "held-out-test": ordered[n_train + n_validation:],
    }
    return partitions


def main() -> None:
    # The configured root is currently a simple relative path. Reading it
    # directly keeps this diagnostic usable without adding a YAML dependency.
    root_value = None
    in_dataset = False
    for line in DATASET_CONFIG.read_text(encoding="utf-8").splitlines():
        if line.strip() == "dataset:":
            in_dataset = True
        elif in_dataset and line and not line[0].isspace():
            break
        elif in_dataset and line.strip().startswith("root:"):
            root_value = line.split(":", 1)[1].strip().strip("\"'")
            break
    if not root_value:
        raise RuntimeError(f"Could not read dataset.root from {DATASET_CONFIG}")
    dataset_root = (DATASET_CONFIG.parent / root_value).resolve()
    pairs = load_pairs(dataset_root)
    partitions = make_split(pairs)
    if not partitions["controller-validation"]:
        raise RuntimeError("At least one pair is required in controller-validation.")

    split_rows = []
    for partition, members in partitions.items():
        for collection, input_path, target_path in members:
            split_rows.append({
                "partition": partition,
                "collection": collection,
                "image_id": f"{collection}/{input_path.stem}",
                "input_path": input_path.relative_to(dataset_root).as_posix(),
                "target_path": target_path.relative_to(dataset_root).as_posix(),
            })
    write_csv(SPLIT_CSV, split_rows,
              ["partition", "collection", "image_id", "input_path", "target_path"])

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Required trained Paper 2 checkpoint not found: {CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state is None:
        raise RuntimeError(f"Checkpoint does not contain model_state_dict: {CHECKPOINT}")
    model = ControllableEnhancer().to(device)
    model.load_state_dict({
        key.removeprefix("enhancer."): value
        for key, value in state.items() if key.startswith("enhancer.")
    }, strict=True)
    model.eval()

    rng = random.Random(SEED)
    validation_pairs = partitions["controller-validation"]
    sample_pairs = [validation_pairs[i % len(validation_pairs)] for i in range(STATE_COUNT)]
    alphas = [rng.uniform(ALPHA_LOW, ALPHA_HIGH) for _ in range(STATE_COUNT)]
    from PIL import Image
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    sample_rows = []
    with torch.inference_mode():
        for sample_index, ((collection, input_path, target_path), alpha) in enumerate(zip(sample_pairs, alphas)):
            x = to_tensor(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)
            target = to_tensor(Image.open(target_path).convert("RGB")).unsqueeze(0).to(device)
            if x.ndim != 3 or x.shape[2] != 3:
             raise RuntimeError(
               f"Expected RGB image for {sample['image_id']}, got shape {x.shape}"
             )
            alpha_minus, alpha_plus = max(0.0, alpha - STEP), min(1.0, alpha + STEP)
            actions = {"minus": alpha_minus, "zero": alpha, "plus": alpha_plus}
            errors = {}
            output_cache = {}
            for name, value in actions.items():
                if value not in output_cache:
                    a = torch.tensor([value], device=device, dtype=x.dtype)
                    output_cache[value] = model(x, a)
                errors[name] = (output_cache[value] - target).abs().mean().item()
            g_minus, g_plus = errors["zero"] - errors["minus"], errors["zero"] - errors["plus"]
            minus_valid = alpha_minus != alpha
            plus_valid = alpha_plus != alpha
            if minus_valid and g_minus > max(g_plus if plus_valid else float("-inf"), 0.0):
                action = "decrease"
            elif plus_valid and g_plus > max(g_minus if minus_valid else float("-inf"), 0.0):
                action = "increase"
            else:
                action = "hold"
            sample_rows.append({
                "image_id": f"{collection}/{input_path.stem}", "alpha": alpha,
                "a_minus": alpha_minus, "a_plus": alpha_plus,
                "q_minus": errors["minus"], "q_0": errors["zero"], "q_plus": errors["plus"],
                "g_minus": g_minus, "g_plus": g_plus,
                "minus_candidate_valid": minus_valid, "plus_candidate_valid": plus_valid,
                "preferred_action": action,
            })
            if (sample_index + 1) % 100 == 0:
                print(f"Measured {sample_index + 1}/{STATE_COUNT} states")
    write_csv(SAMPLES_CSV, sample_rows, list(sample_rows[0]))

    summary = []
    def add(metric, value, group="overall", details=""):
        summary.append({"group": group, "metric": metric, "value": value, "details": details})
    add("seed", SEED)
    add("checkpoint", str(CHECKPOINT.relative_to(PROJECT_ROOT)), details=f"epoch={checkpoint.get('epoch', 'unknown')}")
    add("dataset_root", str(dataset_root))
    add("split_counts", "", details="; ".join(f"{k}={len(v)}" for k, v in partitions.items()))
    add("sample_count", len(sample_rows))
    add("alpha_sampling", f"Uniform({ALPHA_LOW},{ALPHA_HIGH})")
    add("step_h", STEP)
    for gain in ("g_minus", "g_plus"):
        vals = [row[gain] for row in sample_rows]
        add(f"{gain}_mean", sum(vals) / len(vals))
        add(f"{gain}_median", median(vals))
        add(f"{gain}_std", float(torch.tensor(vals, dtype=torch.float64).std(unbiased=False).item()))
        add(f"{gain}_min", min(vals)); add(f"{gain}_max", max(vals))
    counts = Counter(row["preferred_action"] for row in sample_rows)
    for action in ("decrease", "hold", "increase"):
        add(f"preferred_{action}_percent", 100 * counts[action] / len(sample_rows))
    add("decrease_improves_percent", 100 * sum(r["g_minus"] > 0 and r["minus_candidate_valid"] for r in sample_rows) / len(sample_rows))
    add("increase_improves_percent", 100 * sum(r["g_plus"] > 0 and r["plus_candidate_valid"] for r in sample_rows) / len(sample_rows))
    add("neither_improves_percent", 100 * sum(not (r["g_minus"] > 0 and r["minus_candidate_valid"]) and not (r["g_plus"] > 0 and r["plus_candidate_valid"]) for r in sample_rows) / len(sample_rows))
    for threshold in (1e-6, 1e-5, 1e-4):
        count = sum(abs(r["g_minus"]) < threshold or abs(r["g_plus"]) < threshold for r in sample_rows)
        add(f"states_with_gain_abs_below_{threshold:g}_percent", 100 * count / len(sample_rows), details="Either candidate gain is within threshold; diagnostic only")
    margins = []
    for row in sample_rows:
        options = [(row["g_minus"], row["minus_candidate_valid"]), (row["g_plus"], row["plus_candidate_valid"])]
        valid_gains = sorted((gain for gain, valid in options if valid), reverse=True)
        margins.append(valid_gains[0] - max(valid_gains[1], 0.0))
    add("preferred_gain_margin_median", median(margins), details="Winner gain minus best competing gain or zero")
    add("states_preferred_margin_below_1e-5_percent", 100 * sum(m < 1e-5 for m in margins) / len(margins))
    for low, high, label in ALPHA_BINS:
        rows = [r for r in sample_rows if low <= r["alpha"] < high]
        add("sample_count", len(rows), group=label)
        if rows:
            bin_counts = Counter(r["preferred_action"] for r in rows)
            for action in ("decrease", "hold", "increase"):
                add(f"preferred_{action}_percent", 100 * bin_counts[action] / len(rows), group=label)
            for gain in ("g_minus", "g_plus"):
                vals = [r[gain] for r in rows]
                add(f"{gain}_mean", sum(vals) / len(vals), group=label)
                add(f"{gain}_median", median(vals), group=label)
    add("action_rule", "strict comparison of valid gains against zero; otherwise hold")
    add("near_tie_review", "Inspect raw gain values; no epsilon or balancing applied")
    write_csv(SUMMARY_CSV, summary, ["group", "metric", "value", "details"])
    print(f"Split counts: {', '.join(f'{k}={len(v)}' for k, v in partitions.items())}")
    print(f"Sampled {len(sample_rows)} states using {CHECKPOINT.name} on {device}")
    print(f"Wrote {SAMPLES_CSV}, {SUMMARY_CSV}, and {SPLIT_CSV}")


if __name__ == "__main__":
    main()
