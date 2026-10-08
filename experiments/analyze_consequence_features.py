"""
Explore target-free image features associated with local alpha actions.

This diagnostic:
- Reuses the existing 600 Step-1 feedback states.
- Reuses the existing Step-1 controller-validation split.
- Uses the existing Paper 2 enhancer checkpoint.
- Does NOT read target images.
- Does NOT train a model.
- Does NOT modify the model, dataset, or checkpoint.
- Measures target-free image statistics for X, Y0, Y-, and Y+.

Outputs:
    results/metrics/consequence_features.csv
    results/metrics/consequence_feature_summary.csv
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.controllable_enhancer import ControllableEnhancer


DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "paper2_epoch_001.pth"
)

SAMPLES_CSV = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "feedback_target_samples.csv"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "feedback_target_split.csv"
)

FEATURES_CSV = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "consequence_features.csv"
)

SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "consequence_feature_summary.csv"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

# Local alpha step used in Step 1
H = 0.10

# Fixed threshold for measuring meaningful pixel change
CHANGE_THRESHOLD = 0.01

# Fixed saturation thresholds
SATURATION_THRESHOLD = 0.01

# Rec. 709 luminance weights
LUMA_WEIGHTS = np.array(
    [0.2126, 0.7152, 0.0722],
    dtype=np.float32,
)


# ============================================================
# DATASET CONFIG
# ============================================================

def read_dataset_root():
    """
    Read dataset.root from configs/dataset.yaml.

    The path is resolved relative to the config directory.
    """

    in_dataset = False

    config_text = DATASET_CONFIG.read_text(
        encoding="utf-8"
    )

    for line in config_text.splitlines():

        stripped = line.strip()

        if stripped == "dataset:":
            in_dataset = True
            continue

        if in_dataset and line and not line[0].isspace():
            break

        if (
            in_dataset
            and stripped.startswith("root:")
        ):
            value = (
                line.split(":", 1)[1]
                .strip()
                .strip("\"'")
            )

            return (
                DATASET_CONFIG.parent / value
            ).resolve()

    raise RuntimeError(
        f"Could not read dataset.root from "
        f"{DATASET_CONFIG}"
    )


# ============================================================
# CSV HELPERS
# ============================================================

def read_csv(path):
    """
    Read a CSV file into a list of dictionaries.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def write_csv(path, rows, fields):
    """
    Write rows to CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_rgb(relative_path, dataset_root):
    """
    Load an RGB image as float32 in [0, 1].
    """

    path = dataset_root / relative_path

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    with Image.open(path) as image:

        return (
            np.asarray(
                image.convert("RGB"),
                dtype=np.float32,
            )
            / 255.0
        )


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def image_features(
    image,
    reference=None,
):
    """
    Calculate simple target-free image statistics.

    Features include:

    RGB:
        mean
        standard deviation
        channel-mean spread

    Clipping:
        fraction near zero
        fraction near one

    Luminance:
        mean
        standard deviation
        P05
        P50
        P95
        P95-P05

    Detail:
        mean gradient magnitude

    Enhancement magnitude relative to reference:
        mean absolute change
        RMS change
        maximum absolute change
        fraction of pixels changed > 0.01
        mean luminance change
        RMS luminance change

    These are statistical proxies, not claims of perceptual quality.
    """

    # --------------------------------------------------------
    # RGB statistics
    # --------------------------------------------------------

    means = image.mean(
        axis=(0, 1)
    )

    stds = image.std(
        axis=(0, 1)
    )

    # --------------------------------------------------------
    # Luminance
    # --------------------------------------------------------

    luminance = image @ LUMA_WEIGHTS

    # --------------------------------------------------------
    # Gradient / high-frequency proxy
    # --------------------------------------------------------

    grad_y, grad_x = np.gradient(
        luminance
    )

    gradient_magnitude = np.sqrt(
        grad_x ** 2
        + grad_y ** 2
    )

    # --------------------------------------------------------
    # Base image features
    # --------------------------------------------------------

    result = {

        # RGB means
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),

        # RGB standard deviations
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),

        # Simple color imbalance proxy
        "channel_mean_spread": float(
            means.max() - means.min()
        ),

        # Clipping / saturation
        "fraction_near_zero": float(
            (
                image
                <= SATURATION_THRESHOLD
            ).mean()
        ),

        "fraction_near_one": float(
            (
                image
                >= 1.0 - SATURATION_THRESHOLD
            ).mean()
        ),

        # Luminance
        "mean_luminance": float(
            luminance.mean()
        ),

        "std_luminance": float(
            luminance.std()
        ),

        "luminance_p05": float(
            np.percentile(
                luminance,
                5,
            )
        ),

        "luminance_p50": float(
            np.percentile(
                luminance,
                50,
            )
        ),

        "luminance_p95": float(
            np.percentile(
                luminance,
                95,
            )
        ),

        # Simple contrast proxy
        "luminance_p95_minus_p05": float(
            np.percentile(
                luminance,
                95,
            )
            -
            np.percentile(
                luminance,
                5,
            )
        ),

        # Detail proxy
        "gradient_magnitude_mean": float(
            gradient_magnitude.mean()
        ),
    }

    # --------------------------------------------------------
    # Change relative to reference image
    # --------------------------------------------------------

    if reference is None:

        change = np.zeros_like(
            image
        )

        luminance_change = np.zeros_like(
            luminance
        )

    else:

        change = (
            image - reference
        )

        reference_luminance = (
            reference @ LUMA_WEIGHTS
        )

        luminance_change = (
            luminance
            - reference_luminance
        )

    abs_change = np.abs(
        change
    )

    result.update({

        "mean_abs_change_from_input": float(
            abs_change.mean()
        ),

        "rms_change_from_input": float(
            np.sqrt(
                np.mean(
                    change ** 2
                )
            )
        ),

        "max_abs_change_from_input": float(
            abs_change.max()
        ),

        "fraction_change_above_0_01": float(
            (
                abs_change
                > CHANGE_THRESHOLD
            ).mean()
        ),

        "mean_luminance_change_from_input": float(
            luminance_change.mean()
        ),

        "rms_luminance_change_from_input": float(
            np.sqrt(
                np.mean(
                    luminance_change ** 2
                )
            )
        ),
    })

    return result


# ============================================================
# STATISTICS
# ============================================================

def median(values):
    """
    Safe median helper.
    """

    return float(
        np.median(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def summarize(features, samples):
    """
    Produce summaries by action.

    Also calculate descriptive Cohen's d for:

        increase vs decrease

    over:

        all alpha
        alpha >= 0.60
        alpha >= 0.80

    Cohen's d is descriptive only.
    """

    if not samples:
        raise RuntimeError(
            "No feature samples available."
        )

    numeric_fields = [
        key
        for key in samples[0]
        if key
        not in {
            "image_id",
            "preferred_action",
        }
    ]

    report = []

    # --------------------------------------------------------
    # Analysis subsets
    # --------------------------------------------------------

    subsets = [

        (
            "all_alpha",
            samples,
        ),

        (
            "alpha_ge_0.60",
            [
                row
                for row in samples
                if float(row["alpha"])
                >= 0.60
            ],
        ),

        (
            "alpha_ge_0.80",
            [
                row
                for row in samples
                if float(row["alpha"])
                >= 0.80
            ],
        ),
    ]

    # --------------------------------------------------------
    # Per-action statistics
    # --------------------------------------------------------

    for subset_name, subset in subsets:

        for action in (
            "increase",
            "decrease",
            "hold",
        ):

            group = [
                row
                for row in subset
                if row["preferred_action"]
                == action
            ]

            for feature in numeric_fields:

                values = [
                    float(
                        row[feature]
                    )
                    for row in group
                ]

                if not values:
                    continue

                report.append({

                    "subset": subset_name,

                    "row_type":
                        "action_summary",

                    "action": action,

                    "feature": feature,

                    "n": len(values),

                    "mean": float(
                        np.mean(values)
                    ),

                    "median": median(
                        values
                    ),

                    "std": float(
                        np.std(
                            values,
                            ddof=0,
                        )
                    ),

                    "increase_minus_decrease_cohens_d":
                        "",
                })

        # ----------------------------------------------------
        # Increase vs decrease effect sizes
        # ----------------------------------------------------

        increase_group = [
            row
            for row in subset
            if row["preferred_action"]
            == "increase"
        ]

        decrease_group = [
            row
            for row in subset
            if row["preferred_action"]
            == "decrease"
        ]

        for feature in numeric_fields:

            x = np.asarray(
                [
                    float(
                        row[feature]
                    )
                    for row in increase_group
                ],
                dtype=np.float64,
            )

            y = np.asarray(
                [
                    float(
                        row[feature]
                    )
                    for row in decrease_group
                ],
                dtype=np.float64,
            )

            if (
                len(x) < 2
                or len(y) < 2
            ):
                continue

            pooled_variance = (
                (
                    (len(x) - 1)
                    * x.var(ddof=1)
                    +
                    (len(y) - 1)
                    * y.var(ddof=1)
                )
                /
                (
                    len(x)
                    + len(y)
                    - 2
                )
            )

            if pooled_variance > 0:

                effect = (
                    x.mean()
                    - y.mean()
                ) / np.sqrt(
                    pooled_variance
                )

            else:

                effect = 0.0

            report.append({

                "subset": subset_name,

                "row_type":
                    "increase_vs_decrease_effect",

                "action":
                    "increase_minus_decrease",

                "feature": feature,

                "n":
                    f"increase={len(x)};"
                    f"decrease={len(y)}",

                "mean": "",

                "median": "",

                "std": "",

                "increase_minus_decrease_cohens_d":
                    float(effect),
            })

    return report


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_root = read_dataset_root()

    # --------------------------------------------------------
    # Existing Step-1 outputs
    # --------------------------------------------------------

    samples = read_csv(
        SAMPLES_CSV
    )

    manifest = read_csv(
        SPLIT_CSV
    )

    # --------------------------------------------------------
    # Verify Step-1 controller-validation split
    # --------------------------------------------------------

    validation_pairs = {

        row["image_id"]: row

        for row in manifest

        if row["partition"]
        == "controller-validation"
    }

    expected_validation_pairs = 1715

    if len(validation_pairs) != expected_validation_pairs:

        raise RuntimeError(
            "Expected the Step-1 "
            "controller-validation manifest "
            f"({expected_validation_pairs} pairs); "
            f"found {len(validation_pairs)}."
        )

    # --------------------------------------------------------
    # Verify exact 600 states
    # --------------------------------------------------------

    expected_states = 600

    if len(samples) != expected_states:

        raise RuntimeError(
            f"Expected {expected_states} "
            f"Step-1 states; "
            f"found {len(samples)}."
        )

    # --------------------------------------------------------
    # Verify every sampled state belongs to validation split
    # --------------------------------------------------------

    for row in samples:

        image_id = row["image_id"]

        if image_id not in validation_pairs:

            raise RuntimeError(
                "Sample is absent from "
                "controller-validation manifest: "
                f"{image_id}"
            )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Load existing checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    state = checkpoint.get(
        "model_state_dict"
    )

    if state is None:

        raise RuntimeError(
            f"Checkpoint has no "
            f"'model_state_dict': "
            f"{CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Extract enhancer weights
    # --------------------------------------------------------

    enhancer_state = {

        key.removeprefix(
            "enhancer."
        ): value

        for key, value in state.items()

        if key.startswith(
            "enhancer."
        )
    }

    if not enhancer_state:

        raise RuntimeError(
            "No enhancer.* parameters "
            "found in checkpoint."
        )

    # --------------------------------------------------------
    # Create enhancer
    # --------------------------------------------------------

    model = (
        ControllableEnhancer()
        .to(device)
    )

    model.load_state_dict(
        enhancer_state,
        strict=True,
    )

    # Evaluation only
    model.eval()

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    feature_rows = []

    with torch.inference_mode():

        for index, sample in enumerate(
            samples
        ):

            image_id = sample[
                "image_id"
            ]

            pair = validation_pairs[
                image_id
            ]

            # ------------------------------------------------
            # Load input image only
            # ------------------------------------------------

            x = load_rgb(
                pair["input_path"],
                dataset_root,
            )

            # ------------------------------------------------
            # Validate RGB image
            # ------------------------------------------------

            if (
                x.ndim != 3
                or x.shape[2] != 3
            ):

                raise RuntimeError(
                    "Expected RGB image for "
                    f"{image_id}, "
                    f"got shape {x.shape}"
                )

            # ------------------------------------------------
            # Convert input to tensor
            # ------------------------------------------------

            x_tensor = (
                torch.from_numpy(
                    x.copy()
                )
                .permute(
                    2,
                    0,
                    1,
                )
                .unsqueeze(0)
                .to(device)
            )

            # ------------------------------------------------
            # Current alpha
            # ------------------------------------------------

            alpha = float(
                sample["alpha"]
            )

            alpha_minus = max(
                0.0,
                alpha - H,
            )

            alpha_plus = min(
                1.0,
                alpha + H,
            )

            # ------------------------------------------------
            # Enhancer helper
            # ------------------------------------------------

            def enhance(alpha_value):

                alpha_tensor = torch.tensor(
                    [alpha_value],
                    dtype=x_tensor.dtype,
                    device=device,
                )

                output = model(
                    x_tensor,
                    alpha_tensor,
                )[0]

                output = (
                    output
                    .permute(
                        1,
                        2,
                        0,
                    )
                    .cpu()
                    .numpy()
                )

                return np.clip(
                    output,
                    0.0,
                    1.0,
                )

            # ------------------------------------------------
            # Generate current and neighboring outputs
            # ------------------------------------------------

            y0 = enhance(
                alpha
            )

            y_minus = enhance(
                alpha_minus
            )

            y_plus = enhance(
                alpha_plus
            )

            # ------------------------------------------------
            # Extract target-free features
            # ------------------------------------------------

            x_features = image_features(
                x
            )

            y0_features = image_features(
                y0,
                reference=x,
            )

            minus_features = image_features(
                y_minus,
                reference=x,
            )

            plus_features = image_features(
                y_plus,
                reference=x,
            )

            # ------------------------------------------------
            # Base row
            # ------------------------------------------------

            result = {

                "image_id": image_id,

                "alpha": alpha,

                "alpha_minus":
                    alpha_minus,

                "alpha_plus":
                    alpha_plus,

                "preferred_action":
                    sample[
                        "preferred_action"
                    ],
            }

            # ------------------------------------------------
            # X features
            # ------------------------------------------------

            for name, value in (
                x_features.items()
            ):

                result[
                    f"x_{name}"
                ] = value

            # ------------------------------------------------
            # Y0 features
            # ------------------------------------------------

            for name, value in (
                y0_features.items()
            ):

                result[
                    f"y0_{name}"
                ] = value

            # ------------------------------------------------
            # Y- features
            # ------------------------------------------------

            for name, value in (
                minus_features.items()
            ):

                result[
                    f"y_minus_{name}"
                ] = value

            # ------------------------------------------------
            # Y+ features
            # ------------------------------------------------

            for name, value in (
                plus_features.items()
            ):

                result[
                    f"y_plus_{name}"
                ] = value

            # ------------------------------------------------
            # Directional feature changes
            #
            # These compare neighboring alpha states against
            # the current state.
            # ------------------------------------------------

            for name in y0_features:

                result[
                    f"delta_plus_{name}"
                ] = (
                    plus_features[name]
                    -
                    y0_features[name]
                )

                result[
                    f"delta_minus_{name}"
                ] = (
                    minus_features[name]
                    -
                    y0_features[name]
                )

            feature_rows.append(
                result
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (index + 1) % 100
                == 0
            ):

                print(
                    f"Processed "
                    f"{index + 1}/"
                    f"{len(samples)} states"
                )

    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    feature_summary = summarize(
        None,
        feature_rows,
    )

    # --------------------------------------------------------
    # Write feature-level CSV
    # --------------------------------------------------------

    write_csv(
        FEATURES_CSV,
        feature_rows,
        list(
            feature_rows[0].keys()
        ),
    )

    # --------------------------------------------------------
    # Write summary CSV
    # --------------------------------------------------------

    write_csv(
        SUMMARY_CSV,
        feature_summary,
        [
            "subset",
            "row_type",
            "action",
            "feature",
            "n",
            "mean",
            "median",
            "std",
            "increase_minus_decrease_cohens_d",
        ],
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("CONSEQUENCE FEATURE DIAGNOSTIC COMPLETE")
    print("=" * 60)

    print(
        f"Device: {device}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"States: {len(feature_rows)}"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"h: {H}"
    )

    print(
        f"Dataset root: {dataset_root}"
    )

    print()
    print(
        f"Wrote: {FEATURES_CSV}"
    )

    print(
        f"Wrote: {SUMMARY_CSV}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()