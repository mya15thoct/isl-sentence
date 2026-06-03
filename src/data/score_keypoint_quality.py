"""
Score extracted iSign keypoints and split them into pass/suspect/reject sets.

This is the quality-control step after build_keypoint_manifest.py. The earlier
manifest check only verifies that .npy files exist and have shape Nx1662; this
script checks whether the extracted sequence looks useful for sign training.

Example:
  python src/data/score_keypoint_quality.py \
      --manifest /mnt/ngan/ISL-Sequences/manifests/isign_existing_clean.csv \
      --output-scored /mnt/ngan/ISL-Sequences/manifests/isign_quality_scored.csv \
      --output-pass /mnt/ngan/ISL-Sequences/manifests/isign_train_qc.csv \
      --output-reject /mnt/ngan/ISL-Sequences/manifests/isign_reject_qc.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np


KEYPOINT_DIM = 1662
POSE_DIM = 132
FACE_DIM = 1404
HAND_DIM = 63

POSE_START = 0
FACE_START = POSE_START + POSE_DIM
LEFT_HAND_START = FACE_START + FACE_DIM
RIGHT_HAND_START = LEFT_HAND_START + HAND_DIM

QUALITY_FIELDS = [
    "quality_status",
    "quality_reasons",
    "quality_score",
    "num_frames_scored",
    "finite_ratio",
    "zero_frame_ratio",
    "pose_frame_ratio",
    "face_frame_ratio",
    "left_hand_frame_ratio",
    "right_hand_frame_ratio",
    "any_hand_frame_ratio",
    "both_hand_frame_ratio",
    "hand_motion_mean",
    "hand_motion_median",
    "hand_motion_p95",
    "pose_motion_mean",
    "text_noise_flag",
]

TEXT_NOISE_PATTERNS = [
    re.compile(r"^page(\s+number)?\.?\s*\d+\.?$", re.IGNORECASE),
    re.compile(r"^unit\s+\d+\.?$", re.IGNORECASE),
    re.compile(r"^chapter\s+\d+\.?$", re.IGNORECASE),
    re.compile(r"^\d+\.?$"),
    re.compile(r"^[a-z]$", re.IGNORECASE),
    re.compile(r"^reading is fun\.?$", re.IGNORECASE),
    re.compile(r"^new words\.?$", re.IGNORECASE),
    re.compile(r"^say aloud\.?$", re.IGNORECASE),
    re.compile(r"^let'?s (read|write|talk|share|do|practice|make)!?\.?$", re.IGNORECASE),
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_fields(fieldnames: list[str]) -> list[str]:
    fields = list(fieldnames)
    for field in QUALITY_FIELDS:
        if field not in fields:
            fields.append(field)
    return fields


def keypoint_path_for(row: dict[str, str], keypoint_root: Path | None) -> Path | None:
    raw_path = row.get("keypoint_path", "").strip()
    if raw_path:
        return Path(raw_path)

    if keypoint_root is None:
        return None

    uid = row.get("uid", "").strip()
    category = row.get("category", "").strip() or "unknown"
    if not uid:
        return None
    return keypoint_root / category / f"{uid}.npy"


def split_keypoints(
    keypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pose = keypoints[:, POSE_START:FACE_START].reshape(-1, 33, 4)
    face = keypoints[:, FACE_START:LEFT_HAND_START].reshape(-1, 468, 3)
    left_hand = keypoints[:, LEFT_HAND_START:RIGHT_HAND_START].reshape(-1, 21, 3)
    right_hand = keypoints[:, RIGHT_HAND_START:].reshape(-1, 21, 3)
    return pose, face, left_hand, right_hand


def valid_xyz(points: np.ndarray, eps: float) -> np.ndarray:
    return np.any(np.abs(points[..., :3]) > eps, axis=-1)


def valid_pose(points: np.ndarray, eps: float, min_visibility: float) -> np.ndarray:
    return (points[..., 3] > min_visibility) | valid_xyz(points[..., :3], eps)


def frame_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask))


def landmark_center_xy(
    points: np.ndarray,
    valid_points: np.ndarray,
    min_landmarks: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = valid_points.sum(axis=1)
    ok = counts >= min_landmarks
    centers = np.full((points.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(ok):
        xy_sum = (points[:, :, :2] * valid_points[:, :, None]).sum(axis=1)
        centers[ok] = xy_sum[ok] / counts[ok, None]
    return centers, ok


def combined_hand_center(
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    min_hand_landmarks: int,
) -> tuple[np.ndarray, np.ndarray]:
    left_center, left_ok = landmark_center_xy(left_hand, left_valid, min_hand_landmarks)
    right_center, right_ok = landmark_center_xy(right_hand, right_valid, min_hand_landmarks)

    centers = np.full_like(left_center, np.nan)
    counts = np.zeros((left_center.shape[0], 1), dtype=np.float32)
    sums = np.zeros_like(left_center)

    if np.any(left_ok):
        sums[left_ok] += left_center[left_ok]
        counts[left_ok] += 1.0
    if np.any(right_ok):
        sums[right_ok] += right_center[right_ok]
        counts[right_ok] += 1.0

    ok = counts[:, 0] > 0
    centers[ok] = sums[ok] / counts[ok]
    return centers, ok


def motion_stats(centers: np.ndarray, ok: np.ndarray) -> tuple[float, float, float]:
    if centers.shape[0] < 2:
        return 0.0, 0.0, 0.0

    pair_ok = ok[1:] & ok[:-1]
    if not np.any(pair_ok):
        return 0.0, 0.0, 0.0

    deltas = centers[1:] - centers[:-1]
    motion = np.linalg.norm(deltas[pair_ok], axis=1)
    if motion.size == 0:
        return 0.0, 0.0, 0.0

    return (
        float(np.mean(motion)),
        float(np.median(motion)),
        float(np.percentile(motion, 95)),
    )


def looks_like_text_noise(row: dict[str, str]) -> bool:
    category = row.get("category", "").strip()
    text = row.get("text", "").strip()
    if not text:
        return True
    if category and category != "general_sentence":
        return False
    return any(pattern.match(text) for pattern in TEXT_NOISE_PATTERNS)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def score_keypoints(
    keypoints: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    finite = np.isfinite(keypoints)
    finite_ratio = float(np.mean(finite)) if keypoints.size else 0.0
    if not np.all(finite):
        keypoints = np.nan_to_num(keypoints, copy=True)

    pose, face, left_hand, right_hand = split_keypoints(keypoints)

    zero_frame_mask = np.all(np.abs(keypoints) <= args.zero_eps, axis=1)
    pose_valid_points = valid_pose(pose, args.zero_eps, args.min_pose_visibility)
    face_valid_points = valid_xyz(face, args.zero_eps)
    left_valid_points = valid_xyz(left_hand, args.zero_eps)
    right_valid_points = valid_xyz(right_hand, args.zero_eps)

    pose_frame_mask = pose_valid_points.sum(axis=1) >= args.min_pose_landmarks
    face_frame_mask = face_valid_points.sum(axis=1) >= args.min_face_landmarks
    left_frame_mask = left_valid_points.sum(axis=1) >= args.min_hand_landmarks
    right_frame_mask = right_valid_points.sum(axis=1) >= args.min_hand_landmarks
    any_hand_frame_mask = left_frame_mask | right_frame_mask
    both_hand_frame_mask = left_frame_mask & right_frame_mask

    hand_centers, hand_ok = combined_hand_center(
        left_hand,
        right_hand,
        left_valid_points,
        right_valid_points,
        args.min_hand_landmarks,
    )
    hand_motion_mean, hand_motion_median, hand_motion_p95 = motion_stats(
        hand_centers,
        hand_ok,
    )

    pose_centers, pose_ok = landmark_center_xy(
        pose[:, :, :3],
        pose_valid_points,
        args.min_pose_landmarks,
    )
    pose_motion_mean, _, _ = motion_stats(pose_centers, pose_ok)

    return {
        "num_frames_scored": int(keypoints.shape[0]),
        "finite_ratio": finite_ratio,
        "zero_frame_ratio": frame_ratio(zero_frame_mask),
        "pose_frame_ratio": frame_ratio(pose_frame_mask),
        "face_frame_ratio": frame_ratio(face_frame_mask),
        "left_hand_frame_ratio": frame_ratio(left_frame_mask),
        "right_hand_frame_ratio": frame_ratio(right_frame_mask),
        "any_hand_frame_ratio": frame_ratio(any_hand_frame_mask),
        "both_hand_frame_ratio": frame_ratio(both_hand_frame_mask),
        "hand_motion_mean": hand_motion_mean,
        "hand_motion_median": hand_motion_median,
        "hand_motion_p95": hand_motion_p95,
        "pose_motion_mean": pose_motion_mean,
    }


def assign_quality(
    row: dict[str, str],
    metrics: dict[str, float | int],
    args: argparse.Namespace,
) -> tuple[str, list[str], float]:
    reasons: list[str] = []
    status = "pass"

    num_frames = int(metrics["num_frames_scored"])
    finite_ratio = float(metrics["finite_ratio"])
    zero_ratio = float(metrics["zero_frame_ratio"])
    pose_ratio = float(metrics["pose_frame_ratio"])
    hand_ratio = float(metrics["any_hand_frame_ratio"])
    hand_motion = float(metrics["hand_motion_mean"])

    if num_frames < args.min_frames:
        reasons.append("too_short")
        status = "reject"
    if finite_ratio < args.min_finite_ratio:
        reasons.append("nonfinite_values")
        status = "reject"
    if zero_ratio > args.max_zero_frame_ratio:
        reasons.append("mostly_empty")
        status = "reject"
    if pose_ratio < args.min_pose_frame_ratio:
        reasons.append("low_pose_detection")
        status = "reject"

    if hand_ratio < args.min_hand_frame_ratio:
        reasons.append("low_hand_detection")
        if status != "reject":
            status = args.low_hand_status

    if hand_motion < args.min_hand_motion and hand_ratio >= args.min_hand_frame_ratio:
        reasons.append("low_hand_motion")
        if status != "reject":
            status = "suspect"

    text_noise = looks_like_text_noise(row)
    if args.flag_text_noise and text_noise:
        reasons.append("text_noise")
        if status == "pass":
            status = "suspect"

    score = 100.0
    score -= max(0.0, args.min_pose_frame_ratio - pose_ratio) * 160.0
    score -= max(0.0, args.min_hand_frame_ratio - hand_ratio) * 160.0
    score -= max(0.0, zero_ratio - 0.05) * 100.0
    score -= 30.0 if hand_motion < args.min_hand_motion else 0.0
    score -= 20.0 if text_noise and args.flag_text_noise else 0.0
    score = max(0.0, min(100.0, score))

    return status, reasons, score


def empty_metrics() -> dict[str, float | int]:
    return {
        "num_frames_scored": 0,
        "finite_ratio": 0.0,
        "zero_frame_ratio": 1.0,
        "pose_frame_ratio": 0.0,
        "face_frame_ratio": 0.0,
        "left_hand_frame_ratio": 0.0,
        "right_hand_frame_ratio": 0.0,
        "any_hand_frame_ratio": 0.0,
        "both_hand_frame_ratio": 0.0,
        "hand_motion_mean": 0.0,
        "hand_motion_median": 0.0,
        "hand_motion_p95": 0.0,
        "pose_motion_mean": 0.0,
    }


def score_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    output = dict(row)
    path = keypoint_path_for(row, args.keypoint_root)
    text_noise = looks_like_text_noise(row)

    if path is None:
        metrics = empty_metrics()
        status, reasons, score = "reject", ["missing_keypoint_path"], 0.0
    elif not path.is_file():
        metrics = empty_metrics()
        status, reasons, score = "reject", ["missing_keypoint_file"], 0.0
    else:
        try:
            keypoints = np.load(path, mmap_mode="r")
            shape = tuple(int(x) for x in keypoints.shape)
            if len(shape) != 2 or shape[1] != KEYPOINT_DIM:
                metrics = empty_metrics()
                status = "reject"
                reasons = [f"bad_shape:{'x'.join(str(x) for x in shape)}"]
                score = 0.0
            else:
                metrics = score_keypoints(np.asarray(keypoints), args)
                status, reasons, score = assign_quality(row, metrics, args)
        except Exception as exc:
            metrics = empty_metrics()
            status, reasons, score = "reject", [f"read_error:{exc}"], 0.0

    output["quality_status"] = status
    output["quality_reasons"] = ";".join(reasons)
    output["quality_score"] = fmt(score)
    output["num_frames_scored"] = str(int(metrics["num_frames_scored"]))
    output["finite_ratio"] = fmt(float(metrics["finite_ratio"]))
    output["zero_frame_ratio"] = fmt(float(metrics["zero_frame_ratio"]))
    output["pose_frame_ratio"] = fmt(float(metrics["pose_frame_ratio"]))
    output["face_frame_ratio"] = fmt(float(metrics["face_frame_ratio"]))
    output["left_hand_frame_ratio"] = fmt(float(metrics["left_hand_frame_ratio"]))
    output["right_hand_frame_ratio"] = fmt(float(metrics["right_hand_frame_ratio"]))
    output["any_hand_frame_ratio"] = fmt(float(metrics["any_hand_frame_ratio"]))
    output["both_hand_frame_ratio"] = fmt(float(metrics["both_hand_frame_ratio"]))
    output["hand_motion_mean"] = fmt(float(metrics["hand_motion_mean"]))
    output["hand_motion_median"] = fmt(float(metrics["hand_motion_median"]))
    output["hand_motion_p95"] = fmt(float(metrics["hand_motion_p95"]))
    output["pose_motion_mean"] = fmt(float(metrics["pose_motion_mean"]))
    output["text_noise_flag"] = "1" if text_noise else "0"
    return output


def write_optional_outputs(
    args: argparse.Namespace,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    pass_rows = [row for row in rows if row.get("quality_status") == "pass"]
    suspect_rows = [row for row in rows if row.get("quality_status") == "suspect"]
    reject_rows = [row for row in rows if row.get("quality_status") == "reject"]

    if args.output_scored:
        write_csv(args.output_scored, fieldnames, rows)
    if args.output_pass:
        write_csv(args.output_pass, fieldnames, pass_rows)
    if args.output_suspect:
        write_csv(args.output_suspect, fieldnames, suspect_rows)
    if args.output_reject:
        write_csv(args.output_reject, fieldnames, reject_rows)


def print_summary(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    status_counts = Counter(row.get("quality_status", "") for row in rows)
    reason_counts: Counter[str] = Counter()
    category_counts = Counter(row.get("category", "") for row in rows)
    for row in rows:
        for reason in row.get("quality_reasons", "").split(";"):
            if reason:
                reason_counts[reason.split(":", 1)[0]] += 1

    print(f"Rows scored  : {len(rows)}")
    print("By quality:")
    for status, count in status_counts.most_common():
        print(f"  {status or '<blank>':12s}: {count}")
    print("Top reasons:")
    for reason, count in reason_counts.most_common(12):
        print(f"  {reason:24s}: {count}")
    print("By category:")
    for category, count in category_counts.most_common():
        print(f"  {category or '<blank>':18s}: {count}")

    if args.output_scored:
        print(f"Scored       : {args.output_scored}")
    if args.output_pass:
        print(f"Pass         : {args.output_pass}")
    if args.output_suspect:
        print(f"Suspect      : {args.output_suspect}")
    if args.output_reject:
        print(f"Reject       : {args.output_reject}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--keypoint-root",
        type=Path,
        default=None,
        help="Fallback root used when rows do not contain keypoint_path.",
    )
    parser.add_argument("--output-scored", type=Path, default=None)
    parser.add_argument("--output-pass", type=Path, default=None)
    parser.add_argument("--output-suspect", type=Path, default=None)
    parser.add_argument("--output-reject", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1000)

    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--zero-eps", type=float, default=1e-7)
    parser.add_argument("--min-finite-ratio", type=float, default=1.0)
    parser.add_argument("--max-zero-frame-ratio", type=float, default=0.80)
    parser.add_argument("--min-pose-frame-ratio", type=float, default=0.15)
    parser.add_argument("--min-hand-frame-ratio", type=float, default=0.10)
    parser.add_argument("--min-hand-motion", type=float, default=0.001)
    parser.add_argument("--min-pose-visibility", type=float, default=1e-4)
    parser.add_argument("--min-pose-landmarks", type=int, default=6)
    parser.add_argument("--min-face-landmarks", type=int, default=20)
    parser.add_argument("--min-hand-landmarks", type=int, default=6)
    parser.add_argument(
        "--low-hand-status",
        choices=["suspect", "reject"],
        default="suspect",
        help="How to classify rows with low hand detection.",
    )
    parser.add_argument(
        "--flag-text-noise",
        action="store_true",
        help="Mark page/unit/instruction-like text as suspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_csv(args.manifest)

    if args.start:
        rows = rows[args.start:]
    if args.limit is not None:
        rows = rows[: args.limit]

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        output_rows.append(score_row(row, args))
        if args.log_every > 0 and index % args.log_every == 0:
            print(f"Scored {index}/{len(rows)} rows...")

    fields = ensure_fields(fieldnames)
    write_optional_outputs(args, fields, output_rows)
    print_summary(output_rows, args)


if __name__ == "__main__":
    main()
