"""
Render extracted MediaPipe Holistic keypoints to a simple skeleton video.

The extracted keypoints are shoulder-centered and shoulder-scaled, so this
visualization draws them on a blank canvas instead of overlaying on raw video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


POSE_DIM = 132
FACE_DIM = 1404
HAND_DIM = 63
KEYPOINT_DIM = POSE_DIM + FACE_DIM + HAND_DIM + HAND_DIM

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
]

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--draw-face", action="store_true")
    parser.add_argument("--point-radius", type=int, default=3)
    return parser.parse_args()


def split_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if frame.shape[0] != KEYPOINT_DIM:
        raise ValueError(f"Expected keypoint dim {KEYPOINT_DIM}, got {frame.shape[0]}")

    pose = frame[:POSE_DIM].reshape(33, 4)
    face_start = POSE_DIM
    face = frame[face_start : face_start + FACE_DIM].reshape(468, 3)
    left_start = face_start + FACE_DIM
    left_hand = frame[left_start : left_start + HAND_DIM].reshape(21, 3)
    right_start = left_start + HAND_DIM
    right_hand = frame[right_start : right_start + HAND_DIM].reshape(21, 3)
    return pose, face, left_hand, right_hand


def valid_xyz(points: np.ndarray) -> np.ndarray:
    return np.any(np.abs(points[:, :3]) > 1e-7, axis=1)


def valid_pose(points: np.ndarray) -> np.ndarray:
    return (points[:, 3] > 1e-4) | valid_xyz(points)


def frame_indices(num_frames: int, start_frame: int, max_frames: int, stride: int) -> np.ndarray:
    start = min(max(start_frame, 0), num_frames)
    indices = np.arange(start, num_frames, max(stride, 1), dtype=np.int64)
    if max_frames > 0:
        indices = indices[:max_frames]
    return indices


def collect_xy(keypoints: np.ndarray, indices: np.ndarray, draw_face: bool) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for idx in indices:
        pose, face, left_hand, right_hand = split_frame(keypoints[idx])
        for points, mask in (
            (pose[:, :2], valid_pose(pose)),
            (left_hand[:, :2], valid_xyz(left_hand)),
            (right_hand[:, :2], valid_xyz(right_hand)),
        ):
            if np.any(mask):
                chunks.append(points[mask])
        if draw_face:
            mask = valid_xyz(face)
            if np.any(mask):
                chunks.append(face[:, :2][mask])

    if not chunks:
        return np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def build_transform(xy: np.ndarray, width: int, height: int, margin: int = 60):
    low = np.percentile(xy, 1, axis=0)
    high = np.percentile(xy, 99, axis=0)
    center = (low + high) / 2.0
    span = np.maximum(high - low, 1e-3)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def transform(points: np.ndarray) -> np.ndarray:
        out = points[:, :2].astype(np.float32).copy()
        out = (out - center) * scale
        out[:, 0] += width / 2
        out[:, 1] += height / 2
        return np.round(out).astype(np.int32)

    return transform


def draw_points(
    canvas: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    transform,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    if not np.any(mask):
        return
    pts = transform(points[:, :2])
    for point, ok in zip(pts, mask):
        if ok:
            cv2.circle(canvas, tuple(point), radius, color, -1, lineType=cv2.LINE_AA)


def draw_connections(
    canvas: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    connections: list[tuple[int, int]],
    transform,
    color: tuple[int, int, int],
) -> None:
    if not np.any(mask):
        return
    pts = transform(points[:, :2])
    for a, b in connections:
        if mask[a] and mask[b]:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), color, 2, lineType=cv2.LINE_AA)


def render_video(args: argparse.Namespace) -> None:
    keypoints = np.load(args.keypoint)
    if keypoints.ndim != 2 or keypoints.shape[1] != KEYPOINT_DIM:
        raise ValueError(f"Invalid keypoint shape: {keypoints.shape}")

    indices = frame_indices(
        keypoints.shape[0],
        args.start_frame,
        args.max_frames,
        args.stride,
    )
    if len(indices) == 0:
        raise ValueError("No frames selected.")

    xy = collect_xy(keypoints, indices, args.draw_face)
    transform = build_transform(xy, args.width, args.height)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise ValueError(f"Could not open output video writer: {args.output}")

    try:
        for frame_no, idx in enumerate(indices, start=1):
            canvas = np.full((args.height, args.width, 3), 18, dtype=np.uint8)
            pose, face, left_hand, right_hand = split_frame(keypoints[idx])
            pose_mask = valid_pose(pose)
            left_mask = valid_xyz(left_hand)
            right_mask = valid_xyz(right_hand)

            draw_connections(canvas, pose, pose_mask, POSE_CONNECTIONS, transform, (80, 210, 255))
            draw_connections(canvas, left_hand, left_mask, HAND_CONNECTIONS, transform, (80, 255, 130))
            draw_connections(canvas, right_hand, right_mask, HAND_CONNECTIONS, transform, (255, 170, 80))

            draw_points(canvas, pose, pose_mask, transform, (120, 230, 255), args.point_radius)
            draw_points(canvas, left_hand, left_mask, transform, (120, 255, 160), args.point_radius)
            draw_points(canvas, right_hand, right_mask, transform, (255, 190, 120), args.point_radius)

            if args.draw_face:
                face_mask = valid_xyz(face)
                draw_points(canvas, face, face_mask, transform, (190, 190, 190), 1)

            cv2.putText(
                canvas,
                f"{args.keypoint.name} frame {idx} ({frame_no}/{len(indices)})",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (235, 235, 235),
                2,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()

    print(f"Saved: {args.output}")
    print(f"Frames rendered: {len(indices)} / source frames: {keypoints.shape[0]}")


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()
