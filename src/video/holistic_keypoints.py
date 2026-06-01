"""
MediaPipe Holistic keypoint extraction compatible with vsl-recognition.

Output per frame is 1662 dimensions:
  pose: 33 * 4 = 132
  face: 468 * 3 = 1404
  left hand: 21 * 3 = 63
  right hand: 21 * 3 = 63

Normalization matches the recognition project:
  - center x/y at shoulder midpoint
  - scale x/y by shoulder width
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


KEYPOINT_DIM = 1662
POSE_DIM = 132
FACE_DIM = 1404
HAND_DIM = 63


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def normalize_keypoints(
    pose: np.ndarray,
    face: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_shoulder_x, left_shoulder_y = pose[44], pose[45]
    right_shoulder_x, right_shoulder_y = pose[48], pose[49]

    if left_shoulder_x == 0.0 and right_shoulder_x == 0.0:
        return pose, face, left_hand, right_hand

    center_x = (left_shoulder_x + right_shoulder_x) / 2.0
    center_y = (left_shoulder_y + right_shoulder_y) / 2.0
    shoulder_width = np.sqrt(
        (right_shoulder_x - left_shoulder_x) ** 2
        + (right_shoulder_y - left_shoulder_y) ** 2
    )
    scale = shoulder_width if shoulder_width > 1e-6 else 1.0

    def normalize_xy(values: np.ndarray, stride: int) -> np.ndarray:
        values = values.copy()
        values[0::stride] = (values[0::stride] - center_x) / scale
        values[1::stride] = (values[1::stride] - center_y) / scale
        return values

    return (
        normalize_xy(pose, stride=4),
        normalize_xy(face, stride=3),
        normalize_xy(left_hand, stride=3),
        normalize_xy(right_hand, stride=3),
    )


def extract_keypoints(results) -> np.ndarray:
    pose = (
        np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4, dtype=np.float32)
    )

    face = (
        np.array(
            [[res.x, res.y, res.z] for res in results.face_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3, dtype=np.float32)
    )

    left_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )

    right_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )

    pose, face, left_hand, right_hand = normalize_keypoints(
        pose, face, left_hand, right_hand
    )
    return np.concatenate([pose, face, left_hand, right_hand]).astype(np.float32)


def create_holistic_model(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_video_keypoints(video_path: Path, holistic_model) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, results = mediapipe_detection(frame, holistic_model)
            frames.append(extract_keypoints(results))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames extracted: {video_path}")

    return np.stack(frames).astype(np.float32)
