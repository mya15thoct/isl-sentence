"""
Extract keypoints from ISL-Frames-Data static images using MediaPipe Holistic.

Each image → 1662-dim keypoint vector → saved as .npy

Usage:
  python gloss_pipeline/data/extract_frames.py \
      --frames_dir /mnt/ngan/ISL-Frames-Data \
      --output_dir /mnt/ngan/ISL-Sequences/frame_keypoints
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SERVER_BASE, ISL_SEQ_DIR

FRAMES_DIR  = SERVER_BASE / 'ISL-Frames-Data'
OUTPUT_DIR  = ISL_SEQ_DIR / 'frame_keypoints'

mp_holistic = mp.solutions.holistic


def extract_keypoints(results) -> np.ndarray:
    pose = np.array([[r.x, r.y, r.z, r.visibility]
                     for r in results.pose_landmarks.landmark],
                    dtype=np.float32).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[r.x, r.y, r.z]
                     for r in results.face_landmarks.landmark],
                    dtype=np.float32).flatten() if results.face_landmarks else np.zeros(1404)
    lh   = np.array([[r.x, r.y, r.z]
                     for r in results.left_hand_landmarks.landmark],
                    dtype=np.float32).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh   = np.array([[r.x, r.y, r.z]
                     for r in results.right_hand_landmarks.landmark],
                    dtype=np.float32).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])


def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    """Normalize relative to shoulder midpoint and width — matches word model."""
    kp = kp.copy()
    pose = kp[:132].reshape(33, 4)

    ls_x, ls_y = pose[11, 0], pose[11, 1]
    rs_x, rs_y = pose[12, 0], pose[12, 1]
    cx, cy     = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    scale      = max(abs(rs_x - ls_x), 1e-6)

    pose[:, 0] = (pose[:, 0] - cx) / scale
    pose[:, 1] = (pose[:, 1] - cy) / scale
    kp[:132]   = pose.flatten()

    face = kp[132:1536].reshape(-1, 3)
    face[:, 0] = (face[:, 0] - cx) / scale
    face[:, 1] = (face[:, 1] - cy) / scale
    kp[132:1536] = face.flatten()

    for start in [1536, 1536 + 63]:
        hand = kp[start:start + 63].reshape(-1, 3)
        hand[:, 0] = (hand[:, 0] - cx) / scale
        hand[:, 1] = (hand[:, 1] - cy) / scale
        kp[start:start + 63] = hand.flatten()

    return kp


def extract(frames_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    class_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    print(f'Found {len(class_dirs)} classes in {frames_dir}')

    with mp_holistic.Holistic(static_image_mode=True,
                              min_detection_confidence=0.5) as holistic:
        for class_dir in class_dirs:
            out_class = output_dir / class_dir.name
            out_class.mkdir(exist_ok=True)

            images = sorted(list(class_dir.glob('*.jpg')) +
                            list(class_dir.glob('*.png')) +
                            list(class_dir.glob('*.jpeg')))

            saved = 0
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                kp = extract_keypoints(results)
                kp = normalize_keypoints(kp)
                np.save(str(out_class / f'{img_path.stem}.npy'), kp)
                saved += 1

            print(f'  {class_dir.name:30s} → {saved}/{len(images)} images')

    print(f'\nDone. Saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', default=str(FRAMES_DIR))
    parser.add_argument('--output_dir', default=str(OUTPUT_DIR))
    args = parser.parse_args()
    extract(Path(args.frames_dir), Path(args.output_dir))
