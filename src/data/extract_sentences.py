"""
Extract MediaPipe Holistic keypoints from ISL sentence videos.

Structure expected:
  /mnt/ngan/isl-sequence-data/
    <sentence_folder>/          ← folder name = sentence label
      000001.mp4
      000002.mp4
      ...                       ← multiple signers, multiple video formats

Output:
  /mnt/ngan/ISL-Sequences/sequences/
    <sentence_folder>/
      000001.npy                ← shape (T, 1662), variable T per video
      000002.npy
      ...

Usage:
  cd isl-sentence
  python src/data/extract_sentences.py
  python src/data/extract_sentences.py --source /mnt/ngan/isl-sequence-data --output /mnt/ngan/ISL-Sequences/sequences
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add isl-sentence root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_SOURCE = Path('/mnt/ngan/isl-sequence-data')
DEFAULT_OUTPUT = ISL_SEQ_DIR / 'sequences'

# ── Supported video extensions ─────────────────────────────────────────────────
VIDEO_EXTENSIONS = {'.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI',
                    '.mkv', '.MKV', '.webm', '.WEBM'}


# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE HELPERS  (same logic as vsl-recognition/src/utils/extraction.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_holistic_model(min_detection_confidence=0.5, min_tracking_confidence=0.5):
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def mediapipe_detection(frame, holistic):
    import mediapipe as mp
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results


def _normalize_keypoints(pose, face, lh, rh):
    """
    Normalize keypoints relative to shoulder midpoint and shoulder width.
    Identical to vsl-recognition/src/utils/extraction.py — must stay in sync
    so that sentence keypoints are in the same distribution as word model training data.

    Left shoulder  = pose landmark 11 → pose[44:48]
    Right shoulder = pose landmark 12 → pose[48:52]
    """
    left_x,  left_y  = pose[44], pose[45]
    right_x, right_y = pose[48], pose[49]

    if left_x == 0.0 and right_x == 0.0:
        return pose, face, lh, rh

    center_x = (left_x + right_x) / 2.0
    center_y = (left_y + right_y) / 2.0
    shoulder_width = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)
    scale = shoulder_width if shoulder_width > 1e-6 else 1.0

    def _norm_xy(arr, stride):
        arr = arr.copy()
        arr[0::stride] = (arr[0::stride] - center_x) / scale
        arr[1::stride] = (arr[1::stride] - center_y) / scale
        return arr

    return (
        _norm_xy(pose, stride=4),
        _norm_xy(face, stride=3),
        _norm_xy(lh,   stride=3),
        _norm_xy(rh,   stride=3),
    )


def extract_keypoints(results) -> np.ndarray:
    """
    Extract and normalize MediaPipe Holistic landmarks into (1662,) vector.
    Normalization is relative to shoulder midpoint + shoulder width,
    matching vsl-recognition preprocessing so word model transfer works correctly.

      pose:       33 × 4  = 132
      face:      468 × 3  = 1404
      left_hand:  21 × 3  = 63
      right_hand: 21 × 3  = 63
    """
    pose = (np.array([[lm.x, lm.y, lm.z, lm.visibility]
                      for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(132))

    face = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(1404))

    lh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(63))

    rh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(63))

    pose, face, lh, rh = _normalize_keypoints(pose, face, lh, rh)

    return np.concatenate([pose, face, lh, rh])   # (1662,)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_sentence_video(video_path: Path, holistic) -> np.ndarray | None:
    """
    Extract keypoints frame-by-frame from a single sentence video.

    Returns:
        np.ndarray of shape (T, 1662) or None if no frames extracted.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        frames.append(keypoints)

    cap.release()
    return np.array(frames, dtype=np.float32) if frames else None


def extract_all_sentences(source_dir: Path, output_dir: Path, skip_existing: bool = True):
    """
    Extract keypoints from all sentence videos.

    Args:
        source_dir:    /mnt/ngan/isl-sequence-data/
        output_dir:    /mnt/ngan/ISL-Sequences/sequences/
        skip_existing: Skip if .npy already exists (resume support)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_folders = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    print('=' * 60)
    print('EXTRACTING SENTENCE KEYPOINTS')
    print(f'  Source : {source_dir}')
    print(f'  Output : {output_dir}')
    print(f'  Sentences found: {len(sentence_folders)}')
    print('=' * 60)

    print('\nInitializing MediaPipe Holistic...')
    holistic = get_holistic_model()

    total_ok      = 0
    total_skipped = 0
    total_failed  = 0

    for idx, folder in enumerate(sentence_folders, 1):
        videos = sorted([f for f in folder.iterdir()
                         if f.suffix in VIDEO_EXTENSIONS])

        if not videos:
            print(f'  [{idx}/{len(sentence_folders)}] "{folder.name}" — no videos, skip')
            continue

        out_folder = output_dir / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        print(f'\n[{idx}/{len(sentence_folders)}] "{folder.name}"  ({len(videos)} videos)')

        for video_path in videos:
            out_path = out_folder / f'{video_path.stem}.npy'

            if skip_existing and out_path.exists():
                print(f'    ↳ {video_path.name}  [skip — already extracted]')
                total_skipped += 1
                continue

            seq = extract_sentence_video(video_path, holistic)

            if seq is None or len(seq) == 0:
                print(f'    ↳ {video_path.name}  [FAILED — no frames]')
                total_failed += 1
                continue

            np.save(out_path, seq)
            print(f'    ↳ {video_path.name}  →  shape {seq.shape}  ✓')
            total_ok += 1

    holistic.close()

    print(f'\n{"=" * 60}')
    print(f'[OK] EXTRACTION COMPLETE')
    print(f'  Extracted : {total_ok}')
    print(f'  Skipped   : {total_skipped}  (already done)')
    print(f'  Failed    : {total_failed}')
    print(f'  Saved to  : {output_dir}')
    print(f'{"=" * 60}')


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract keypoints from ISL sentence videos.')
    parser.add_argument('--source', type=str, default=str(DEFAULT_SOURCE),
                        help=f'Sentence video directory (default: {DEFAULT_SOURCE})')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help=f'Output .npy directory (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-extract even if .npy already exists')
    args = parser.parse_args()

    extract_all_sentences(
        source_dir    = Path(args.source),
        output_dir    = Path(args.output),
        skip_existing = not args.no_skip,
    )
