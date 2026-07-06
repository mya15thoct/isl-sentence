"""Extract MediaPipe Holistic keypoints for How2Sign clips.

Uses the SAME extraction + normalisation code as the iSign pipeline
(`src/keypoints/holistic.py`), so the input protocol is identical across
datasets (fairness item 6 in README.md). CPU-only; resumable (skips clips whose
.npy already exists); optionally deletes each clip after successful extraction
to stay inside the disk budget.

    python cross_dataset/extract_how2sign.py \
        --clips /mnt/recover/ngan/How2Sign/clips_train \
        --out   /mnt/recover/ngan/How2Sign/keypoints/train \
        --workers 6 --delete-after
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

_MODEL = None  # one Holistic model per worker process


def _init_worker() -> None:
    global _MODEL
    from src.keypoints.holistic import create_holistic_model
    _MODEL = create_holistic_model()


def _extract_one(job: tuple[str, str, bool]) -> tuple[str, str]:
    video_path, out_path, delete_after = job
    from src.keypoints.holistic import extract_video_keypoints
    try:
        arr = extract_video_keypoints(Path(video_path), _MODEL)
        if arr.ndim != 2 or arr.shape[1] != 1662 or arr.shape[0] == 0:
            return video_path, f"BAD_SHAPE {getattr(arr, 'shape', None)}"
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".npy.tmp")
        np.save(tmp, arr.astype(np.float32))
        tmp.replace(out)
        if delete_after:
            Path(video_path).unlink(missing_ok=True)
        return video_path, "OK"
    except Exception as exc:  # keep the batch going; log the failure
        return video_path, f"FAIL {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True, help="folder of .mp4 clips")
    ap.add_argument("--out", type=Path, required=True, help="output folder for .npy")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--delete-after", action="store_true", help="delete each clip once its .npy is saved")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    clips = sorted(args.clips.glob("*.mp4"))
    if args.limit:
        clips = clips[: args.limit]
    jobs = []
    for clip in clips:
        out = args.out / (clip.stem + ".npy")
        if out.exists():
            continue  # resume
        jobs.append((str(clip), str(out), args.delete_after))
    print(f"clips: {len(clips)}  to-extract: {len(jobs)}  workers: {args.workers}", flush=True)

    failures: list[tuple[str, str]] = []
    start = time.time()
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(_extract_one, jobs, chunksize=4), 1):
            if status != "OK":
                failures.append((path, status))
            if i % 200 == 0:
                rate = i / max(time.time() - start, 1e-6)
                eta_h = (len(jobs) - i) / max(rate, 1e-6) / 3600
                print(f"{i}/{len(jobs)}  {rate:.1f} clips/s  ETA {eta_h:.1f}h  fails={len(failures)}", flush=True)

    print(f"done: {len(jobs) - len(failures)} ok, {len(failures)} failed")
    if failures:
        log = args.out / "_failures.log"
        log.write_text("\n".join(f"{p}\t{s}" for p, s in failures), encoding="utf-8")
        print(f"failure log: {log}")


if __name__ == "__main__":
    main()
