"""
End-to-end gloss pipeline inference.

Input : path to .npy keypoint file (T, 1662) or raw video
Output: predicted sentence string

Usage:
  python gloss_pipeline/pipeline.py \
      --input /mnt/ngan/ISL-Sequences/keypoints/i_am_hungry/sample_1.npy

  python gloss_pipeline/pipeline.py \
      --input video.mp4 --extract
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from gloss_pipeline.inference.dual_stream import DualStreamDecoder
from gloss_pipeline.inference.gloss_to_sentence import GlossToSentence


def run(
    npy_path: str,
    window_size:    int   = 30,
    stride:         int   = 10,
    conf_threshold: float = 0.3,
    verbose:        bool  = True,
) -> str:
    sequence = np.load(npy_path).astype(np.float32)
    if verbose:
        print(f'Sequence shape: {sequence.shape}')

    decoder  = DualStreamDecoder(
        window_size=window_size, stride=stride, conf_threshold=conf_threshold)
    g2s      = GlossToSentence()

    results  = decoder.decode(sequence)
    glosses  = [g for g, _ in results]

    if verbose:
        print(f'Raw predictions  : {results}')
        print(f'Gloss sequence   : {glosses}')

    sentence, score = g2s.match(glosses)

    if verbose:
        print(f'Matched sentence : "{sentence}"  (score={score:.3f})')

    return sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',       required=True, help='Path to .npy keypoint file')
    parser.add_argument('--window_size', type=int,   default=30)
    parser.add_argument('--stride',      type=int,   default=10)
    parser.add_argument('--threshold',   type=float, default=0.3)
    args = parser.parse_args()

    result = run(
        npy_path=args.input,
        window_size=args.window_size,
        stride=args.stride,
        conf_threshold=args.threshold,
    )
    print(f'\nResult: "{result}"')
