# Cross-dataset evaluation — running HARP on public sign-retrieval benchmarks

Goal: train/evaluate our model on a benchmark other papers use, with THEIR
protocol, so the paper can place our numbers next to published ones and show the
hand-aware design generalises beyond ISL/iSign.

## Dataset research (which benchmark, and why)

| Dataset | Language | Access | Verdict |
|---|---|---|---|
| **How2Sign** | ASL, English captions | **Public, CC BY-NC 4.0**, Google Drive links on how2sign.github.io | ✅ **CHOSEN** |
| PHOENIX-2014T | DGS, **German** captions | Public (RWTH) | ❌ German text breaks our English bge encoder; swapping to a multilingual encoder changes the model → unfair comparison |
| CSL-Daily | CSL, **Chinese** captions | **Requires signed agreement** | ❌ Restricted access + Chinese text |
| BOBSL | BSL | **Institutional licence required** | ❌ Restricted |
| Spreadthesign (SignCLIP) | multi | Not redistributable | ❌ Not obtainable |

**How2Sign is the only benchmark that is simultaneously public, English-captioned,
sentence-level, and carries published retrieval numbers.** Official splits:
**train 31,164 / val 1,740 / test 2,356** sentence-video pairs (the split is
inherited from How2, grouped at video level — no re-splitting needed or allowed).

What we download (fits the ~76 GB free disk):
- Green Screen RGB **CLIPS** (frontal): train 31 GB + val 1.7 GB + test 2.2 GB.
  (NOT the 290 GB full videos; clips are already cut per sentence.)
- English translations (re-aligned): 3 small CSVs.
- The provided B-F-H 2D keypoints are **OpenPose format (137 pts)** — incompatible
  with our MediaPipe-1662 pipeline, so we re-extract MediaPipe Holistic from the
  clips with the SAME code used for iSign (`src/keypoints/holistic.py`), keeping
  the input protocol identical across datasets.

## Published numbers to compare against (test set = full 2,356 pool)

From the SEDS paper's comparison table (verify against the PDFs when writing):

| Method | Modality | T2V R@1/R@5/R@10 | V2T R@1/R@5/R@10 |
|---|---|---|---|
| CiCo (CVPR'23) | RGB (I3D) | 56.6 / 69.9 / 74.7 | 51.6 / 64.8 / 70.1 |
| SEDS (MM'24) | RGB + pose | 62.5 / 75.1 / 80.1 | 57.9 / 70.4 / 74.9 |
| **HARP (ours)** | **pose only** | ? | ? |

**Honest expectation:** we are pose-only; SEDS/CiCo use RGB (+pretrained I3D /
video features). We will likely land BELOW them. That is fine and must be framed
correctly: the claim is NOT "we beat SEDS" — it is (a) the first hand-aware
pose-only system evaluated on How2Sign, (b) competitive recall without any RGB,
at a fraction of the compute and with privacy preserved, and (c) **the hand-aware
vs uniform-fusion gap replicating on a second language (ASL) and dataset** — that
replication is the real prize for the paper (it generalises contribution #2).

## Fairness checklist (all enforced by the scripts here)

1. **Official split only** — train/val/test exactly as distributed; no re-split,
   no filtering of test rows. Every test row stays in the pool even if keypoint
   extraction fails (a failed row scores rank ∞ for us — disclosed, not dropped).
2. **No iSign-style text cleaning** — captions used as-is (whitespace-trimmed
   only). Our cleaning was iSign-specific (textbook artefacts); applying it here
   would change the benchmark.
3. **Full test pool (2,356)** for headline numbers — same as CiCo/SEDS. Both
   directions (T2V and V2T), R@1/5/10 + MdR + MnR.
4. **Report cosine AND re-ranked** numbers separately. Baselines may or may not
   use dual-softmax at inference; giving both makes comparison possible either
   way and avoids hidden-trick advantages.
5. **Same recipe as the iSign headline** (50 epochs, batch 128, EMA 0.999,
   lr 2e-4/1e-5, max 512 frames, seed 42) — no per-dataset tuning beyond what
   prior work also treats as standard. Any deviation must be listed in the paper.
6. **Same keypoint protocol** — MediaPipe Holistic 1662-d, shoulder-midpoint/width
   normalisation, extracted with the identical code path as iSign.
7. Run the hand-aware ablation row (`no_handaware`) on How2Sign too — the
   transfer claim needs the SAME comparison, not just the headline.

## Pipeline (run on the server, in order)

```
1. download_how2sign.sh      # gdown the 3 clip zips + 3 CSVs  (~35 GB)
2. extract_how2sign.py       # MediaPipe Holistic -> .npy per clip (CPU, hours)
                             #   --delete-after frees each clip once extracted
3. build_how2sign_manifest.py# official CSVs -> our manifest format
4. run_how2sign.sh           # train (HARP + no_handaware) + evaluate full pool
```

Disk budget: 35 GB clips + ~40 GB keypoints; delete clips as you extract
(`--delete-after`) to stay under the 76 GB free.

License note: How2Sign is CC BY-NC 4.0 (non-commercial research use — our use
qualifies); cite Duarte et al., CVPR 2021, and respect the terms. Do NOT
redistribute the data; the paper reports numbers only.
