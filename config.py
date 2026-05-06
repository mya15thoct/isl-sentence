"""
Configuration for ISL Sentence Recognition project.
Adjust paths before running on server.

Server layout (READ-ONLY except ISL-Sequences):
  /mnt/ngan/recognition/           ← Base word model sequences (READ ONLY)
  /mnt/ngan/ISL-Frames-Data/       ← Static images (DO NOT TOUCH)
  /mnt/ngan/vsl_data/              ← Video gloss classes (DO NOT TOUCH)
  /mnt/ngan/ISL-Sequences/         ← ALL isl-sentence outputs go here ✅
"""
from pathlib import Path

# ── Local paths (for development) ────────────────────────────────────────────
ROOT_DIR  = Path(__file__).parent
SRC_DIR   = ROOT_DIR / 'src'

# ── Server paths ──────────────────────────────────────────────────────────────
SERVER_BASE      = Path('/mnt/ngan')

# Working directory — ALL outputs go here
ISL_SEQ_DIR      = SERVER_BASE / 'ISL-Sequences'
SEQUENCE_PATH    = ISL_SEQ_DIR / 'sequences'          # extracted .npy keypoints
LABEL_PATH       = ISL_SEQ_DIR / 'labels.csv'         # sentence gloss labels CSV
VIDEO_PATH       = ISL_SEQ_DIR / 'videos'             # raw sentence videos
CHECKPOINT_DIR   = ISL_SEQ_DIR / 'checkpoints'        # sentence model checkpoints
LOGS_DIR         = ISL_SEQ_DIR / 'logs'

# Attention Dictionary output
DICTIONARY_SAVE_PATH = ISL_SEQ_DIR / 'attention_dictionary'
DICTIONARY_PATH      = DICTIONARY_SAVE_PATH / 'dictionary.npy'
ACTION_NAMES_PATH    = DICTIONARY_SAVE_PATH / 'action_names.json'

# Word Model — combined model lives in ISL-Sequences (READ ONLY for isl-sentence)
WORD_MODEL_PATH      = ISL_SEQ_DIR / 'checkpoints' / 'best_model_combined'
WORD_SEQUENCE_PATH   = SERVER_BASE / 'recognition' / 'sequences'   # video keypoints
ISL_WORD_PATH        = ISL_SEQ_DIR / 'word'                         # image keypoints
ACTION_MAPPING_PATH  = ISL_SEQ_DIR / 'checkpoints' / 'action_mapping_combined.json'

# ── Data Config ───────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 200    # max frames for sentence video
KEYPOINT_DIM     = 1662   # mediapipe holistic keypoints

# ── Model Config ──────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'pose_dim':    132,
    'face_dim':    1404,
    'hand_dim':    126,
    'pose_units':  64,
    'face_units':  128,
    'hand_units':  64,
    'bilstm_units_1': 64,
    'bilstm_units_2': 32,
    'dropout':     0.3,
}

# ── Training Config ───────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    'learning_rate':           1e-4,
    'batch_size':              8,      # sentence videos are long → smaller batch
    'epochs':                  200,
    'early_stopping_patience': 20,
    'reduce_lr_patience':      10,
    'val_split':               0.10,
    'lambda_distill':          0.1,   # weight for distillation loss
    'ctc_blank_index':         0,     # CTC blank token index
}
