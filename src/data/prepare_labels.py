"""
Prepare and clean gloss labels from glosses.csv.

Steps:
  1. Fix CSV parsing issues (trailing commas, periods)
  2. Map OOV glosses to in-vocab equivalents (ME → I_ME_MINE_MY, etc.)
  3. Drop ISL function words (IS, AM, BE, A, THE, IN, OF, ...)
  4. Report coverage and save clean labels to ISL-Sequences/labels.json

Usage:
  cd isl-sentence
  python src/data/prepare_labels.py \
    --csv glosses.csv \
    --vocab /mnt/ngan/ISL-Sequences/checkpoints/action_mapping_combined.json \
    --output /mnt/ngan/ISL-Sequences/labels.json
"""

import json
import csv
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ISL_SEQ_DIR, ACTION_MAPPING_PATH

# ─────────────────────────────────────────────────────────────────────────────
# ISL GRAMMAR: Drop these function words
# ISL omits: to-be verbs, articles, prepositions, conjunctions, auxiliaries
# ─────────────────────────────────────────────────────────────────────────────
REMOVE_WORDS = {
    # Articles
    'A', 'AN', 'THE',
    # Copula / auxiliaries
    'AM', 'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING',
    'WILL', 'WOULD', 'COULD', 'SHOULD', 'HAVE', 'HAS', 'HAD',
    # Prepositions (commonly dropped in ISL)
    'IN', 'INTO', 'ON', 'OF', 'BY', 'AT', 'TO', 'FOR', 'WITH',
    'FROM', 'THROUGH', 'OFF',
    # Conjunctions
    'AND', 'OR', 'BUT', 'SO',
    # Special placeholders
    'XXXXXXXX', '(AGE)',
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOSS MAP: OOV → in-vocab equivalent
# ─────────────────────────────────────────────────────────────────────────────
GLOSS_MAP = {
    # Pronoun compounds (ISL-Frames-Data uses I_ME_MINE_MY)
    'ME':        'I_ME_MINE_MY',
    'MY':        'I_ME_MINE_MY',
    'MINE':      'I_ME_MINE_MY',
    'MYSELF':    'I_ME_MINE_MY',
    'YOUR':      'YOU',
    'YOURSELF':  'YOU',
    'HIM':       'HE',
    'HIS':       'HE',
    'HER':       'SHE',

    # Negation
    'DONOT':     'NOT',
    'DONT':      'NOT',
    "DON'T":     'NOT',
    'NO':        'NOT',
    'NEVER':     'NOT',

    # Compound words
    'SOMEHOW':   'SOME_HOW',
    'SOMEONE':   'SOME_ONE',
    'HI':        'HELLO_HI',
    'HELLO':     'HELLO_HI',
    'COLLEGE':   'COLLEGE_SCHOOL',

    # Verb forms → root
    'COMING':    'COME',
    'CAME':      'COME',
    'HIDING':    'HIDING',
    'HIDE':      'HIDING',
    'ENJOYED':   'ENJOY',
    'LIKES':     'LIKE',
    'LOVED':     'LIKE_LOVE',
    'LOVE':      'LIKE_LOVE',
    'STOPPED':   'STOP',
    'SUFFERING': 'HURT',

    # Other mappings
    'CARE':      'TAKE_CARE',
    'TAKE':      'TAKE_CARE',
    'TURN':      'TURN_ON',
    'VERY':      'REALLY',
    'MUCH':      'A_LOT',
    'LOT':       'A_LOT',
    'MORE':      'A_LOT',
    'SOME':      'SOMETHING',
    'KNOW':      'UNDERSTAND',
    'NEED':      'WANT',
    'WAY':       'GO',        # "on the way" → GO (preserves movement meaning)
    'WHEN':      'HOW',       # question word approximation
    'WHY':       'HOW',       # question word approximation
    'WHICH':     'WHAT',
    'ONE':       'NUMBER',
    'THIS':      'THAT',
}

# Words to skip entirely (no reasonable mapping)
SKIP_WORDS = {
    'ABOUT', 'ANY', 'GOT', 'HAIR', 'PLAN', 'CAREER',
    'GLASS', 'ONWARDS', 'NOW', 'MEDICINE', 'FINE',
    'FEELING', 'THERE', 'TRY', 'MAKE', 'LET', 'SO', 'SIR',
}

# Force drop even if token IS in vocab
# (e.g. SCHOOL is in vocab but redundant when COLLEGE → COLLEGE_SCHOOL)
FORCE_DROP = {'SCHOOL'}

# Minimum number of glosses required to keep a sentence
MIN_GLOSSES = 3


def clean_gloss_token(token: str) -> str:
    """Remove trailing punctuation from a gloss token."""
    return token.strip().rstrip('.,;:!?').upper()


def map_gloss(token: str, vocab: set) -> str | None:
    """
    Map a single gloss token to its in-vocab equivalent.
    Returns None if it should be dropped.
    """
    token = clean_gloss_token(token)

    if not token:
        return None

    # Force drop — overrides vocab check
    if token in FORCE_DROP:
        return None

    # Already in vocab → keep as-is
    if token in vocab:
        return token

    # In remove set → drop
    if token in REMOVE_WORDS:
        return None

    # Try direct map
    if token in GLOSS_MAP:
        mapped = GLOSS_MAP[token]
        if mapped in vocab:
            return mapped
        # If mapping also OOV, drop
        return None

    # In skip set → drop
    if token in SKIP_WORDS:
        return None

    # Unknown → drop (will be reported)
    return f'__OOV__{token}'


def prepare_labels(
    csv_path:   str,
    vocab_path: str,
    output_path: str,
    seq_dir:    str = None,
):
    csv_path    = Path(csv_path)
    vocab_path  = Path(vocab_path)
    output_path = Path(output_path)
    seq_dir     = Path(seq_dir) if seq_dir else ISL_SEQ_DIR / 'sequences'

    # Load vocab
    with open(vocab_path) as f:
        mapping = json.load(f)
    vocab = set(v.upper() for v in mapping.values())
    print(f'[VOCAB] {len(vocab)} classes in action_mapping_combined')

    # Read CSV
    labels = {}
    oov_report = {}
    skipped = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row['Sentence'].strip().lower()
            raw_glosses = row['SIGN GLOSSES'].strip().split()

            mapped = []
            row_oov = []
            seen = set()  # dedup: avoid duplicate consecutive tokens from mapping
            for token in raw_glosses:
                result = map_gloss(token, vocab)
                if result is None:
                    pass  # dropped (function word / known drop)
                elif result.startswith('__OOV__'):
                    oov_word = result[7:]
                    row_oov.append(oov_word)
                else:
                    # Avoid duplicate tokens introduced by mapping
                    # e.g. TAKE+CARE both → TAKE_CARE → keep only first
                    if result not in seen:
                        mapped.append(result)
                        seen.add(result)
                    # Reset seen after non-duplicate to allow legitimate repetition
                    # (e.g. WORRY WORRY in "no need to worry dont worry")
                    elif result == (mapped[-1] if mapped else None):
                        pass  # skip consecutive duplicate
                    else:
                        mapped.append(result)

            if row_oov:
                oov_report[sentence] = row_oov

            if not mapped or len(mapped) < MIN_GLOSSES:
                skipped.append((sentence, mapped))
                continue

            labels[sentence] = mapped

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    # Report
    print(f'\n{"=" * 60}')
    print(f'LABEL PREPARATION COMPLETE')
    print(f'{"=" * 60}')
    print(f'  Total sentences in CSV   : {len(labels) + len(skipped)}')
    print(f'  Sentences with labels    : {len(labels)}')
    print(f'  Sentences skipped (< {MIN_GLOSSES} glosses): {len(skipped)}')
    if skipped:
        for sent, gl in skipped:
            print(f'    "{sent}" → {gl}')

    if oov_report:
        print(f'\n  Sentences with remaining OOV (gloss dropped):')
        for sent, oovs in oov_report.items():
            print(f'    "{sent}" → dropped: {oovs}')

    print(f'\n  Saved to: {output_path}')
    print(f'{"=" * 60}')

    # Sample output
    print('\nSample labels:')
    for sent, glosses in list(labels.items())[:5]:
        print(f'  "{sent}" → {glosses}')

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',    type=str,
                        default='glosses.csv',
                        help='Path to glosses.csv')
    parser.add_argument('--vocab',  type=str,
                        default=str(ACTION_MAPPING_PATH),
                        help='Path to action_mapping_combined.json')
    parser.add_argument('--output', type=str,
                        default=str(ISL_SEQ_DIR / 'labels.json'),
                        help='Output labels JSON path')
    parser.add_argument('--seq_dir', type=str,
                        default=str(ISL_SEQ_DIR / 'sequences'),
                        help='Path to extracted .npy sequences')
    args = parser.parse_args()

    prepare_labels(
        csv_path    = args.csv,
        vocab_path  = args.vocab,
        output_path = args.output,
        seq_dir     = args.seq_dir,
    )
