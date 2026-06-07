"""
Build a cleaned/canonical text manifest for SignPose2Text training.

The source iSign text contains textbook headers, activity prompts, page markers,
spelling noise, and very short non-semantic fragments. This script keeps the
original rows intact, adds a canonical text target, and optionally writes rejected
rows for audit.

Example:
  python src/data/build_canonical_text_manifest.py \
      --manifest /mnt/ngan/ISL-Sequences/manifests/isign_sentence_train_qc_20260603_1540.csv \
      --output /mnt/ngan/ISL-Sequences/manifests/isign_signpose2text_clean.csv \
      --reject-output /mnt/ngan/ISL-Sequences/manifests/isign_signpose2text_reject.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import string
from collections import Counter
from pathlib import Path


DEFAULT_KEEP_CATEGORIES = ("general_sentence", "example_sentence", "description")

DROP_EXACT = {
    "reading is fun",
    "new words",
    "new words learnt from the poem",
    "say aloud",
    "let's talk",
    "lets talk",
    "let's read",
    "lets read",
    "let's write",
    "lets write",
    "let's share",
    "lets share",
    "let's listen",
    "lets listen",
    "let's do",
    "lets do",
    "let's make",
    "lets make",
    "let's practice",
    "lets practice",
    "word fun",
    "fun time",
    "picture story",
    "creative writing",
    "riddle time",
    "team time",
    "talk time",
    "composition corner",
    "material required",
    "method",
    "answer",
    "question",
    "blank",
    "dash",
}

DROP_PATTERNS = [
    re.compile(r"^page(?:\s+number)?\.?\s*\d+\.?$", re.IGNORECASE),
    re.compile(r"^page\s+(?:no\.?\s*)?\d+\.?$", re.IGNORECASE),
    re.compile(r"^page(?:\s+number)?\s+[a-z]+\.?$", re.IGNORECASE),
    re.compile(r"^page\.?$", re.IGNORECASE),
    re.compile(r"^unit\s+\d+\.?$", re.IGNORECASE),
    re.compile(r"^chapter\s+\d+\.?$", re.IGNORECASE),
    re.compile(r"^\d+\.?$"),
    re.compile(r"^[a-z]$", re.IGNORECASE),
    re.compile(r"^(?:[a-z]\s*){1,3}$", re.IGNORECASE),
    re.compile(r"^to the teacher\.?$", re.IGNORECASE),
    re.compile(r"^one has been done for you!?\.?$", re.IGNORECASE),
    re.compile(r"^now read further\.?$", re.IGNORECASE),
    re.compile(r"^now fill in the blanks\.?$", re.IGNORECASE),
    re.compile(r"^fill in the blanks\.?$", re.IGNORECASE),
    re.compile(r"^complete the sentences?\.?$", re.IGNORECASE),
    re.compile(r"^look at (?:the|these|this) pictures?\.?$", re.IGNORECASE),
    re.compile(r"^here is the space provided\.?$", re.IGNORECASE),
]

NOISY_SUBSTRINGS = [
    "space provided",
    "connect the dots",
    "trace the dotted",
    "trace over the dotted",
    "join the dots",
    "draw along the dots",
    "circle the letters",
    "tick the objects",
    "write in the space",
    "use the space provided",
    "one has been done",
]

OCR_FIXES = [
    (re.compile(r"\bpleace\b", re.IGNORECASE), "please"),
    (re.compile(r"\bcrowl\b", re.IGNORECASE), "crow"),
    (re.compile(r"\bcircel\b", re.IGNORECASE), "circle"),
    (re.compile(r"\bshot\b", re.IGNORECASE), "shut"),
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


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, lowercase: bool) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = normalize_spaces(text)

    for pattern, replacement in OCR_FIXES:
        text = pattern.sub(replacement, text)

    text = normalize_spaces(text)
    if lowercase:
        text = text.lower()
    return text


def strip_numbering(text: str) -> str:
    text = re.sub(r"^\s*(?:question|answer)?\s*\d+\s*[\.\)]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[a-z]\s*[\.\)]\s*", "", text, flags=re.IGNORECASE)
    return normalize_spaces(text)


def strip_page_markers(text: str) -> str:
    # Remove mixed headers such as "Page number 106. Reading is fun." while
    # preserving the meaningful sentence that may follow.
    text = re.sub(
        r"^\s*page(?:\s+number)?\.?\s*\d+\s*[,:\.-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*unit\s+\d+\s*[,:\.-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*chapter\s+\d+\s*[,:\.-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+page(?:\s+number)?\.?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+page(?:\s+number)?\.?\s*\d+\.?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+unit\s+\d+\.?\s*$", "", text, flags=re.IGNORECASE)
    return normalize_spaces(text)


def strip_activity_prefixes(text: str) -> str:
    prefixes = [
        r"reading is fun[.!?]?\s*",
        r"let'?s (?:read|write|talk|share|listen|do|make|practice)[.!?]?\s*",
        r"say aloud[.!?]?\s*",
        r"new words(?: learnt from the poem)?[.!?]?\s*",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            next_text = re.sub(r"^\s*" + prefix, "", text, flags=re.IGNORECASE)
            if next_text != text:
                text = next_text
                changed = True
    return normalize_spaces(text)


def trim_punctuation_for_match(text: str) -> str:
    return text.strip().strip(string.punctuation + " ").lower()


def alpha_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)


def repeated_char_noise(text: str) -> bool:
    tokens = alpha_tokens(text)
    if not tokens:
        return False
    long_repeated = [token for token in tokens if re.fullmatch(r"([A-Za-z])\1{3,}", token)]
    return len(long_repeated) > 0 and len(long_repeated) == len(tokens)


def should_drop_text(
    raw_text: str,
    canonical: str,
    min_words: int,
    max_words: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    compact = trim_punctuation_for_match(canonical)
    tokens = alpha_tokens(canonical)

    if not compact:
        reasons.append("empty_text")
    if compact in DROP_EXACT:
        reasons.append("textbook_heading")
    if any(pattern.match(canonical) for pattern in DROP_PATTERNS):
        reasons.append("textbook_or_page_pattern")
    if any(fragment in compact for fragment in NOISY_SUBSTRINGS):
        reasons.append("activity_instruction")
    if len(tokens) < min_words:
        reasons.append("too_few_words")
    if len(tokens) > max_words:
        reasons.append("too_many_words")
    if repeated_char_noise(canonical):
        reasons.append("repeated_character_noise")
    if sum(ch.isalpha() for ch in canonical) == 0:
        reasons.append("no_alpha")

    # Drop rows where punctuation-only or dash placeholders dominate.
    placeholder_count = len(re.findall(r"\b(?:blank|dash)\b", raw_text, flags=re.IGNORECASE))
    if placeholder_count > 0:
        reasons.append("placeholder_token")

    return bool(reasons), reasons


def process_rows(
    rows: list[dict[str, str]],
    keep_categories: set[str],
    text_column: str,
    output_text_column: str,
    lowercase: bool,
    min_words: int,
    max_words: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    kept: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []

    for row in rows:
        category = row.get("category", "").strip()
        next_row = dict(row)
        reasons: list[str] = []

        if keep_categories and category not in keep_categories:
            reasons.append("category_not_kept")

        raw_text = row.get(text_column, "")
        canonical = normalize_text(
            strip_activity_prefixes(strip_page_markers(strip_numbering(raw_text))),
            lowercase=lowercase,
        )
        drop_text, text_reasons = should_drop_text(
            raw_text=raw_text,
            canonical=canonical,
            min_words=min_words,
            max_words=max_words,
        )
        if drop_text:
            reasons.extend(text_reasons)

        next_row[output_text_column] = canonical
        next_row["text_clean_status"] = "reject" if reasons else "keep"
        next_row["text_clean_reasons"] = "|".join(sorted(set(reasons)))
        next_row["text_num_words"] = str(len(alpha_tokens(canonical)))

        if reasons:
            rejected.append(next_row)
        else:
            kept.append(next_row)

    return kept, rejected


def ensure_fields(fieldnames: list[str], output_text_column: str) -> list[str]:
    fields = list(fieldnames)
    for field in (
        output_text_column,
        "text_clean_status",
        "text_clean_reasons",
        "text_num_words",
    ):
        if field not in fields:
            fields.append(field)
    return fields


def print_summary(
    kept: list[dict[str, str]],
    rejected: list[dict[str, str]],
    output_text_column: str,
) -> None:
    all_rows = kept + rejected
    status_counts = Counter(row.get("text_clean_status", "") for row in all_rows)
    category_kept = Counter(row.get("category", "") for row in kept)
    reject_reasons = Counter()
    for row in rejected:
        for reason in row.get("text_clean_reasons", "").split("|"):
            if reason:
                reject_reasons[reason] += 1

    print(f"rows input   : {len(all_rows)}")
    print(f"rows kept    : {len(kept)}")
    print(f"rows rejected: {len(rejected)}")
    print("by status:")
    for status, count in status_counts.most_common():
        print(f"  {status:12s} {count}")
    print("kept by category:")
    for category, count in category_kept.most_common():
        print(f"  {category or '<blank>':18s} {count}")
    print("reject reasons:")
    for reason, count in reject_reasons.most_common():
        print(f"  {reason:24s} {count}")
    if kept:
        print(f"target column: {output_text_column}")
        print("examples:")
        for row in kept[:5]:
            print(f"  {row.get('uid', '')}: {row.get(output_text_column, '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reject-output", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-text-column", default="canonical_text")
    parser.add_argument(
        "--keep-category",
        action="append",
        default=list(DEFAULT_KEEP_CATEGORIES),
        help="Category to keep. Pass multiple times. Use --keep-category all to disable filtering.",
    )
    parser.add_argument("--preserve-case", action="store_true")
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fieldnames, rows = read_csv(args.manifest)

    keep_categories = set(args.keep_category)
    if "all" in keep_categories:
        keep_categories = set()

    kept, rejected = process_rows(
        rows=rows,
        keep_categories=keep_categories,
        text_column=args.text_column,
        output_text_column=args.output_text_column,
        lowercase=not args.preserve_case,
        min_words=args.min_words,
        max_words=args.max_words,
    )

    fields = ensure_fields(fieldnames, args.output_text_column)
    write_csv(args.output, fields, kept)
    if args.reject_output:
        write_csv(args.reject_output, fields, rejected)
    print_summary(kept, rejected, args.output_text_column)
    print(f"output       : {args.output}")
    if args.reject_output:
        print(f"reject output: {args.reject_output}")


if __name__ == "__main__":
    main()
