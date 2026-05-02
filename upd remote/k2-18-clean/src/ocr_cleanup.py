#!/usr/bin/env python3
"""
OCR cleanup/profiling utility for large text corpora.

The module is intentionally conservative: it can profile likely OCR artifacts
and apply deterministic fixes that are safe enough to run before slicing.
Ambiguous repairs should still go through a reviewed sample workflow.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

WORD_CHAR_CLASS = "A-Za-zА-Яа-яЁёІіѢѣѲѳѴѵ"


@dataclass(frozen=True)
class IssuePattern:
    """Pattern used to profile likely OCR artifacts."""

    name: str
    pattern: str
    description: str


@dataclass(frozen=True)
class RuleResult:
    """Applied cleanup rule and number of replacements."""

    rule: str
    replacements: int


ISSUE_PATTERNS: tuple[IssuePattern, ...] = (
    IssuePattern(
        "soft_hyphen",
        "\u00ad",
        "Discretionary soft hyphen characters left by OCR/layout extraction.",
    ),
    IssuePattern(
        "form_feed",
        "\f",
        "Page-break form feed characters embedded in the text stream.",
    ),
    IssuePattern(
        "page_number_before_break",
        r"(?m)(?:^|\n)[ \t]*\d{1,3}[ \t]*\n(?:[ \t]*\n)*[ \t]*\f",
        "1-3 digit standalone page number directly before a page break.",
    ),
    IssuePattern(
        "standalone_number_line",
        r"(?m)^[ \t]*\d{1,4}[ \t]*$",
        "Standalone numeric lines; these are candidates, not automatic removals.",
    ),
    IssuePattern(
        "soft_hyphen_linebreak",
        rf"[{WORD_CHAR_CLASS}]\u00ad[ \t]*\n[ \t]*[{WORD_CHAR_CLASS}]",
        "A word split by soft hyphen and a physical line break.",
    ),
    IssuePattern(
        "plain_hyphen_linebreak",
        rf"[{WORD_CHAR_CLASS}]-[ \t]*\n[ \t]*[{WORD_CHAR_CLASS}]",
        "A word probably split by regular hyphen and a physical line break.",
    ),
    IssuePattern(
        "letter_digit_letter",
        r"[А-Яа-яЁёA-Za-z]\d[А-Яа-яЁёA-Za-z]",
        "Digit inserted inside a word-like token.",
    ),
    IssuePattern(
        "mixed_script_word",
        r"(?i)\b(?=[A-Za-zА-Яа-яЁё]*[A-Za-z])(?=[A-Za-zА-Яа-яЁё]*[А-Яа-яЁё])[A-Za-zА-Яа-яЁё]{4,}\b",
        "Single token mixing Latin and Cyrillic letters.",
    ),
    IssuePattern(
        "spaced_letters",
        r"(?m)(?:\b[А-Яа-яЁёA-Za-z]\s+){3,}[А-Яа-яЁёA-Za-z]\b",
        "Heading-like words printed as separated letters.",
    ),
    IssuePattern(
        "comma_dash_without_space",
        r"[,;:]\s*[—-](?=\S)",
        "Dash glued to punctuation and the following word.",
    ),
    IssuePattern(
        "dash_between_letters_without_space",
        r"[А-Яа-яЁёA-Za-z]—[А-Яа-яЁёA-Za-z]",
        "Em dash glued between two letters.",
    ),
)


def normalize_newlines(text: str) -> str:
    """Normalize line endings without changing paragraph structure."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def snippet(text: str, start: int, end: int, radius: int = 80) -> str:
    """Return a compact escaped context around a match."""

    left = max(0, start - radius)
    right = min(len(text), end + radius)
    sample = text[left:right]
    return sample.replace("\n", "⏎").replace("\f", "␌").replace("\u00ad", "¬")


def profile_text(text: str, max_examples: int = 5) -> dict[str, Any]:
    """
    Count likely OCR artifacts and collect examples.

    Args:
        text: Source text.
        max_examples: Max examples per issue.

    Returns:
        JSON-serializable profile dictionary.
    """

    issues: dict[str, Any] = {}
    for issue in ISSUE_PATTERNS:
        count = 0
        examples: list[str] = []
        for match in re.finditer(issue.pattern, text):
            count += 1
            if len(examples) < max_examples:
                examples.append(snippet(text, match.start(), match.end()))
        issues[issue.name] = {
            "count": count,
            "description": issue.description,
            "examples": examples,
        }

    return {
        "chars": len(text),
        "lines": text.count("\n") + 1 if text else 0,
        "issues": issues,
    }


def _subn(text: str, pattern: str, repl: str) -> tuple[str, int]:
    """re.subn wrapper with multiline defaults used by cleanup rules."""

    return re.subn(pattern, repl, text)


def join_wrapped_lines(text: str) -> tuple[str, int]:
    """
    Join single physical line breaks inside paragraphs.

    Blank lines are kept as paragraph boundaries. The function is deliberately
    simple and should be enabled only after sampling the corpus, because poems,
    tables, and formulas may need a domain-specific exception list.
    """

    def looks_structural(line: str) -> bool:
        if re.fullmatch(r"\d{1,3}", line):
            return True
        if line.startswith(("§", "№")):
            return True
        if re.fullmatch(r"[IVXLCDMivxlcdmІVXХLCMνш]{1,8}", line):
            return True
        if re.fullmatch(r"(?:[А-ЯЁA-Z]\s+){2,}[А-ЯЁA-Z]", line):
            return True

        letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", line)
        return bool(letters) and len(line) <= 80 and line == line.upper()

    before = text
    text = re.sub(r"[ \t]+\n", "\n", text)
    output: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            output.append(re.sub(r"[ \t]{2,}", " ", " ".join(paragraph)))
            paragraph.clear()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            if output and output[-1] != "":
                output.append("")
            continue

        if looks_structural(line):
            flush_paragraph()
            output.append(line)
            continue

        paragraph.append(line)

    flush_paragraph()
    text = "\n".join(output)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, 0 if text == before else before.count("\n") - text.count("\n")


def clean_text(
    text: str,
    *,
    join_lines: bool = False,
    fix_plain_hyphen_linebreaks: bool = False,
) -> tuple[str, list[RuleResult]]:
    """
    Apply conservative OCR cleanup rules.

    Args:
        text: Source text.
        join_lines: Join line-wrapped prose inside paragraphs.
        fix_plain_hyphen_linebreaks: Also dehyphenate regular '-' line breaks.

    Returns:
        Tuple of cleaned text and applied rule statistics.
    """

    rules: list[RuleResult] = []

    normalized = unicodedata.normalize("NFC", normalize_newlines(text))
    rules.append(RuleResult("unicode_nfc_and_newlines", int(normalized != text)))
    text = normalized

    # Remove page numbers only when they are strongly tied to a page break.
    text, count = _subn(
        text,
        r"(?m)(^|\n)[ \t]*\d{1,3}[ \t]*\n(?:[ \t]*\n)*[ \t]*\f",
        r"\1\f",
    )
    rules.append(RuleResult("remove_1_3_digit_page_number_before_form_feed", count))

    text, count = _subn(text, r"(?m)\f[ \t]*\n?[ \t]*\d{1,3}[ \t]*(?=\n)", "\f")
    rules.append(RuleResult("remove_1_3_digit_page_number_after_form_feed", count))

    text, count = _subn(
        text,
        rf"([{WORD_CHAR_CLASS}])\u00ad[ \t]*\n[ \t]*([{WORD_CHAR_CLASS}])",
        r"\1\2",
    )
    rules.append(RuleResult("join_soft_hyphen_linebreak_words", count))

    if fix_plain_hyphen_linebreaks:
        text, count = _subn(
            text,
            rf"([{WORD_CHAR_CLASS}])-[ \t]*\n[ \t]*([{WORD_CHAR_CLASS}])",
            r"\1\2",
        )
        rules.append(RuleResult("join_plain_hyphen_linebreak_words", count))

    text, count = _subn(text, "\u00ad", "")
    rules.append(RuleResult("remove_remaining_soft_hyphens", count))

    text, count = _subn(text, "\f", "\n\n")
    rules.append(RuleResult("replace_form_feeds_with_paragraph_breaks", count))

    text, count = _subn(text, r"([,;:])\s*[—-](?=\S)", r"\1 — ")
    rules.append(RuleResult("space_punctuation_dash_word", count))

    text, count = _subn(
        text,
        r"([А-Яа-яЁёA-Za-z])—([А-Яа-яЁёA-Za-z])",
        r"\1 — \2",
    )
    rules.append(RuleResult("space_letter_dash_letter", count))

    text, count = _subn(text, r"[ \t]{2,}", " ")
    rules.append(RuleResult("collapse_horizontal_spaces", count))

    text, count = _subn(text, r"\n{4,}", "\n\n\n")
    rules.append(RuleResult("cap_long_blank_runs", count))

    if join_lines:
        text, count = join_wrapped_lines(text)
        rules.append(RuleResult("join_wrapped_lines", count))

    return text, rules


def build_report(
    *,
    input_path: Path,
    output_path: Path | None,
    before: dict[str, Any],
    after: dict[str, Any] | None = None,
    rules: list[RuleResult] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable profile/cleanup report."""

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "output": str(output_path) if output_path else None,
        "before": before,
        "after": after,
        "rules": [rule.__dict__ for rule in rules or []],
    }


def read_text(path: Path) -> str:
    """Read UTF-8 text."""

    return path.read_text(encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command_profile(args: argparse.Namespace) -> int:
    """Run profile command."""

    input_path = Path(args.input)
    text = read_text(input_path)
    report = build_report(
        input_path=input_path,
        output_path=None,
        before=profile_text(text, max_examples=args.examples),
    )
    if args.report:
        write_json(Path(args.report), report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_clean(args: argparse.Namespace) -> int:
    """Run clean command."""

    input_path = Path(args.input)
    output_path = Path(args.output)
    text = read_text(input_path)
    before = profile_text(text, max_examples=args.examples)
    cleaned, rules = clean_text(
        text,
        join_lines=args.join_lines,
        fix_plain_hyphen_linebreaks=args.fix_plain_hyphen_linebreaks,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")

    report = build_report(
        input_path=input_path,
        output_path=output_path,
        before=before,
        after=profile_text(cleaned, max_examples=args.examples),
        rules=rules,
    )
    if args.report:
        write_json(Path(args.report), report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""

    parser = argparse.ArgumentParser(
        description="Profile and conservatively clean OCR artifacts before K2-18 slicing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Profile likely OCR artifacts.")
    profile.add_argument("--input", required=True, help="UTF-8 source text path.")
    profile.add_argument("--report", help="Optional JSON report path.")
    profile.add_argument("--examples", type=int, default=5, help="Examples per issue.")
    profile.set_defaults(func=command_profile)

    clean = subparsers.add_parser("clean", help="Apply conservative cleanup rules.")
    clean.add_argument("--input", required=True, help="UTF-8 source text path.")
    clean.add_argument("--output", required=True, help="Cleaned UTF-8 text path.")
    clean.add_argument("--report", help="Optional JSON report path.")
    clean.add_argument("--examples", type=int, default=5, help="Examples per issue.")
    clean.add_argument(
        "--join-lines",
        action="store_true",
        help="Join physical line wraps inside paragraphs after reviewing samples.",
    )
    clean.add_argument(
        "--fix-plain-hyphen-linebreaks",
        action="store_true",
        help="Also join words split by regular '-' line breaks.",
    )
    clean.set_defaults(func=command_clean)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
