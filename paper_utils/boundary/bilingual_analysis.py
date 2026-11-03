"""Vocabulary allocation and compression diagnostics for bilingual tokenizers."""

from __future__ import annotations

from collections import Counter, defaultdict
import unicodedata
from typing import Any

import regex

from paper_utils.boundary.vocab_duplicates import decase, despace, special_texts, surface


CATEGORIES = ("latin", "hangul", "mixed", "other_script", "nonlexical", "undecodable")
LATIN = regex.compile(r"\p{Script=Latin}")
HANGUL = regex.compile(r"\p{Script=Hangul}")


def _character_script(char: str) -> str | None:
    """Return the lexical script bucket for one Unicode character, if it has one."""
    if not unicodedata.category(char).startswith(("L", "M")):
        return None
    if LATIN.fullmatch(char):
        return "latin"
    if HANGUL.fullmatch(char):
        return "hangul"
    # A combining mark has no useful script name on its own. It should not turn an
    # otherwise Latin token such as ``e\u0301`` into a mixed-script token.
    if unicodedata.category(char).startswith("M"):
        return None
    return "other_script"


def _classify_text(text: str) -> str:
    scripts = {_character_script(char) for char in text}
    scripts.discard(None)
    if not scripts:
        return "nonlexical"
    if len(scripts) > 1:
        return "mixed"
    return scripts.pop()


def _decode_ordinary_runs(pretokenizer: Any, atomic_tokens, specials: dict[int, str]) -> str | None:
    """Strictly decode non-special runs, omitting boundary and case-control atomics."""
    text_parts: list[str] = []
    run: list[int] = []
    for token_id in atomic_tokens:
        if token_id in specials:
            if run:
                decoded = pretokenizer.try_decode_strict(run)
                if decoded is None:
                    return None
                text_parts.append(decoded)
                run = []
        else:
            run.append(int(token_id))
    if run:
        decoded = pretokenizer.try_decode_strict(run)
        if decoded is None:
            return None
        text_parts.append(decoded)
    return "".join(text_parts)


def _duplicate_measure(entries: list[tuple[str, str]], key) -> dict[str, Any]:
    """Count whole duplicate groups and their avoidable ``n - 1`` slots."""
    groups: dict[str, list[str]] = defaultdict(list)
    for form, category in entries:
        groups[key(form)].append(category)

    involved = 0
    surplus = 0
    by_script = {category: {"involved_entries": 0, "surplus_entries": 0} for category in CATEGORIES}
    for categories in groups.values():
        size = len(categories)
        if size < 2:
            continue
        involved += size
        surplus += size - 1
        category_counts = Counter(categories)
        for category, count in category_counts.items():
            by_script[category]["involved_entries"] += count

        # A duplicate group normally has one script bucket. If an unusual Unicode
        # transformation joins buckets, assign its n - 1 surplus to the bucket with
        # the most entries, making every reported surplus add up to the total.
        group_category = min(category_counts, key=lambda c: (-category_counts[c], c))
        by_script[group_category]["surplus_entries"] += size - 1

    return {
        "involved_entries": involved,
        "surplus_entries": surplus,
        "by_script": by_script,
    }


def _vocabulary_analysis(tokenizer: Any) -> tuple[dict[str, Any], dict[int, str]]:
    pretokenizer = tokenizer.pretokenizer
    specials = special_texts(pretokenizer)
    category_by_id: dict[int, str] = {}
    learned_entries: list[tuple[str, str]] = []
    categories = Counter()
    atomic_categories = Counter()

    for token in tokenizer.tokens.values():
        decoded = _decode_ordinary_runs(pretokenizer, token.atomic_tokens, specials)
        category = "undecodable" if decoded is None else _classify_text(decoded)
        category_by_id[int(token.id)] = category
        if len(token.atomic_tokens) == 1:
            atomic_categories[category] += 1
            continue
        categories[category] += 1
        learned_entries.append((surface(pretokenizer, token.atomic_tokens, specials), category))

    forms = [form for form, _ in learned_entries]
    if len(forms) != len(set(forms)):
        raise ValueError("Vocabulary surface collisions would invalidate duplicate counts")
    return {
        "total_tokens": len(tokenizer.tokens),
        "atomic_tokens": sum(atomic_categories.values()),
        "atomic_categories": {category: atomic_categories[category] for category in CATEGORIES},
        "learned_tokens": sum(categories.values()),
        "categories": {category: categories[category] for category in CATEGORIES},
        "duplicates": {
            "space": _duplicate_measure(learned_entries, despace),
            "case": _duplicate_measure(learned_entries, decase),
            "combined": _duplicate_measure(learned_entries, lambda form: decase(despace(form))),
        },
    }, category_by_id


def _evaluate(tokenizer: Any, eval_texts: dict[str, list[str]], category_by_id: dict[int, str]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for language, documents in eval_texts.items():
        characters = byte_count = token_count = roundtrip_failures = 0
        usage = Counter()
        used_token_ids: set[int] = set()
        for document in documents:
            expected = tokenizer.pretokenizer.normalize(document)
            token_ids = tokenizer.encode(document)
            token_count += len(token_ids)
            characters += len(expected)
            byte_count += len(expected.encode("utf-8"))
            if tokenizer.decode(token_ids) != expected:
                roundtrip_failures += 1
            for token_id in token_ids:
                token_id = int(token_id)
                usage[category_by_id.get(token_id, "undecodable")] += 1
                used_token_ids.add(token_id)
        unique_usage = Counter(category_by_id.get(token_id, "undecodable") for token_id in used_token_ids)
        results[language] = {
            "documents": len(documents),
            "characters": characters,
            "bytes": byte_count,
            "tokens": token_count,
            "chars_per_token": characters / token_count if token_count else 0.0,
            "bytes_per_token": byte_count / token_count if token_count else 0.0,
            "roundtrip_failures": roundtrip_failures,
            "script_usage": {category: usage[category] for category in CATEGORIES},
            "unique_token_counts": {category: unique_usage[category] for category in CATEGORIES},
        }
    return results


def analyze_tokenizer(tokenizer, eval_texts: dict[str, list[str]]) -> dict[str, Any]:
    """Return JSON-safe vocabulary and streamed evaluation diagnostics for ``tokenizer``.

    Evaluation text is encoded and discarded one document at a time. This intentionally
    avoids retaining a corpus-sized collection of token ids on machines running the grid.
    """
    vocabulary, category_by_id = _vocabulary_analysis(tokenizer)
    return {"vocabulary": vocabulary, "evaluation": _evaluate(tokenizer, eval_texts, category_by_id)}
