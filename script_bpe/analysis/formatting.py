"""LaTeX and output formatting utilities."""

import pandas as pd


def format_with_relchange(value: float, baseline: float, decimals: int = 4) -> str:
    """
    Format value with \\relchange superscript showing % change vs baseline.

    Args:
        value: The value to format
        baseline: The baseline value to compare against
        decimals: Number of decimal places for the main value

    Returns:
        LaTeX string like "0.1234\\relchange{+1.23}"
    """
    rel_pct = (value - baseline) / baseline * 100
    return f"{value:.{decimals}f}\\relchange{{{rel_pct:+.2f}}}"


def format_tokens_millions(
    tokens: float,
    baseline_tokens: float | None = None,
    decimals: int = 1,
) -> str:
    """
    Format token count in millions, optionally with relchange.

    Args:
        tokens: Token count
        baseline_tokens: If provided, add relchange superscript
        decimals: Number of decimal places

    Returns:
        Formatted string like "58.7" or "58.7\\relchange{+1.2}"
    """
    tokens_m = tokens / 1e6
    if baseline_tokens is None:
        return f"{tokens_m:.{decimals}f}"
    rel_pct = (tokens - baseline_tokens) / baseline_tokens * 100
    return f"{tokens_m:.{decimals}f}\\relchange{{{rel_pct:+.{decimals}f}}}"


def format_latex_value(value, lower_is_better: bool = True, threshold: float = 0.5) -> str:
    """
    Format a value with LaTeX highlighting commands for outliers.

    Args:
        value: The numeric value to format
        lower_is_better: If True, negative values are good, positive are bad
        threshold: Threshold for outlier highlighting (absolute value)

    Returns:
        Formatted LaTeX string with \\goodoutlier or \\badoutlier command if applicable
    """
    if pd.isna(value):
        return "---"

    # Format the number
    formatted = f"{value:+.2f}" if value != 0 else "0.00"

    # Determine which command to use
    if lower_is_better:
        if value > threshold:
            return f"\\badoutlier{{{formatted}}}"
        elif value < -threshold:
            return f"\\goodoutlier{{{formatted}}}"
    else:
        if value < -threshold:
            return f"\\badoutlier{{{formatted}}}"
        elif value > threshold:
            return f"\\goodoutlier{{{formatted}}}"

    return formatted


def format_vocab_value(value) -> str:
    """
    Format vocabulary overlap percentage.

    Args:
        value: The numeric value (0-100 scale)

    Returns:
        Formatted string with 1 decimal place, or "---" if NaN
    """
    if pd.isna(value):
        return "---"
    return f"{value:.1f}"


def mark_biggest_in_group(
    all_formatted_values: list[list[str]],
    all_raw_values: list[list[float]],
) -> list[list[str]]:
    """
    Find biggest absolute value across all corpora in a group and mark with \\biggestoutlier.

    Args:
        all_formatted_values: List of lists (one per corpus) of already-formatted LaTeX strings
        all_raw_values: List of lists (one per corpus) of raw numeric values

    Returns:
        List of lists with the biggest value marked with \\biggestoutlier
    """
    if not all_raw_values or not all_raw_values[0]:
        return all_formatted_values

    # Flatten all values to find global maximum
    flat_values = []
    for corpus_values in all_raw_values:
        for val in corpus_values:
            if not pd.isna(val):
                flat_values.append(abs(val))

    if not flat_values:
        return all_formatted_values

    max_abs = max(flat_values)

    # Find and mark the first occurrence of this maximum
    result = [corpus_vals.copy() for corpus_vals in all_formatted_values]
    for corpus_idx, corpus_raw_values in enumerate(all_raw_values):
        for val_idx, val in enumerate(corpus_raw_values):
            if not pd.isna(val) and abs(val) == max_abs:
                result[corpus_idx][val_idx] = f"\\biggestoutlier{{{all_formatted_values[corpus_idx][val_idx]}}}"
                return result

    return result

