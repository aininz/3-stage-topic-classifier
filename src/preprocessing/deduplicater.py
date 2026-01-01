from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Sequence

import pandas as pd


@dataclass
class DedupReport:
    train_before: int
    val_before: int
    test_before: int

    removed_train_within: int
    removed_val_within: int
    removed_test_within: int

    removed_val_vs_train: int
    removed_test_vs_trainval: int

    train_after: int
    val_after: int
    test_after: int


def _make_key_series(
    df: pd.DataFrame,
    text_col: str,
    key_fn: Optional[Callable[[str], str]],
) -> pd.Series:
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in df columns: {list(df.columns)}")

    s = df[text_col].astype("string").fillna("")
    if key_fn is None:
        return s

    return s.map(lambda x: key_fn(str(x)))


def dedup_within_split(
    df: pd.DataFrame,
    text_col: str = "text",
    key_fn: Optional[Callable[[str], str]] = None,
    keep: str = "first",
) -> Tuple[pd.DataFrame, int]:
    """
    Remove duplicates within a single split based on a derived key (typically normalized text).
    Returns (deduped_df, removed_count).
    """
    df = df.copy()
    key = _make_key_series(df, text_col, key_fn)

    before = len(df)
    df["_dedup_key"] = key
    df = df.drop_duplicates(subset="_dedup_key", keep=keep).drop(columns=["_dedup_key"])
    removed = before - len(df)
    return df, removed


def dedup_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "text",
    key_fn: Optional[Callable[[str], str]] = None,
    dedup_within: bool = True,
    keep_within: str = "first",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DedupReport]:
    """
    Deduplicate:
      1) (optional) within each split
      2) val against train (drop from val)
      3) test against train+val (drop from test)

    Uses key_fn(text) -> key for matching (e.g., TextCleaner.normalize).
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_before, val_before, test_before = len(train_df), len(val_df), len(test_df)

    removed_train_within = removed_val_within = removed_test_within = 0

    if dedup_within:
        train_df, removed_train_within = dedup_within_split(train_df, text_col, key_fn, keep_within)
        val_df, removed_val_within = dedup_within_split(val_df, text_col, key_fn, keep_within)
        test_df, removed_test_within = dedup_within_split(test_df, text_col, key_fn, keep_within)

    train_key = _make_key_series(train_df, text_col, key_fn)
    val_key = _make_key_series(val_df, text_col, key_fn)
    test_key = _make_key_series(test_df, text_col, key_fn)

    train_keys = set(train_key)

    val_mask_keep = ~val_key.isin(train_keys)
    removed_val_vs_train = int((~val_mask_keep).sum())
    val_df = val_df[val_mask_keep].copy()

    train_val_keys = train_keys | set(_make_key_series(val_df, text_col, key_fn))
    test_mask_keep = ~test_key.isin(train_val_keys)
    removed_test_vs_trainval = int((~test_mask_keep).sum())
    test_df = test_df[test_mask_keep].copy()

    report = DedupReport(
        train_before=train_before,
        val_before=val_before,
        test_before=test_before,
        removed_train_within=removed_train_within,
        removed_val_within=removed_val_within,
        removed_test_within=removed_test_within,
        removed_val_vs_train=removed_val_vs_train,
        removed_test_vs_trainval=removed_test_vs_trainval,
        train_after=len(train_df),
        val_after=len(val_df),
        test_after=len(test_df),
    )

    return train_df, val_df, test_df, report


def overlap_counts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "text",
    key_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """
    Quick overlap diagnostics: returns counts of intersections by key.
    """
    tr = set(_make_key_series(train_df, text_col, key_fn))
    va = set(_make_key_series(val_df, text_col, key_fn))
    te = set(_make_key_series(test_df, text_col, key_fn))

    return {
        "train_val": len(tr & va),
        "train_test": len(tr & te),
        "val_test": len(va & te),
        "train": len(tr),
        "val": len(va),
        "test": len(te),
    }

def drop_duplicates_normalized(
    df: pd.DataFrame,
    *,
    subset: Sequence[str],
    normalize_fn: Callable[[str], str],
    keep: str = "first",
    key_col: Optional[str] = None,
    drop_key_col: bool = True,
) -> pd.DataFrame:
    """
    Drop duplicates based on a normalized version of one or more text columns.

    - Does NOT modify original columns.
    - Creates a temporary key column (or uses `key_col` if given).
    - Dedups using that key.

    Example:
        df = drop_duplicates_normalized(
            df,
            subset=["text"],
            normalize_fn=IndonesianNewsCleaner.clean
        )
    """
    if not subset:
        raise ValueError("subset must be non-empty")

    out = df.copy()

    # Build a normalized key Series
    def norm_cell(x) -> str:
        if pd.isna(x):
            return ""
        return normalize_fn(str(x))

    if len(subset) == 1:
        key = out[subset[0]].map(norm_cell)
    else:
        parts = [out[c].map(norm_cell) for c in subset]
        key = parts[0]
        for p in parts[1:]:
            key = key + " " + p
        key = key.str.strip()

    if key_col is None:
        key_col = "__dedup_key__"

    out[key_col] = key

    out = out[out[key_col].str.len() > 0].copy()

    out = out.drop_duplicates(subset=[key_col], keep=keep).copy()

    if drop_key_col:
        out = out.drop(columns=[key_col])

    return out