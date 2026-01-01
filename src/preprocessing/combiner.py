from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import pandas as pd


def _to_str_or_empty(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "<na>"} else s


@dataclass(frozen=True)
class TextCombiner:
    text_cols: Sequence[str]
    sep: str = " "

    def combine_row(self, row: Mapping[str, Any]) -> str:
        parts = []
        for c in self.text_cols:
            v = _to_str_or_empty(row.get(c, ""))
            if v:
                parts.append(v)
        return self.sep.join(parts).strip()

    def combine_df(self, df: pd.DataFrame, out_col: str = "text_for_cluster") -> pd.DataFrame:
        df = df.copy()
        df[out_col] = df.apply(lambda r: self.combine_row(r), axis=1)
        return df