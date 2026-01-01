from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfUnionConfig:
    ngram_max: int = 2
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    word_max_features: int = 250_000
    char_max_features: int = 150_000


def build_tfidf_union(cfg: TfidfUnionConfig) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, cfg.ngram_max),
                    min_df=cfg.min_df,
                    max_df=cfg.max_df,
                    sublinear_tf=cfg.sublinear_tf,
                    max_features=cfg.word_max_features,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=max(2, cfg.min_df),
                    max_df=min(0.98, cfg.max_df + 0.02),
                    sublinear_tf=True,
                    max_features=cfg.char_max_features,
                    dtype=np.float32,
                ),
            ),
        ]
    )
