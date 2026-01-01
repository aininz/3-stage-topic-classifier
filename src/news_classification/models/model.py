from __future__ import annotations

import re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering

import optuna
from optuna.pruners import MedianPruner

from .features import TfidfUnionConfig, build_tfidf_union


@dataclass
class HierarchicalTopicClassifier:
    clf_aggregate: Any
    granular_clfs_by_aggregate: Dict[str, Any]
    fallback_granular_by_aggregate: Dict[str, str]

    def predict(self, X):
        agg = self.clf_aggregate.predict(X)
        agg = np.asarray([str(a) for a in agg], dtype=object)

        out = np.empty(len(agg), dtype=object)

        idx_by_agg = defaultdict(list)
        for i, a in enumerate(agg):
            idx_by_agg[a].append(i)

        for a, idxs in idx_by_agg.items():
            if a in self.fallback_granular_by_aggregate:
                out[idxs] = self.fallback_granular_by_aggregate[a]
                continue

            clf_g = self.granular_clfs_by_aggregate.get(a)
            if clf_g is not None:
                out[idxs] = clf_g.predict(X[idxs])
                continue

            if self.fallback_granular_by_aggregate:
                any_fallback = next(iter(self.fallback_granular_by_aggregate.values()))
                out[idxs] = any_fallback
            else:
                raise RuntimeError(f"Missing granular clf AND fallback for aggregate={a}.")

        return out


class HierarchicalSVMModel:
    """
    ML model:
    - fit(X_train, y_train, X_val, y_val)
    - predict(texts)
    - evaluate(texts, y)
    - save/load
    """

    def __init__(
        self,
        clean_fn: Optional[Callable[[str], str]] = None,
        optuna_trials: int = 25,
        optuna_timeout_sec: int = 600,
        min_bucket_size: int = 80,
        min_label_support_for_cluster: int = 10,
        n_aggregate_clusters: int = 10,
    ):
        self.clean_fn = clean_fn

        self.OPTUNA_TRIALS = int(optuna_trials)
        self.OPTUNA_TIMEOUT_SEC = int(optuna_timeout_sec)

        self.MIN_BUCKET_SIZE = int(min_bucket_size)
        self.MIN_LABEL_SUPPORT_FOR_CLUSTER = int(min_label_support_for_cluster)
        self.N_AGGREGATE_CLUSTERS = int(n_aggregate_clusters)

        self.best_params: Optional[dict] = None
        self.val_macro_f1: Optional[float] = None
        self.test_macro_f1: Optional[float] = None

        self.preprocessor_ = None  # FeatureUnion
        self.clf_: Optional[HierarchicalTopicClassifier] = None
        self.le_: Optional[LabelEncoder] = None
        self.le_aggregate_: Optional[LabelEncoder] = None
        self.aggregate_map_: Optional[dict] = None

        self.label_names_: Optional[List[str]] = None

    #-----------
    # Utilities
    #-----------

    def _clean_texts(self, X_text) -> np.ndarray:
        X_text = np.asarray(X_text, dtype=object)

        def to_text(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return ""
            s = str(x)
            return "" if s.strip().lower() == "nan" else s

        if self.clean_fn is None:
            return np.asarray([to_text(x) for x in X_text], dtype=object)

        return np.asarray([self.clean_fn(to_text(x)) for x in X_text], dtype=object)
    
    def _coerce_y_str(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=object).ravel()

        def norm(v):
            if v is None:
                return None
            if isinstance(v, float) and np.isnan(v):
                return None
            s = str(v).strip()
            if not s or s.lower() == "nan":
                return None
            return s

        out = np.asarray([norm(v) for v in y], dtype=object)

        # I strongly recommend: fail fast instead of dropping
        if np.any(pd.isna(out)):
            bad = int(np.sum(pd.isna(out)))
            raise ValueError(f"Found {bad} missing/invalid labels in y.")
        return out.astype(str)

    def _default_cfg(self) -> TfidfUnionConfig:
        return TfidfUnionConfig()

    #----------
    # Tuning
    #----------

    def tune(self, X_train_text, y_train, X_val_text, y_val) -> dict:
        X_train_text = self._clean_texts(X_train_text)
        X_val_text = self._clean_texts(X_val_text)
        y_train = self._coerce_y_str(y_train)
        y_val   = self._coerce_y_str(y_val)

        def objective(trial):
            cfg = TfidfUnionConfig(
                ngram_max=trial.suggest_int("ngram_max", 1, 3),
                min_df=trial.suggest_int("min_df", 1, 5),
                max_df=trial.suggest_float("max_df", 0.85, 0.99),
                sublinear_tf=True,
                word_max_features=trial.suggest_int("word_max_features", 80_000, 300_000, step=55_000),
                char_max_features=trial.suggest_int("char_max_features", 60_000, 220_000, step=40_000),
            )
            C = trial.suggest_float("C", 0.2, 5.0, log=True)

            feats = build_tfidf_union(cfg)
            Xtr = feats.fit_transform(X_train_text)
            Xva = feats.transform(X_val_text)

            clf = LinearSVC(C=C, class_weight="balanced")
            clf.fit(Xtr, y_train)
            pred = clf.predict(Xva)
            return f1_score(y_val, pred, average="macro")

        study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_startup_trials=5))
        study.optimize(
            objective,
            n_trials=self.OPTUNA_TRIALS,
            timeout=self.OPTUNA_TIMEOUT_SEC,
            show_progress_bar=False,
        )

        self.best_params = dict(study.best_params)
        self.val_macro_f1 = float(study.best_value)
        return self.best_params

    def _params_to_cfg_and_C(self, params: Optional[dict]) -> tuple[TfidfUnionConfig, float]:
        if not params:
            cfg = self._default_cfg()
            return cfg, 1.0
        cfg = TfidfUnionConfig(
            ngram_max=int(params["ngram_max"]),
            min_df=int(params["min_df"]),
            max_df=float(params["max_df"]),
            sublinear_tf=True,
            word_max_features=int(params["word_max_features"]),
            char_max_features=int(params["char_max_features"]),
        )
        C = float(params["C"])
        return cfg, C

    #------------------------------
    # Hierarchical aggregation map
    #------------------------------

    def _build_aggregate_map_from_centroids(self, X_train_text, y_train, cfg: TfidfUnionConfig) -> dict:
        feats = build_tfidf_union(cfg)
        X = feats.fit_transform(X_train_text)

        labels = list(np.unique(y_train))
        supports = {l: int(np.sum(y_train == l)) for l in labels}

        kept = [l for l in labels if supports[l] >= self.MIN_LABEL_SUPPORT_FOR_CLUSTER]
        rare = [l for l in labels if supports[l] < self.MIN_LABEL_SUPPORT_FOR_CLUSTER]

        aggregate_map: Dict[str, str] = {}

        if len(kept) > 1:
            centroids = []
            for l in kept:
                idx = np.where(y_train == l)[0]
                c = X[idx].mean(axis=0)
                centroids.append(np.asarray(c).ravel().astype(np.float32))

            Cmat = np.vstack(centroids)
            D = cosine_distances(Cmat, Cmat)

            n_clusters = min(self.N_AGGREGATE_CLUSTERS, len(kept))

            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="precomputed",
                    linkage="average",
                )
            except TypeError:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",
                    linkage="average",
                )

            cluster_ids = clustering.fit_predict(D)
            for l, cid in zip(kept, cluster_ids):
                aggregate_map[str(l)] = f"C{int(cid)}"
        elif len(kept) == 1:
            aggregate_map[str(kept[0])] = "C0"

        start = 0 if not aggregate_map else (max(int(v[1:]) for v in aggregate_map.values()) + 1)
        for i, l in enumerate(rare):
            aggregate_map[str(l)] = f"C{start + i}"

        return aggregate_map

    def _apply_aggregate_map(self, y, aggregate_map: dict) -> np.ndarray:
        y = np.asarray(y, dtype=object)
        return np.asarray([aggregate_map.get(str(lbl), "C999") for lbl in y], dtype=object)

    #-----------------------------
    # Fit hierarchical classifier
    #-----------------------------

    def _fit_hierarchical(self, X_text, y_granular, y_aggregate, cfg: TfidfUnionConfig, C: float):
        feats = build_tfidf_union(cfg)
        X = feats.fit_transform(X_text)

        y_aggregate = np.asarray([str(a) for a in y_aggregate], dtype=object)
        uniq_agg = np.unique(y_aggregate)

        if len(uniq_agg) < 2:
            only = str(uniq_agg[0]) if len(uniq_agg) == 1 else "C0"
            clf_agg = DummyClassifier(strategy="constant", constant=only)
            clf_agg.fit(X, y_aggregate)
        else:
            clf_agg = LinearSVC(C=C, class_weight="balanced")
            clf_agg.fit(X, y_aggregate)

        le_aggregate = LabelEncoder().fit(y_aggregate)

        granular_clfs: Dict[str, Any] = {}
        fallback: Dict[str, str] = {}

        for c in sorted(set(y_aggregate), key=str):
            idx = np.where(y_aggregate == c)[0]
            y_f = np.asarray([str(v) for v in y_granular[idx]], dtype=object)
            uniq = np.unique(y_f)

            if len(uniq) == 1:
                fallback[str(c)] = str(uniq[0])
                continue

            if len(idx) < self.MIN_BUCKET_SIZE:
                vals, cnts = np.unique(y_f, return_counts=True)
                fallback[str(c)] = str(vals[np.argmax(cnts)])
                continue

            clf_f = LinearSVC(C=C, class_weight="balanced")
            clf_f.fit(X[idx], y_f)
            granular_clfs[str(c)] = clf_f

        return feats, HierarchicalTopicClassifier(clf_agg, granular_clfs, fallback), le_aggregate

    #------------
    # Public API
    #------------

    def fit(
        self,
        X_train_text,
        y_train,
        X_val_text=None,
        y_val=None,
        tune: bool = True,
        refit_on_train_val: bool = True,
        params: Optional[dict] = None,
    ):
        X_train_text = self._clean_texts(X_train_text)
        y_train = self._coerce_y_str(y_train)

        # tune only if params are not given
        if params is None and tune and (X_val_text is not None) and (y_val is not None):
            X_val_text = self._clean_texts(X_val_text)
            y_val = self._coerce_y_str(y_val)
            self.tune(X_train_text, y_train, X_val_text, y_val)

        # precedence: explicit params > tuned params > defaults
        chosen_params = dict(params) if params is not None else self.best_params
        if params is not None:
            self.best_params = dict(params)

        cfg, C = self._params_to_cfg_and_C(chosen_params)

        aggregate_map = self._build_aggregate_map_from_centroids(X_train_text, y_train, cfg)
        self.aggregate_map_ = aggregate_map

        if refit_on_train_val and (X_val_text is not None) and (y_val is not None):
            X_val_text = self._clean_texts(X_val_text)
            y_val = self._coerce_y_str(y_val)

            X_tv = np.concatenate([X_train_text, X_val_text])
            y_tv = np.concatenate([y_train, y_val])
            y_tv_agg = self._apply_aggregate_map(y_tv, aggregate_map)

            feats, hier, le_agg = self._fit_hierarchical(X_tv, y_tv, y_tv_agg, cfg, C)
            self.le_ = LabelEncoder().fit(y_tv.astype(str))
        else:
            y_train_agg = self._apply_aggregate_map(y_train, aggregate_map)
            feats, hier, le_agg = self._fit_hierarchical(X_train_text, y_train, y_train_agg, cfg, C)
            self.le_ = LabelEncoder().fit(y_train.astype(str))

        self.preprocessor_ = feats
        self.clf_ = hier
        self.le_aggregate_ = le_agg

        return self

    def predict(self, X_text):
        if self.clf_ is None or self.preprocessor_ is None:
            raise RuntimeError("Model not trained/loaded. Call fit() or load() first.")
        X_text = self._clean_texts(X_text)
        X = self.preprocessor_.transform(X_text)
        pred = self.clf_.predict(X)
        return np.asarray([str(p) for p in pred], dtype=object)

    def predict_one(self, text: str):
        return self.predict([text])[0]

    def evaluate(self, X_text, y_true, top_k_confusions: int = 20) -> dict:
        """
        Evaluate on (X_text, y_true).

        Returns a dict with:
        - macro_f1, micro_f1
        - report_text (string)
        - report_dict (raw sklearn dict)
        - report_df (pandas DataFrame, pretty)
        - confusion_matrix (np.ndarray)
        - confusion_df (pandas DataFrame with labels)
        - top_confusions_df (largest off-diagonal confusions)
        """

        y_true = self._coerce_y_str(y_true)
        y_pred = self.predict(X_text).astype(str)

        report_text = classification_report(y_true, y_pred, zero_division=0)
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        macro = float(f1_score(y_true, y_pred, average="macro"))
        micro = float(f1_score(y_true, y_pred, average="micro"))

        report_df = pd.DataFrame(report_dict).T
        overall_rows = [r for r in ["accuracy", "macro avg", "weighted avg"] if r in report_df.index]
        label_rows = [r for r in report_df.index if r not in overall_rows]
        report_df = report_df.loc[label_rows + overall_rows]

        labels_order = sorted(set(y_true), key=str)
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)

        # Top confusions (off-diagonal)
        cm_off = cm_df.copy()
        np.fill_diagonal(cm_off.values, 0)
        top_confusions_df = (
            cm_off.stack()
                .sort_values(ascending=False)
                .head(int(top_k_confusions))
                .rename("count")
                .reset_index()
                .rename(columns={"level_0": "true", "level_1": "pred"})
        )
        top_confusions_df = top_confusions_df[top_confusions_df["count"] > 0].copy()

        return {
            "macro_f1": macro,
            "micro_f1": micro,
            "report_text": report_text,
            "report_dict": report_dict,
            "report_df": report_df,
            "confusion_matrix": cm,
            "confusion_df": cm_df,
            "labels_order": labels_order,
            "top_confusions_df": top_confusions_df,
        }

    #---------------
    # Model's state
    #---------------

    def get_state(self) -> dict:
        if self.preprocessor_ is None or self.clf_ is None:
            raise RuntimeError("Model not fitted.")
        return {
            "preprocessor": self.preprocessor_,
            "clf": self.clf_,
            "le": self.le_,
            "le_aggregate": self.le_aggregate_,
            "aggregate_map": self.aggregate_map_ or {},
            "best_params": self.best_params,
            "val_macro_f1": self.val_macro_f1,
            "test_macro_f1": self.test_macro_f1,
            "meta": {
                "label_names": self.label_names_,
            },
        }

    # @classmethod
    # def from_state(cls, state: dict) -> HierarchicalSVMModel:
    #     m = cls()
    #     m.preprocessor_ = state["preprocessor"]
    #     m.clf_ = state["clf"]
    #     m.le_ = state.get("le")
    #     m.le_aggregate_ = state.get("le_aggregate")
    #     m.aggregate_map_ = state.get("aggregate_map", {})
    #     m.best_params = state.get("best_params")
    #     m.val_macro_f1 = state.get("val_macro_f1")
    #     m.test_macro_f1 = state.get("test_macro_f1")

    #     meta = state.get("meta", {})
    #     m.label_is_int_ = meta.get("label_is_int")
    #     m.label_names_ = meta.get("label_names")
    #     return m
    
    @classmethod
    def from_state(cls, state: dict, *, clean_fn=None) -> "HierarchicalSVMModel":
        m = cls(clean_fn=clean_fn)
        m.preprocessor_ = state["preprocessor"]
        m.clf_ = state["clf"]
        m.le_ = state.get("le")
        m.le_aggregate_ = state.get("le_aggregate")
        m.aggregate_map_ = state.get("aggregate_map", {})
        m.best_params = state.get("best_params")
        m.val_macro_f1 = state.get("val_macro_f1")
        m.test_macro_f1 = state.get("test_macro_f1")

        meta = state.get("meta", {})
        m.label_names_ = meta.get("label_names")
        return m