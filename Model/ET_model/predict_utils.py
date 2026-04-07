from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# 兼容两种运行方式：
# 1) 作为包模块：python -m Model.ET_model.xxx
# 2) 直接脚本：python Model\ET_model\xxx.py
try:
    from .eyerunn_cluster.cognitive import extract_cognitive_features
except ImportError:  # pragma: no cover - 仅在脚本直接运行时触发
    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    from eyerunn_cluster.cognitive import extract_cognitive_features  # type: ignore[no-redef]


# 兼容：某些 joblib 产物在序列化时记录了 `Model.ET_model...` 的模块路径。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# 兜底映射：当找不到 `cluster_load_mapping.csv` 时使用
_FALLBACK_CLUSTER_LOAD_MAPPING: dict[str, tuple[int, str]] = {
    "0": (2, "低负荷 / 轻量任务型"),
    "1": (2, "低负荷 / 轻量任务型"),
    "2": (3, "中高负荷 / 信息整合型"),
    "3": (4, "高负荷 / 持续专注解题型"),
    "4": (3, "中高负荷 / 信息整合型"),
    "5": (1, "极低负荷 / 轻松浏览型"),
}

DEFAULT_LOAD_LEVEL = 0
DEFAULT_LOAD_LABEL = "未知负荷"


def _sklearn_feature_matrix(
    estimator: Any,
    feats_df: pd.DataFrame,
    *,
    fallback_cols: list[str] | None,
) -> np.ndarray:
    names_inner = getattr(estimator, "feature_names_in_", None)
    if names_inner is not None:
        cols = [str(x) for x in names_inner]
        sub = feats_df.reindex(columns=cols, fill_value=np.nan)
    elif fallback_cols:
        sub = feats_df.reindex(columns=fallback_cols, fill_value=np.nan)
    else:
        sub = feats_df
    if any(not pd.api.types.is_numeric_dtype(sub[c]) for c in sub.columns):
        sub = sub.copy()
        for c in sub.columns:
            if not pd.api.types.is_numeric_dtype(sub[c]):
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub.to_numpy(dtype=np.float64)


def _load_cluster_load_mapping_csv(path: Path) -> dict[str, tuple[int, str]]:
    mapping: dict[str, tuple[int, str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = str(row.get("cluster", "")).strip()
            if c == "":
                continue
            try:
                level = int(str(row.get("relative_load_level", "")).strip())
            except Exception:
                level = DEFAULT_LOAD_LEVEL
            label = str(row.get("relative_load_label", "")).strip()
            if label == "":
                label = DEFAULT_LOAD_LABEL
            mapping[c] = (level, label)
    return mapping


def get_relative_load_for_cluster(
    cluster: str | int,
    *,
    mapping: dict[str, tuple[int, str]] | None = None,
) -> tuple[int, str]:
    m = mapping or _FALLBACK_CLUSTER_LOAD_MAPPING
    return m.get(str(cluster), (DEFAULT_LOAD_LEVEL, DEFAULT_LOAD_LABEL))


@dataclass(frozen=True)
class PredictionResult:
    sample_key: str
    session_id: str
    task_id: str | None
    predicted_cluster: str
    predicted_cluster_encoded: int
    coordinates_2d: tuple[float, float]
    probabilities: dict[str, float]
    relative_load_level: int
    relative_load_label: str


class SessionPredictor:

    def __init__(
        self,
        classifier_model: str | Path,
        pca_model: str | Path,
        features_template: str | Path,
    ):
        self.classifier_model_path = Path(classifier_model)
        self.pca_model_path = Path(pca_model)
        self.features_template_path = Path(features_template)

        if not self.classifier_model_path.exists():
            raise FileNotFoundError(f"分类器模型不存在: {self.classifier_model_path}")
        if not self.pca_model_path.exists():
            raise FileNotFoundError(f"PCA 模型不存在: {self.pca_model_path}")
        if not self.features_template_path.exists():
            raise FileNotFoundError(f"特征模板不存在: {self.features_template_path}")

        self._clf_data: dict[str, Any] | None = None
        self._pca_data: dict[str, Any] | None = None
        self._feat_cols: list[str] | None = None
        self._cluster_load_mapping: dict[str, tuple[int, str]] | None = None

    def _ensure_loaded(self) -> None:
        if self._clf_data is None:
            self._clf_data = joblib.load(self.classifier_model_path)
        if self._pca_data is None:
            self._pca_data = joblib.load(self.pca_model_path)
        if self._feat_cols is None:
            clf_model = self._clf_data["model"]
            model_cols = self._clf_data.get("feature_columns")
            if isinstance(model_cols, list) and model_cols:
                self._feat_cols = [str(c) for c in model_cols]
            else:
                fn_in = getattr(clf_model, "feature_names_in_", None)
                if fn_in is not None:
                    self._feat_cols = [str(x) for x in fn_in]
                else:
                    feats_template = pd.read_csv(self.features_template_path, index_col=0)
                    self._feat_cols = [c for c in feats_template.columns if c != "sample_key"]
        if self._cluster_load_mapping is None:
            mapping_path = self.features_template_path.parent / "cluster_load_mapping.csv"
            loaded = _load_cluster_load_mapping_csv(mapping_path)
            self._cluster_load_mapping = loaded if loaded else _FALLBACK_CLUSTER_LOAD_MAPPING

    def _classifier_predict(
        self,
        feats_row: pd.DataFrame,
    ) -> tuple[str, int, dict[str, float]]:
        """用监督分类器（SVM/XGBoost pipeline）预测簇标签与各类概率。"""
        assert self._clf_data is not None and self._feat_cols is not None
        clf_model = self._clf_data["model"]
        label_encoder: Any = self._clf_data["label_encoder"]
        feat_cols = self._feat_cols

        X_clf = feats_row.reindex(columns=feat_cols, fill_value=np.nan).copy()
        for c in X_clf.columns:
            if not pd.api.types.is_numeric_dtype(X_clf[c]):
                X_clf[c] = pd.to_numeric(X_clf[c], errors="coerce")

        proba_dict: dict[str, float] = {}
        if hasattr(clf_model, "predict_proba"):
            proba_arr = clf_model.predict_proba(X_clf)[0]
            for i, p in enumerate(proba_arr):
                proba_dict[str(label_encoder.classes_[i])] = float(p)
            best_idx = int(np.argmax(proba_arr))
            predicted_cluster = str(label_encoder.inverse_transform(np.array([best_idx]))[0])
        else:
            pred_enc_arr = clf_model.predict(X_clf)
            pred_enc = int(pred_enc_arr[0])
            predicted_cluster = str(label_encoder.inverse_transform(np.array([pred_enc]))[0])
            proba_dict[predicted_cluster] = 1.0

        try:
            encoded_display = int(float(predicted_cluster))
        except (ValueError, TypeError):
            encoded_display = pred_enc if "pred_enc" in dir() else -1
        return predicted_cluster, encoded_display, proba_dict

    def _predict_from_features(
        self,
        feats_row: pd.DataFrame,
    ) -> tuple[str, PredictionResult]:
        assert len(feats_row) == 1
        sample_key = str(feats_row.index[0])

        if "::task=" in sample_key:
            session_id, task_part = sample_key.split("::task=", 1)
            task_id: str | None = task_part or None
        else:
            session_id = sample_key
            task_id = None

        pca_pipeline = self._pca_data["pipeline"]
        pca_model = self._pca_data["pca"]

        default_prefixes = ("fix__", "blink__", "trans__", "task__")
        expected_n = getattr(pca_pipeline, "n_features_in_", None)

        if expected_n is not None and feats_row.shape[1] != int(expected_n):
            cols_subset = [c for c in feats_row.columns if c.startswith(default_prefixes)]
            feats_for_pca = feats_row[cols_subset]
            if feats_for_pca.shape[1] == int(expected_n):
                feats_aligned_pca = feats_for_pca
            else:
                feats_aligned_pca = None
        else:
            feats_aligned_pca = feats_row

        if feats_aligned_pca is None or feats_aligned_pca.empty or feats_aligned_pca.shape[1] == 0:
            x, y = float("nan"), float("nan")
        else:
            try:
                X_pca = _sklearn_feature_matrix(
                    pca_pipeline, feats_aligned_pca,
                    fallback_cols=list(feats_aligned_pca.columns),
                )
                X_processed = pca_pipeline.transform(X_pca)
                coords_2d = pca_model.transform(X_processed)[0]
                x, y = float(coords_2d[0]), float(coords_2d[1])
            except Exception:
                x, y = float("nan"), float("nan")

        # 认知负荷等级由监督分类器决定；PCA 坐标仅用于可视化
        try:
            predicted_cluster, pred_enc_display, proba_dict = self._classifier_predict(feats_row)
        except Exception:
            predicted_cluster = "?"
            pred_enc_display = -1
            proba_dict = {}

        load_level, load_label = get_relative_load_for_cluster(
            predicted_cluster,
            mapping=self._cluster_load_mapping,
        )

        result = PredictionResult(
            sample_key=sample_key,
            session_id=session_id,
            task_id=task_id,
            predicted_cluster=predicted_cluster,
            predicted_cluster_encoded=pred_enc_display,
            coordinates_2d=(x, y),
            probabilities=proba_dict,
            relative_load_level=load_level,
            relative_load_label=load_label,
        )
        return sample_key, result

    def predict(self, session_dir: str | Path) -> list[PredictionResult]:
        self._ensure_loaded()
        session_dir = Path(session_dir)
        if not session_dir.exists():
            raise FileNotFoundError(f"session 目录不存在: {session_dir}")

        feats_new = extract_cognitive_features(session_dir, unit="task")
        if feats_new.empty:
            raise ValueError("未能从该 session 提取到任何特征")

        results: list[PredictionResult] = []
        for idx in feats_new.index:
            feats_row = feats_new.loc[[idx]]
            _, r = self._predict_from_features(feats_row)
            results.append(r)
        return results


def predict_session(
    session_dir: str | Path,
    *,
    classifier_model: str | Path = "outputs_supervised_task/model_svm.joblib",
    pca_model: str | Path = "outputs_task_cluster/pca_model.joblib",
    features_template: str | Path = "outputs_task_cluster/features.csv",
) -> list[PredictionResult]:
    predictor = SessionPredictor(
        classifier_model=classifier_model,
        pca_model=pca_model,
        features_template=features_template,
    )
    return predictor.predict(session_dir)
