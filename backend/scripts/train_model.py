from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from backend.app.ml.features import artifact_path, make_dataset

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


def pr_auc(y_true: pd.Series, probabilities: np.ndarray) -> float:
    return float(average_precision_score(y_true, probabilities))


def false_positive_rate(y_true: pd.Series, predictions: np.ndarray) -> float:
    tn, fp, _fn, _tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return float(fp / max(fp + tn, 1))


def tune_threshold(y_true: pd.Series, probabilities: np.ndarray, amounts: pd.Series) -> dict[str, float]:
    fraud_loss = amounts.to_numpy() * y_true.to_numpy()
    baseline_loss = float(fraud_loss.sum())
    best: dict[str, float] | None = None
    for threshold in np.linspace(0.05, 0.90, 86):
        predictions = (probabilities >= threshold).astype(int)
        false_negative_loss = float(((predictions == 0) & (y_true.to_numpy() == 1)) @ amounts.to_numpy())
        false_positive_cost = float(((predictions == 1) & (y_true.to_numpy() == 0)).sum() * 15.0)
        estimated_loss = false_negative_loss + false_positive_cost
        candidate = {
            "threshold": round(float(threshold), 3),
            "estimated_loss": round(estimated_loss, 2),
            "losses_prevented": round(baseline_loss - estimated_loss, 2),
        }
        if best is None or candidate["estimated_loss"] < best["estimated_loss"]:
            best = candidate
    return best or {"threshold": 0.5, "estimated_loss": baseline_loss, "losses_prevented": 0.0}


def evaluate_model(name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series, amounts: pd.Series) -> dict[str, Any]:
    probabilities = model.predict_proba(X_test)[:, 1]
    threshold_info = tune_threshold(y_test, probabilities, amounts)
    predictions = (probabilities >= threshold_info["threshold"]).astype(int)
    return {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(pr_auc(y_test, probabilities), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "false_positive_rate": round(false_positive_rate(y_test, predictions), 4),
        **threshold_info,
    }


def model_feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_[0], dtype=float))
    else:
        values = np.ones(len(feature_names), dtype=float)
    total = float(values.sum()) or 1.0
    return {feature: round(float(value / total), 7) for feature, value in zip(feature_names, values)}


def build_models(y_train: pd.Series) -> dict[str, Any]:
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(label): float(weight) for label, weight in zip(classes, weights)}
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    models: dict[str, Any] = {
        "LogisticRegression baseline": LogisticRegression(max_iter=1200, class_weight="balanced", solver="liblinear"),
        "RandomForestClassifier baseline": RandomForestClassifier(
            n_estimators=260,
            max_depth=11,
            min_samples_leaf=8,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier stronger_model": GradientBoostingClassifier(random_state=42),
    }
    if XGBClassifier is not None:
        models["XGBoost XGBClassifier stronger_model"] = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.055,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        )
    lightgbm_enabled = (
        LGBMClassifier is not None
        and (platform.system() != "Windows" or os.getenv("ENABLE_LIGHTGBM") == "1")
    )
    if lightgbm_enabled:
        models["LightGBM LGBMClassifier stronger_model"] = LGBMClassifier(
            n_estimators=260,
            learning_rate=0.045,
            num_leaves=31,
            random_state=42,
            class_weight=class_weight,
            verbose=-1,
        )
    return models


def train() -> dict[str, Any]:
    bundle = make_dataset(n_rows=7200, seed=42)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        bundle.features,
        bundle.target,
        bundle.transactions.index,
        test_size=0.25,
        random_state=42,
        stratify=bundle.target,
    )

    results = []
    failed_models = []
    if LGBMClassifier is not None and platform.system() == "Windows" and os.getenv("ENABLE_LIGHTGBM") != "1":
        failed_models.append(
            {
                "model": "LightGBM LGBMClassifier stronger_model",
                "status": "skipped",
                "error": "Skipped on Windows because the local LightGBM native wheel raised an access violation. Set ENABLE_LIGHTGBM=1 to force training.",
            }
        )
    fitted_models: dict[str, Any] = {}
    for name, model in build_models(y_train).items():
        try:
            model.fit(X_train, y_train)
            fitted_models[name] = model
            test_amounts = bundle.transactions.loc[test_idx, "amount"].reset_index(drop=True)
            results.append(evaluate_model(name, model, X_test, y_test.reset_index(drop=True), test_amounts))
        except Exception as exc:
            failed_models.append({"model": name, "status": "failed", "error": str(exc)})

    selected = max(results, key=lambda item: (item["losses_prevented"], item["pr_auc"]))
    selected_model = fitted_models[selected["model"]]
    feature_names = list(bundle.features.columns)
    medians = bundle.features.median().to_dict()
    stds = bundle.features.std().replace(0, 1).fillna(1).to_dict()

    artifact = {
        "model": selected_model,
        "model_name": selected["model"],
        "feature_names": feature_names,
        "threshold": selected["threshold"],
        "feature_importance": model_feature_importance(selected_model, feature_names),
        "training_profile": {
            "feature_medians": {key: float(value) for key, value in medians.items()},
            "feature_stds": {key: float(value) for key, value in stds.items()},
            "defaults": {
                "amount": float(bundle.transactions["amount"].median()),
                "customer_avg_amount": float(bundle.transactions["customer_avg_amount"].median()),
                "customer_std_amount": float(bundle.transactions["customer_std_amount"].median()),
                "merchant_risk_score": float(bundle.transactions["merchant_risk_score"].median()),
                "device_trust_score": float(bundle.transactions["device_trust_score"].median()),
            },
        },
    }

    artifact_dir = artifact_path()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path("fraud_risk_model.joblib"))

    sample = bundle.transactions.sort_values("timestamp", ascending=False).head(450).copy()
    sample["timestamp"] = sample["timestamp"].astype(str)
    artifact_path("transactions.json").write_text(sample.to_json(orient="records", indent=2), encoding="utf-8")

    metrics = {
        "dataset": {
            "rows": int(len(bundle.transactions)),
            "fraud_rate": round(float(bundle.target.mean()), 4),
            "target": "is_fraud",
            "positive_class": "fraudulent transaction",
            "class_balance": bundle.target.value_counts().sort_index().astype(int).to_dict(),
        },
        "models_trained": results,
        "failed_models": failed_models,
        "selected_model": selected,
        "feature_count": len(feature_names),
        "top_features": sorted(
            artifact["feature_importance"].items(), key=lambda item: item[1], reverse=True
        )[:12],
        "feature_groups": {
            "transaction": ["amount", "hour", "day_of_week", "channel", "card_present"],
            "customer_behavior": [
                "customer_avg_amount",
                "amount_to_customer_avg",
                "amount_zscore",
                "customer_txn_count_24h",
                "spending_velocity_1h",
                "spending_velocity_24h",
            ],
            "merchant_and_context": [
                "merchant_category",
                "merchant_risk_score",
                "international",
                "distance_from_home_km",
                "is_new_merchant",
            ],
            "security": ["device_trust_score", "failed_login_24h", "customer_chargeback_rate"],
        },
    }
    artifact_path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics["selected_model"], indent=2))
    print("Models trained:")
    for result in metrics["models_trained"]:
        print(
            f"- {result['model']}: PR-AUC={result['pr_auc']}, "
            f"precision={result['precision']}, recall={result['recall']}, F1={result['f1']}, "
            f"FPR={result['false_positive_rate']}, threshold={result['threshold']}"
        )

