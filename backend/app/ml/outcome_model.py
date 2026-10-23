from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import artifact_path, build_feature_matrix, transaction_from_payload


MODEL_ARTIFACT = artifact_path("fraud_risk_model.joblib")
METRICS_ARTIFACT = artifact_path("metrics.json")
TRANSACTIONS_ARTIFACT = artifact_path("transactions.json")


FEATURE_DESCRIPTIONS = {
    "amount": "Transaction value in account currency.",
    "amount_to_customer_avg": "How large this transaction is relative to the customer's usual spend.",
    "amount_zscore": "Standardized unusual amount score for this customer.",
    "merchant_risk_score": "Historical risk profile assigned to the merchant and category.",
    "device_trust_score": "Device reputation score where lower values indicate more risk.",
    "international": "Whether the transaction is cross-border.",
    "distance_from_home_km": "Approximate distance from the customer's normal spending area.",
    "failed_login_24h": "Failed login attempts before the transaction.",
    "spending_velocity_1h": "Recent spending intensity in the last hour.",
    "spending_velocity_24h": "Recent spending intensity in the last day.",
    "is_new_merchant": "Whether this customer has no established history with the merchant.",
}


class FraudModelService:
    def __init__(self, model_path: Path = MODEL_ARTIFACT):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model artifact at {model_path}. Run: python -m backend.scripts.train_model"
            )
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.model_name = artifact["model_name"]
        self.feature_names = artifact["feature_names"]
        self.threshold = float(artifact["threshold"])
        self.training_profile = artifact.get("training_profile", {})
        self.feature_importance = artifact.get("feature_importance", {})

    def predict_frame(self, transactions: pd.DataFrame) -> list[dict[str, Any]]:
        matrix = build_feature_matrix(transactions, self.feature_names)
        probabilities = self.model.predict_proba(matrix)[:, 1]
        results = []
        for idx, probability in enumerate(probabilities):
            row = transactions.iloc[idx].to_dict()
            risk_score = int(round(probability * 1000))
            flagged = bool(probability >= self.threshold)
            results.append(
                {
                    "transaction_id": row.get("transaction_id", f"TX-LIVE-{idx}"),
                    "customer_id": int(row.get("customer_id", 0)),
                    "merchant_id": int(row.get("merchant_id", 0)),
                    "amount": float(row.get("amount", 0)),
                    "merchant_category": row.get("merchant_category", "unknown"),
                    "channel": row.get("channel", "unknown"),
                    "fraud_probability": round(float(probability), 4),
                    "risk_score": risk_score,
                    "risk_band": risk_band(probability),
                    "flagged": flagged,
                    "decision_threshold": self.threshold,
                    "estimated_loss_exposure": round(float(row.get("amount", 0)) * float(probability), 2),
                    "recommended_action": recommended_action(probability, self.threshold),
                    "explanations": self.explain(matrix.iloc[idx]),
                }
            )
        return results

    def predict_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self.training_profile.get("defaults", {})
        frame = transaction_from_payload(payload, defaults=defaults)
        return self.predict_frame(frame)[0]

    def explain(self, feature_row: pd.Series, limit: int = 6) -> list[dict[str, Any]]:
        shap_explanations = self._shap_explain(feature_row, limit)
        if shap_explanations:
            return shap_explanations

        return self._importance_explain(feature_row, limit)

    def _shap_explain(self, feature_row: pd.Series, limit: int) -> list[dict[str, Any]]:
        try:
            import shap

            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(pd.DataFrame([feature_row], columns=self.feature_names))
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                array = np.asarray(shap_values)
                values = array[0, :, 1] if array.ndim == 3 else array[0]
        except Exception:
            return []

        explanations = []
        for feature, value in feature_row.items():
            idx = self.feature_names.index(feature)
            contribution = float(values[idx])
            explanations.append(
                {
                    "feature": feature,
                    "label": humanize_feature(feature),
                    "value": round(float(value), 4),
                    "impact": round(contribution, 4),
                    "direction": "raises risk" if contribution >= 0 else "lowers risk",
                }
            )
        explanations.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return explanations[:limit]

    def _importance_explain(self, feature_row: pd.Series, limit: int) -> list[dict[str, Any]]:
        medians = self.training_profile.get("feature_medians", {})
        stds = self.training_profile.get("feature_stds", {})
        explanations = []
        for feature, value in feature_row.items():
            importance = float(self.feature_importance.get(feature, 0.0))
            if importance <= 0:
                continue
            median = float(medians.get(feature, 0.0))
            std = float(stds.get(feature, 1.0)) or 1.0
            contribution = ((float(value) - median) / std) * importance
            explanations.append(
                {
                    "feature": feature,
                    "label": humanize_feature(feature),
                    "value": round(float(value), 4),
                    "impact": round(float(contribution), 4),
                    "direction": "raises risk" if contribution >= 0 else "lowers risk",
                }
            )
        explanations.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return explanations[:limit]


def humanize_feature(feature: str) -> str:
    cleaned = feature.replace("merchant_category_", "category: ").replace("channel_", "channel: ")
    return cleaned.replace("_", " ").title()


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "critical"
    if probability >= 0.45:
        return "high"
    if probability >= 0.22:
        return "medium"
    return "low"


def recommended_action(probability: float, threshold: float) -> str:
    if probability >= max(0.70, threshold):
        return "Block and escalate to fraud operations"
    if probability >= threshold:
        return "Step-up authentication and analyst review"
    if probability >= threshold * 0.65:
        return "Monitor and enrich with customer confirmation"
    return "Approve with standard monitoring"


@lru_cache(maxsize=1)
def get_model_service() -> FraudModelService:
    return FraudModelService()


def load_metrics() -> dict[str, Any]:
    if not METRICS_ARTIFACT.exists():
        return {}
    return json.loads(METRICS_ARTIFACT.read_text(encoding="utf-8"))


def load_transactions(limit: int = 400) -> list[dict[str, Any]]:
    if not TRANSACTIONS_ARTIFACT.exists():
        return []
    data = json.loads(TRANSACTIONS_ARTIFACT.read_text(encoding="utf-8"))
    return data[:limit]


def score_saved_transactions(limit: int = 200) -> list[dict[str, Any]]:
    service = get_model_service()
    transactions = pd.DataFrame(load_transactions(limit))
    if transactions.empty:
        return []
    return service.predict_frame(transactions)

