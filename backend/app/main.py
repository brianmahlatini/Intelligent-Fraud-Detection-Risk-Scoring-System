from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.ml.features import CHANNELS, MERCHANT_CATEGORIES
from backend.app.ml.outcome_model import (
    FEATURE_DESCRIPTIONS,
    get_model_service,
    load_metrics,
    load_transactions,
    score_saved_transactions,
)


class TransactionRequest(BaseModel):
    transaction_id: str = Field(default="TX-LIVE-001")
    customer_id: int = Field(default=10991)
    merchant_id: int = Field(default=2201)
    amount: float = Field(default=420.0, ge=0)
    merchant_category: str = Field(default="digital_goods")
    channel: str = Field(default="online")
    card_present: int = Field(default=0, ge=0, le=1)
    international: int = Field(default=0, ge=0, le=1)
    distance_from_home_km: float = Field(default=130.0, ge=0)
    merchant_risk_score: float = Field(default=0.72, ge=0, le=1)
    device_trust_score: float = Field(default=0.34, ge=0, le=1)
    account_age_days: int = Field(default=420, ge=0)
    failed_login_24h: int = Field(default=2, ge=0)
    customer_chargeback_rate: float = Field(default=0.04, ge=0, le=1)
    customer_avg_amount: float = Field(default=75.0, ge=0)
    customer_std_amount: float = Field(default=45.0, ge=0)
    customer_txn_count_24h: int = Field(default=4, ge=0)
    spending_velocity_1h: float = Field(default=610.0, ge=0)
    spending_velocity_24h: float = Field(default=980.0, ge=0)
    is_new_merchant: int = Field(default=1, ge=0, le=1)


app = FastAPI(
    title="Intelligent Fraud Detection & Risk Scoring API",
    description="Scores financial transactions with supervised ML and cost-aware fraud thresholds.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    service = get_model_service()
    return {
        "status": "ok",
        "model_name": service.model_name,
        "decision_threshold": service.threshold,
    }


@app.get("/api/ml/metrics")
def metrics() -> dict[str, Any]:
    return load_metrics()


@app.get("/api/ml/features")
def features() -> dict[str, Any]:
    metrics_payload = load_metrics()
    return {
        "target": "is_fraud",
        "prediction": "Probability that a transaction is fraudulent plus a 0-1000 risk score.",
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "feature_groups": metrics_payload.get("feature_groups", {}),
        "merchant_categories": MERCHANT_CATEGORIES,
        "channels": CHANNELS,
    }


@app.get("/api/transactions")
def transactions(limit: int = 120) -> dict[str, Any]:
    service = get_model_service()
    scored = score_saved_transactions(limit=limit)
    raw_lookup = {item["transaction_id"]: item for item in load_transactions(limit)}
    enriched = []
    for item in scored:
        raw = raw_lookup.get(item["transaction_id"], {})
        enriched.append(
            {
                **item,
                "timestamp": raw.get("timestamp"),
                "investigation_status": raw.get("investigation_status", "new"),
                "is_fraud": raw.get("is_fraud"),
            }
        )
    summary = {
        "count": len(enriched),
        "flagged": sum(1 for item in enriched if item["flagged"]),
        "critical": sum(1 for item in enriched if item["risk_band"] == "critical"),
        "estimated_loss_exposure": round(sum(item["estimated_loss_exposure"] for item in enriched), 2),
        "model_name": service.model_name,
        "decision_threshold": service.threshold,
    }
    return {"summary": summary, "transactions": enriched}


@app.post("/api/score")
def score_transaction(payload: TransactionRequest) -> dict[str, Any]:
    return get_model_service().predict_payload(payload.model_dump())

