from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RANDOM_SEED = 42

MERCHANT_CATEGORIES = [
    "grocery",
    "fuel",
    "electronics",
    "travel",
    "gaming",
    "jewelry",
    "digital_goods",
    "restaurants",
    "cash_advance",
    "pharmacy",
]

CHANNELS = ["pos", "online", "mobile", "atm"]
INVESTIGATION_STATUSES = ["new", "in_review", "cleared", "escalated"]

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "day_of_week",
    "is_weekend",
    "card_present",
    "international",
    "distance_from_home_km",
    "merchant_risk_score",
    "device_trust_score",
    "account_age_days",
    "failed_login_24h",
    "customer_avg_amount",
    "customer_std_amount",
    "customer_txn_count_24h",
    "spending_velocity_1h",
    "spending_velocity_24h",
    "amount_to_customer_avg",
    "amount_zscore",
    "is_new_merchant",
    "customer_chargeback_rate",
]

CATEGORICAL_FEATURES = ["merchant_category", "channel"]


@dataclass(frozen=True)
class DatasetBundle:
    transactions: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series


def _rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_transactions(n_rows: int = 6500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate deterministic card transaction data with realistic fraud signals."""
    rng = _rng(seed)
    customer_count = 850
    merchant_count = 180

    customer_ids = rng.integers(10000, 10000 + customer_count, n_rows)
    merchant_ids = rng.integers(2000, 2000 + merchant_count, n_rows)
    categories = rng.choice(
        MERCHANT_CATEGORIES,
        n_rows,
        p=[0.18, 0.11, 0.10, 0.08, 0.09, 0.05, 0.09, 0.14, 0.04, 0.12],
    )
    channels = rng.choice(CHANNELS, n_rows, p=[0.48, 0.30, 0.17, 0.05])
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 150 * 24 * 60, n_rows), unit="m"
    )

    customer_income_band = rng.choice([0, 1, 2, 3], n_rows, p=[0.22, 0.36, 0.29, 0.13])
    base_amount = rng.lognormal(mean=3.25 + customer_income_band * 0.22, sigma=0.82, size=n_rows)
    category_multiplier = pd.Series(categories).map(
        {
            "grocery": 0.72,
            "fuel": 0.85,
            "electronics": 2.25,
            "travel": 2.75,
            "gaming": 1.18,
            "jewelry": 3.8,
            "digital_goods": 1.35,
            "restaurants": 0.95,
            "cash_advance": 2.1,
            "pharmacy": 0.62,
        }
    ).to_numpy()
    amount = np.round(np.clip(base_amount * category_multiplier, 2, 4500), 2)

    risk_by_category = pd.Series(categories).map(
        {
            "grocery": 0.12,
            "fuel": 0.20,
            "electronics": 0.54,
            "travel": 0.50,
            "gaming": 0.60,
            "jewelry": 0.74,
            "digital_goods": 0.69,
            "restaurants": 0.18,
            "cash_advance": 0.82,
            "pharmacy": 0.14,
        }
    ).to_numpy()
    merchant_risk_score = np.clip(risk_by_category + rng.normal(0, 0.08, n_rows), 0.03, 0.98)

    card_present = (channels == "pos").astype(int)
    international = rng.binomial(1, np.where(np.isin(categories, ["travel", "digital_goods"]), 0.18, 0.05))
    distance = np.round(rng.gamma(2.0, 22.0, n_rows) + international * rng.gamma(3.5, 95.0, n_rows), 2)
    device_trust_score = np.clip(rng.beta(8, 2, n_rows) - international * 0.18 - (channels == "online") * 0.08, 0.02, 1)
    account_age_days = rng.integers(4, 3650, n_rows)
    failed_login_24h = rng.poisson(np.where((channels == "online") | (channels == "mobile"), 0.55, 0.08), n_rows)
    customer_chargeback_rate = np.clip(rng.beta(1.2, 40, n_rows) + merchant_risk_score * 0.025, 0, 0.42)

    df = pd.DataFrame(
        {
            "transaction_id": [f"TX-{i:06d}" for i in range(1, n_rows + 1)],
            "customer_id": customer_ids,
            "merchant_id": merchant_ids,
            "timestamp": timestamps,
            "amount": amount,
            "merchant_category": categories,
            "channel": channels,
            "card_present": card_present,
            "international": international,
            "distance_from_home_km": distance,
            "merchant_risk_score": np.round(merchant_risk_score, 4),
            "device_trust_score": np.round(device_trust_score, 4),
            "account_age_days": account_age_days,
            "failed_login_24h": failed_login_24h,
            "customer_chargeback_rate": np.round(customer_chargeback_rate, 4),
        }
    ).sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    customer_stats = df.groupby("customer_id")["amount"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "customer_avg_amount", "std": "customer_std_amount", "count": "customer_lifetime_txns"}
    )
    df = df.join(customer_stats, on="customer_id")
    df["customer_std_amount"] = df["customer_std_amount"].fillna(df["amount"].std())
    df["customer_txn_count_24h"] = rng.poisson(
        1.4 + (df["channel"].isin(["online", "mobile"]).astype(float) * 1.3), n_rows
    )
    df["spending_velocity_1h"] = np.round(rng.gamma(1.3, 18, n_rows) + df["customer_txn_count_24h"] * rng.uniform(2, 12, n_rows), 2)
    df["spending_velocity_24h"] = np.round(df["spending_velocity_1h"] + rng.gamma(2.1, 35, n_rows), 2)
    df["amount_to_customer_avg"] = np.round(df["amount"] / (df["customer_avg_amount"] + 1), 4)
    df["amount_zscore"] = np.round((df["amount"] - df["customer_avg_amount"]) / (df["customer_std_amount"] + 1), 4)
    df["is_new_merchant"] = rng.binomial(1, np.clip(0.10 + df["merchant_risk_score"] * 0.22, 0, 0.44))
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    logit = (
        -5.1
        + 0.85 * df["international"]
        + 1.05 * df["is_new_merchant"]
        + 1.65 * df["merchant_risk_score"]
        + 0.70 * (1 - df["device_trust_score"])
        + 0.42 * np.log1p(df["failed_login_24h"])
        + 0.55 * (df["amount_to_customer_avg"] > 2.6).astype(float)
        + 0.50 * (df["amount_zscore"] > 2.0).astype(float)
        + 0.36 * (df["distance_from_home_km"] > 180).astype(float)
        + 0.32 * df["channel"].isin(["online", "atm"]).astype(float)
        + 0.28 * df["merchant_category"].isin(["cash_advance", "digital_goods", "jewelry"]).astype(float)
    )
    probability = 1 / (1 + np.exp(-logit))
    df["is_fraud"] = rng.binomial(1, probability)
    df["investigation_status"] = np.where(
        df["is_fraud"].eq(1),
        rng.choice(["new", "in_review", "escalated"], n_rows, p=[0.44, 0.34, 0.22]),
        rng.choice(INVESTIGATION_STATUSES, n_rows, p=[0.24, 0.20, 0.53, 0.03]),
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def build_feature_matrix(transactions: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> pd.DataFrame:
    frame = transactions.copy()
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["hour"] = frame.get("hour", frame["timestamp"].dt.hour)
        frame["day_of_week"] = frame.get("day_of_week", frame["timestamp"].dt.dayofweek)
        frame["is_weekend"] = frame.get("is_weekend", frame["day_of_week"].isin([5, 6]).astype(int))

    for col in NUMERIC_FEATURES:
        if col not in frame:
            frame[col] = 0
    encoded = pd.get_dummies(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, dtype=float)
    encoded = encoded.astype(float)

    if feature_columns is not None:
        encoded = encoded.reindex(columns=list(feature_columns), fill_value=0.0)
    return encoded


def make_dataset(n_rows: int = 6500, seed: int = RANDOM_SEED) -> DatasetBundle:
    transactions = generate_transactions(n_rows=n_rows, seed=seed)
    features = build_feature_matrix(transactions)
    target = transactions["is_fraud"].astype(int)
    return DatasetBundle(transactions=transactions, features=features, target=target)


def transaction_from_payload(payload: dict, defaults: dict | None = None) -> pd.DataFrame:
    defaults = defaults or {}
    now = pd.Timestamp(payload.get("timestamp") or pd.Timestamp.utcnow())
    amount = float(payload.get("amount", defaults.get("amount", 100.0)))
    avg = float(payload.get("customer_avg_amount", defaults.get("customer_avg_amount", 75.0)))
    std = float(payload.get("customer_std_amount", defaults.get("customer_std_amount", 45.0)))

    row = {
        "transaction_id": payload.get("transaction_id", "TX-LIVE"),
        "customer_id": int(payload.get("customer_id", defaults.get("customer_id", 99999))),
        "merchant_id": int(payload.get("merchant_id", defaults.get("merchant_id", 2999))),
        "timestamp": now,
        "amount": amount,
        "merchant_category": payload.get("merchant_category", defaults.get("merchant_category", "digital_goods")),
        "channel": payload.get("channel", defaults.get("channel", "online")),
        "card_present": int(payload.get("card_present", defaults.get("card_present", 0))),
        "international": int(payload.get("international", defaults.get("international", 0))),
        "distance_from_home_km": float(payload.get("distance_from_home_km", defaults.get("distance_from_home_km", 20.0))),
        "merchant_risk_score": float(payload.get("merchant_risk_score", defaults.get("merchant_risk_score", 0.5))),
        "device_trust_score": float(payload.get("device_trust_score", defaults.get("device_trust_score", 0.75))),
        "account_age_days": int(payload.get("account_age_days", defaults.get("account_age_days", 600))),
        "failed_login_24h": int(payload.get("failed_login_24h", defaults.get("failed_login_24h", 0))),
        "customer_chargeback_rate": float(payload.get("customer_chargeback_rate", defaults.get("customer_chargeback_rate", 0.02))),
        "customer_avg_amount": avg,
        "customer_std_amount": std,
        "customer_txn_count_24h": int(payload.get("customer_txn_count_24h", defaults.get("customer_txn_count_24h", 1))),
        "spending_velocity_1h": float(payload.get("spending_velocity_1h", defaults.get("spending_velocity_1h", amount))),
        "spending_velocity_24h": float(payload.get("spending_velocity_24h", defaults.get("spending_velocity_24h", amount * 1.4))),
        "is_new_merchant": int(payload.get("is_new_merchant", defaults.get("is_new_merchant", 0))),
    }
    row["amount_to_customer_avg"] = amount / (avg + 1)
    row["amount_zscore"] = (amount - avg) / (std + 1)
    row["hour"] = now.hour
    row["day_of_week"] = now.dayofweek
    row["is_weekend"] = int(row["day_of_week"] in [5, 6])
    return pd.DataFrame([row])


def artifact_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / Path(*parts)

