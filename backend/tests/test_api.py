from backend.scripts.train_model import train


def test_api_scores_transaction():
    train()

    from fastapi.testclient import TestClient

    from backend.app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/score",
        json={
            "transaction_id": "TX-TEST-001",
            "amount": 1300,
            "merchant_category": "cash_advance",
            "channel": "online",
            "international": 1,
            "merchant_risk_score": 0.91,
            "device_trust_score": 0.16,
            "failed_login_24h": 4,
            "customer_avg_amount": 70,
            "customer_std_amount": 35,
            "is_new_merchant": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transaction_id"] == "TX-TEST-001"
    assert 0 <= payload["fraud_probability"] <= 1
    assert 0 <= payload["risk_score"] <= 1000
    assert payload["explanations"]


def test_metrics_endpoint_contains_models():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    client = TestClient(app)
    response = client.get("/api/ml/metrics")

    assert response.status_code == 200
    models = response.json()["models_trained"]
    assert any("RandomForestClassifier" in item["model"] for item in models)
    assert any("XGBoost" in item["model"] or "GradientBoosting" in item["model"] for item in models)

