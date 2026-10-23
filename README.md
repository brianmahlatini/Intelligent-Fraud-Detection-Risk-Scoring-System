# Intelligent Fraud Detection & Risk Scoring System

A senior-level full-stack machine learning project for detecting fraudulent financial transactions, assigning a 0-1000 risk score, explaining risk drivers, and supporting fraud analyst investigations through a React monitoring dashboard.

1. [brianmahlatini](https://github.com/brianmahlatini)
2. [**Intelligent-Fraud-Detection-Risk-Scoring-System**](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System)

## Repository Navigation

- [**Code**](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System)
- [Issues](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System/issues)
- [Pull requests](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System/pulls)
- [Actions](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System/actions)
- [Projects](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System/projects)
- [Wiki](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System/wiki)
- [Security and quality](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System/security)
- [Insights](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System/pulse)
- [Settings](https://github.com/brianmahlatini/Intelligent-Fraud-Detection-Risk-Scoring-System/settings)

## Features

- Synthetic but realistic transaction dataset generation with account, merchant, device, behavioral, and velocity signals.
- Clear supervised ML target: `is_fraud`.
- Clear input matrix `X` and target vector `y`.
- Class imbalance analysis and class-weighted training.
- Baseline model: `LogisticRegression`.
- Strong supervised models: `RandomForestClassifier`, `GradientBoostingClassifier`, and `XGBoost XGBClassifier`.
- LightGBM integration included; on this Windows environment it is skipped by default because the local native wheel raised an access violation. Set `ENABLE_LIGHTGBM=1` to force training.
- Cost-aware threshold tuning using fraud loss exposure and false-positive review cost.
- Saved model artifacts with `joblib`.
- FastAPI backend loads the saved artifact for runtime predictions.
- SHAP-based explanations when supported, with a feature-importance fallback.
- React TypeScript fraud-monitoring dashboard with transaction scores, flagged cases, explanations, and investigation status.
- Backend tests for feature engineering and API scoring.
- Docker Compose with FastAPI, React, and PostgreSQL service boundaries.

## Tech Stack

**Machine Learning**

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM optional
- SHAP
- joblib

**Backend**

- FastAPI
- Uvicorn
- pytest

**Frontend**

- React
- TypeScript
- Vite
- Lucide React

**Database Readiness**

- PostgreSQL service is included in Docker Compose.
- The demo uses generated artifacts for reproducible local scoring.
- The API boundary is ready to replace artifact-backed transactions with PostgreSQL tables for transactions, customers, merchants, risk decisions, and investigation cases.

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── ml/
│   │       ├── artifacts/
│   │       │   ├── fraud_risk_model.joblib
│   │       │   ├── metrics.json
│   │       │   └── transactions.json
│   │       ├── features.py
│   │       └── outcome_model.py
│   ├── scripts/
│   │   └── train_model.py
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_api.py
│   │   └── test_features.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── types/
│   │   ├── utils/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.mjs
├── infra/
│   └── docker-compose.yml
├── .gitignore
└── README.md
```

## ML Design

### Dataset

The training script generates 7,200 deterministic transactions with a 3.36% fraud rate.

Each transaction includes customer behavior, merchant category, channel, amount, device trust, failed-login activity, international usage, distance from home, spending velocity, chargeback profile, and investigation status.

### Features X

The model uses 34 engineered features, including:

- Transaction signals: `amount`, `hour`, `day_of_week`, `channel`, `card_present`.
- Customer behavior: `customer_avg_amount`, `customer_std_amount`, `amount_to_customer_avg`, `amount_zscore`, `customer_txn_count_24h`.
- Velocity signals: `spending_velocity_1h`, `spending_velocity_24h`.
- Merchant and context: `merchant_category`, `merchant_risk_score`, `international`, `distance_from_home_km`, `is_new_merchant`.
- Security signals: `device_trust_score`, `failed_login_24h`, `customer_chargeback_rate`.

### Target y

`is_fraud = 1` means the transaction is fraudulent.

`is_fraud = 0` means the transaction is legitimate.

### Threshold Tuning

The selected threshold minimizes estimated business loss:

```text
estimated_loss = missed_fraud_amount + false_positive_review_count * 15
```

This avoids blindly using a `0.50` threshold and aligns the model with fraud operations cost.

## Latest Training Metrics

Dataset:

- Rows: `7200`
- Fraud rate: `0.0336`
- Class balance: `6958` legitimate, `242` fraud

| Model | ROC AUC | PR-AUC | Precision | Recall | F1 | FPR | Threshold | Losses Prevented |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LogisticRegression baseline | 0.7190 | 0.1682 | 0.1391 | 0.3500 | 0.1991 | 0.0747 | 0.750 | $1,929.21 |
| RandomForestClassifier baseline | 0.7042 | 0.1828 | 0.2179 | 0.2833 | 0.2464 | 0.0351 | 0.400 | $3,226.16 |
| GradientBoostingClassifier stronger_model | 0.7144 | 0.1147 | 0.1544 | 0.3500 | 0.2143 | 0.0661 | 0.070 | $2,202.75 |
| XGBoost XGBClassifier stronger_model | 0.6930 | 0.1076 | 0.1818 | 0.2000 | 0.1905 | 0.0310 | 0.610 | $2,468.47 |

Selected runtime model:

```text
RandomForestClassifier baseline
```

Reason: it produced the best cost-aware loss-prevention score in the latest deterministic training run.

LightGBM status:

```text
Skipped on Windows by default because the local LightGBM native wheel raised an access violation.
Set ENABLE_LIGHTGBM=1 to force training.
```

## API Endpoints

- `GET /api/health`
- `GET /api/ml/features`
- `GET /api/ml/metrics`
- `GET /api/transactions?limit=120`
- `POST /api/score`

The prediction response includes:

- Fraud probability
- Risk score from 0 to 1000
- Risk band
- Flagged decision
- Decision threshold
- Estimated loss exposure
- Recommended action
- SHAP-style feature explanations

## Local Setup

### Backend

```bash
pip install -r backend/requirements.txt
python -m backend.scripts.train_model
uvicorn backend.app.main:app --reload
```

Backend API:

```text
http://localhost:8000/api/health
```

### Frontend

```bash
cd frontend
npm install
npm run build
npm run dev
```

Frontend dashboard:

```text
http://localhost:5173
```

## Retraining

Run:

```bash
python -m backend.scripts.train_model
```

This regenerates:

- Synthetic transaction dataset
- Feature matrix
- Baseline and stronger supervised ML models
- Class imbalance report
- PR-AUC, precision, recall, F1, FPR, ROC AUC
- Cost-aware decision threshold
- Estimated financial losses prevented
- Saved model artifact
- Metrics JSON artifact
- Dashboard transaction sample

## Testing

Backend:

```bash
pytest backend/tests
```

Frontend:

```bash
cd frontend
npm run build
```

Latest verification:

- Backend tests: `4 passed`
- Frontend build: passed

## Docker

```bash
docker compose -f infra/docker-compose.yml up --build
```

Services:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
- PostgreSQL: `localhost:5432`

