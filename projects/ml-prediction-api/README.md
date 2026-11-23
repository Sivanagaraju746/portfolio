# ML Prediction API

Personal portfolio prototype using Python, FastAPI, Pydantic, NumPy, and scikit-learn. Trains logistic regression on **synthetic customer records** and serves churn probabilities. Demonstrates reproducible training, held-out evaluation, validated inference, and aggregate monitoring.

## Features

- 2,400 generated records, fixed random seed, stratified 75/25 split; StandardScaler fits only training data.
- Held-out ROC-AUC, accuracy at 0.5, and Brier score.
- Data-only JSON model parameters; no pickle/joblib loading. Version and SHA-256 artifact fingerprint accompany predictions.
- Authenticated single/batch endpoints, feature ranges, extra-field rejection, finite values, and 100-record batch limit.
- Process-local request counts, scoring latency, and descriptive normalized feature mean shifts; no individual inputs in logs.
- Ten automated tests, including exported inference parity with scikit-learn and reproducible training.

## Run locally

Python 3.12; run from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
export ML_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(24))')"
uvicorn app:app --host 127.0.0.1 --port 8001
```

In another shell with the same key:

```bash
curl http://127.0.0.1:8001/predict \
  -H "X-API-Key: $ML_API_KEY" -H 'Content-Type: application/json' \
  -d '{"tenure_months":12,"monthly_charge":85,"support_tickets":2,"month_to_month":1}'
```

API docs: `http://127.0.0.1:8001/docs`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Public health check |
| `POST /predict` | One validated record |
| `POST /predict/batch` | Up to 100 records in `{"records":[...]}` |
| `GET /model` | Provenance, feature order, metrics, fingerprint |
| `GET /metrics` | Aggregate inference monitoring |

All except health require `X-API-Key`.

## Verify and reproduce

```bash
python -m unittest discover -s tests -v
python train.py --output artifacts
```

`artifacts/model.json` is the runnable artifact; `artifacts/evaluation.json` records evaluation. A measured report is committed at `reports/evaluation.json`. Repeated training reproduces coefficients and metrics; timestamps and fingerprints change.

## Docker

```bash
docker build -t ml-prediction-api .
docker run --rm -p 127.0.0.1:8001:8001 -e ML_API_KEY ml-prediction-api
```

The image trains on synthetic data during build and runs inference as a non-root user. Docker configuration is included; container execution has not been verified in this workspace.

## Limits

Synthetic test metrics do not establish real-world performance or business value. Target generation is defined in `train.py`; threshold 0.5 is a demo choice. Feature mean shifts are descriptive, not a statistical drift detector. Monitoring is process-local and resets on restart. Fingerprinting identifies an artifact but does not establish authenticity. Public deployment would also need TLS, rate limits, managed identity, trusted artifact signing, and centralized monitoring.

## Resume description

Built a reproducible synthetic-data classification pipeline and FastAPI inference service with single/batch validation, versioned JSON artifacts, held-out evaluation, and aggregate request and feature monitoring.
