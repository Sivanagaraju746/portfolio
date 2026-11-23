"""Validated model inference service using a data-only JSON artifact."""
import hashlib
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated
from uuid import uuid4

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

FEATURES = ['tenure_months', 'monthly_charge', 'support_tickets', 'month_to_month']
logger = logging.getLogger('prediction.audit')


class Customer(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    tenure_months: int = Field(ge=0, le=120)
    monthly_charge: float = Field(ge=0, le=500)
    support_tickets: int = Field(ge=0, le=50)
    month_to_month: int = Field(ge=0, le=1)


class Batch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    records: list[Customer] = Field(min_length=1, max_length=100)


def load_bundle(path):
    raw = Path(path).read_bytes()
    bundle = json.loads(raw)
    if bundle.get('schema_version') != 1 or bundle.get('features') != FEATURES:
        raise ValueError('Unsupported model schema or feature order')
    for field in ['mean', 'scale', 'coefficients']:
        values = np.asarray(bundle[field], dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError('Invalid model parameters: ' + field)
    if (np.asarray(bundle['scale']) <= 0).any() or not np.isfinite(float(bundle['intercept'])):
        raise ValueError('Invalid scale or intercept')
    return bundle, hashlib.sha256(raw).hexdigest()


def score(bundle, rows):
    x = np.asarray(rows, dtype=float)
    z = ((x - np.asarray(bundle['mean'])) / np.asarray(bundle['scale'])) @ np.asarray(bundle['coefficients'])
    z = np.clip(z + bundle['intercept'], -700, 700)
    return 1 / (1 + np.exp(-z))


def create_app(artifact_path=None, api_key=None):
    path = artifact_path or os.getenv('MODEL_PATH', 'artifacts/model.json')
    token = api_key if api_key is not None else os.getenv('ML_API_KEY', '')
    state = {'predictions': 0, 'requests': 0, 'total_latency_ms': 0., 'sum': np.zeros(4)}
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        if len(token) < 24:
            raise RuntimeError('Set ML_API_KEY to a secret at least 24 characters long')
        try:
            app.state.bundle, app.state.sha256 = load_bundle(path)
        except FileNotFoundError as exc:
            raise RuntimeError('Run python train.py before starting the API') from exc
        yield

    app = FastAPI(title='ML Prediction API', version='1.0.0', lifespan=lifespan)

    def authenticate(x_api_key: Annotated[str | None, Header()] = None):
        if not x_api_key or not hmac.compare_digest(token.encode(), x_api_key.encode()):
            raise HTTPException(401, 'Valid X-API-Key required')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'version': '1.0.0'}

    def predict_rows(records):
        start = perf_counter()
        rows = [[record.model_dump()[feature] for feature in FEATURES] for record in records]
        probabilities = score(app.state.bundle, rows)
        request_id = uuid4().hex
        elapsed = (perf_counter() - start) * 1000
        with lock:
            state['predictions'] += len(rows)
            state['requests'] += 1
            state['total_latency_ms'] += elapsed
            state['sum'] += np.asarray(rows).sum(axis=0)
        logger.info(json.dumps({'event': 'predict', 'request_id': request_id, 'batch_size': len(rows),
                                'model_version': app.state.bundle['model_version'], 'latency_ms': round(elapsed, 3)}))
        return {'request_id': request_id, 'model_version': app.state.bundle['model_version'],
                'model_sha256': app.state.sha256, 'data_source': app.state.bundle['data_source'],
                'predictions': [{'churn_probability': float(p), 'label': int(p >= .5)} for p in probabilities]}

    @app.post('/predict', dependencies=[Depends(authenticate)])
    def predict(customer: Customer):
        return predict_rows([customer])

    @app.post('/predict/batch', dependencies=[Depends(authenticate)])
    def batch(request: Batch):
        return predict_rows(request.records)

    @app.get('/model', dependencies=[Depends(authenticate)])
    def model():
        bundle = app.state.bundle
        return {**{k: bundle[k] for k in ['model_version', 'features', 'data_source', 'train_rows', 'test_rows', 'metrics']},
                'sha256': app.state.sha256}

    @app.get('/metrics', dependencies=[Depends(authenticate)])
    def metrics():
        with lock:
            count = state['predictions']
            mean = state['sum'].copy() / count if count else None
            shift = ((mean - app.state.bundle['mean']) / app.state.bundle['scale']).tolist() if count else None
            return {'requests': state['requests'], 'predictions': count,
                    'mean_scoring_latency_ms': state['total_latency_ms'] / state['requests'] if state['requests'] else 0.,
                    'normalized_feature_mean_shift': dict(zip(FEATURES, shift)) if shift else None,
                    'scope': 'current_process_since_start',
                    'note': 'Descriptive aggregate feature shifts; not a statistical drift test.'}

    return app


app = create_app()
