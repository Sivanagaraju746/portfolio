"""Train a reproducible model on explicitly synthetic customer records."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['tenure_months', 'monthly_charge', 'support_tickets', 'month_to_month']


def synthetic_data(seed=42, rows=2400):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, rows)
    charge = rng.uniform(20, 150, rows)
    tickets = rng.poisson(1.4, rows).clip(0, 10)
    contract = rng.integers(0, 2, rows)
    x = np.column_stack([tenure, charge, tickets, contract]).astype(float)
    logit = -1.4 - .035 * tenure + .01 * charge + .4 * tickets + .8 * contract
    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability)
    return x, y


def train(output=Path('artifacts'), seed=42):
    output = Path(output)
    x, y = synthetic_data(seed)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.25, random_state=seed, stratify=y)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    pipeline.fit(train_x, train_y)
    probability = pipeline.predict_proba(test_x)[:, 1]
    scaler = pipeline.named_steps['standardscaler']
    model = pipeline.named_steps['logisticregression']
    metrics = {'roc_auc': float(roc_auc_score(test_y, probability)),
               'accuracy_at_0_5': float(accuracy_score(test_y, probability >= .5)),
               'brier_score': float(brier_score_loss(test_y, probability))}
    bundle = {'schema_version': 1, 'model_version': '1.0.0', 'model_type': 'logistic_regression',
              'features': FEATURES, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
              'coefficients': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
              'data_source': 'synthetic_demo_only', 'seed': seed, 'train_rows': len(train_y),
              'test_rows': len(test_y), 'metrics': metrics,
              'trained_at': datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model.json').write_text(json.dumps(bundle, indent=2) + '\n')
    report = {k: bundle[k] for k in ['model_version', 'data_source', 'seed', 'train_rows', 'test_rows', 'metrics']}
    report['limitations'] = ['Synthetic held-out evaluation, not evidence of real customer performance.',
                             'Threshold 0.5 is a demo choice, not a validated business decision.',
                             'Aggregate feature mean shifts are descriptive and are not proof of model drift.']
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    return bundle, pipeline, test_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    bundle, _, _ = train(args.output)
    print(json.dumps({'model_version': bundle['model_version'], 'metrics': bundle['metrics']}, indent=2))
