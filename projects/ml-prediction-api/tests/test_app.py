import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fastapi.testclient import TestClient
from app import create_app, load_bundle, score
from train import train

KEY = 'm' * 32
CUSTOMER = {'tenure_months': 12, 'monthly_charge': 85, 'support_tickets': 2, 'month_to_month': 1}

class PredictionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.bundle, cls.pipeline, cls.test_x = train(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.client = TestClient(create_app(str(Path(self.temp.name) / 'model.json'), KEY))
        self.client.__enter__()
        self.headers = {'X-API-Key': KEY}

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_export_matches_sklearn_heldout_predictions(self):
        bundle, _ = load_bundle(Path(self.temp.name) / 'model.json')
        np.testing.assert_allclose(score(bundle, self.test_x), self.pipeline.predict_proba(self.test_x)[:, 1], atol=1e-12)

    def test_missing_authentication(self):
        self.assertEqual(self.client.post('/predict', json=CUSTOMER).status_code, 401)

    def test_prediction_metadata_and_probability(self):
        response = self.client.post('/predict', json=CUSTOMER, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['data_source'], 'synthetic_demo_only')
        self.assertEqual(len(result['model_sha256']), 64)
        self.assertTrue(0 <= result['predictions'][0]['churn_probability'] <= 1)

    def test_batch_matches_single(self):
        single = self.client.post('/predict', json=CUSTOMER, headers=self.headers).json()
        batch = self.client.post('/predict/batch', json={'records': [CUSTOMER, CUSTOMER]}, headers=self.headers).json()
        for prediction in batch['predictions']:
            self.assertAlmostEqual(prediction['churn_probability'], single['predictions'][0]['churn_probability'], places=12)
            self.assertEqual(prediction['label'], single['predictions'][0]['label'])

    def test_out_of_range_rejected(self):
        for field, value in [('tenure_months', -1), ('monthly_charge', 999), ('month_to_month', 2)]:
            with self.subTest(field=field):
                self.assertEqual(self.client.post('/predict', json={**CUSTOMER, field: value}, headers=self.headers).status_code, 422)

    def test_extra_and_missing_features_rejected(self):
        self.assertEqual(self.client.post('/predict', json={**CUSTOMER, 'unexpected': 1}, headers=self.headers).status_code, 422)
        self.assertEqual(self.client.post('/predict', json={'tenure_months': 12}, headers=self.headers).status_code, 422)

    def test_batch_limit(self):
        self.assertEqual(self.client.post('/predict/batch', json={'records': [CUSTOMER] * 101}, headers=self.headers).status_code, 422)

    def test_metrics_count_scored_records(self):
        self.client.post('/predict/batch', json={'records': [CUSTOMER] * 2}, headers=self.headers)
        metrics = self.client.get('/metrics', headers=self.headers).json()
        self.assertEqual(metrics['predictions'], 2)
        self.assertEqual(metrics['requests'], 1)
        self.assertIn('monthly_charge', metrics['normalized_feature_mean_shift'])

    def test_invalid_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.json'
            path.write_text(json.dumps({**self.bundle, 'scale': [0, 1, 1, 1]}))
            with self.assertRaises(ValueError): load_bundle(path)

    def test_training_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            second, _, _ = train(temp)
            self.assertEqual(self.bundle['coefficients'], second['coefficients'])
            self.assertEqual(self.bundle['metrics'], second['metrics'])

if __name__ == '__main__':
    unittest.main()
