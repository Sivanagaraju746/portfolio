# Personal Projects

Two runnable portfolio prototypes. Neither is an employer system or a production deployment.

| Project | Purpose | Start here |
| --- | --- | --- |
| Access Controlled Document Q&A | Role-filtered extractive retrieval over fictional documents | [Setup and code](document-qa/README.md) |
| ML Prediction API | Synthetic-data classification, evaluation, and validated inference | [Setup and code](ml-prediction-api/README.md) |

Each directory contains its own pinned dependencies, Dockerfile, tests, and limitations. Use Python 3.12. The source and tests were executed locally; Docker builds and public API hosting have not been verified. GitHub Pages hosts the portfolio and PDF, not these Python services.

Local validation: 9 document Q&A tests and 10 model-serving tests passed. Tests exercise real endpoints through FastAPI TestClient and train the model locally. See each test file for the cases and assertions.
