# Access Controlled Document Q&A

Personal portfolio prototype using Python, FastAPI, SQLite, scikit-learn TF-IDF, and cosine similarity. All included documents are fictional. Answers are **cited source excerpts**, not generated LLM responses. No external model calls, downloads, or API charges.

## Features

- Admin-only ingestion, overlapping word chunks, and persisted role permissions.
- Analyst/HR API keys map to server-controlled roles. Authorization filters documents **before vectorization and ranking**.
- Cited excerpts with stable chunk IDs, or refusal when authorized sources do not match.
- Sample email/SSN pattern redaction before storage; audit events omit questions, documents, and keys.
- Nine automated tests for authentication, access boundaries, citations, refusal, redaction, and input limits.

## Setup

Python 3.12; run from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export QA_ADMIN_KEY="$(python -c 'import secrets; print(secrets.token_hex(24))')"
export QA_ANALYST_KEY="$(python -c 'import secrets; print(secrets.token_hex(24))')"
export QA_HR_KEY="$(python -c 'import secrets; print(secrets.token_hex(24))')"
export QA_API_KEYS="$(python -c 'import json,os; print(json.dumps({os.environ["QA_ADMIN_KEY"]:"admin",os.environ["QA_ANALYST_KEY"]:"analyst",os.environ["QA_HR_KEY"]:"hr"}))')"
uvicorn app:app --host 127.0.0.1 --port 8000 &
QA_SERVER_PID=$!
python load_samples.py
curl http://127.0.0.1:8000/query \
  -H "X-API-Key: $QA_ANALYST_KEY" -H 'Content-Type: application/json' \
  -d '{"question":"Who reviews portfolio risk reports?","top_k":2}'
# When finished:
kill "$QA_SERVER_PID"
```

Wait until Uvicorn reports startup complete before running `load_samples.py`. API documentation is at `http://127.0.0.1:8000/docs`. `POST /documents` requires admin access; `POST /query` requires a valid key; `GET /health` is public. Similarity scores are retrieval scores, not answer confidence.

## Tests and Docker

```bash
python -m unittest discover -s tests -v
docker build -t document-qa .
docker run --rm -p 127.0.0.1:8000:8000 -e QA_API_KEYS -v qa-data:/data document-qa
```

The container runs as a non-root user and persists SQLite in the mounted volume. Docker configuration is included; container execution has not been verified in this workspace.

## Limits

Single-organization demonstration, not a production security/compliance product. Regex redaction is incomplete. API keys need rotation and secure distribution. Add TLS, rate limits, centralized identity, tenant isolation, and storage encryption before public deployment. Database limit: 2,000 chunks. TF-IDF is fitted per query and unsuitable for large corpora. Authorization tests do not constitute a comprehensive security assessment. Generative RAG would need a model integration and additional evaluations.

## Resume description

Built an extractive document Q&A prototype with FastAPI, SQLite, TF-IDF, role-filtered retrieval, source citations, sample redaction, and access-control/refusal tests.
