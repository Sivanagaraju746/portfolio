"""Access-controlled extractive Q&A demo; no external LLM calls."""
import hmac
import json
import logging
import os
import re
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("document_qa.audit")
Role = Literal["analyst", "hr"]


class DocumentInput(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=20, max_length=20000)
    allowed_roles: list[Role] = Field(min_length=1, max_length=2)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


def redact(text: str) -> str:
    """Demonstration patterns only; not a complete sensitive-data detector."""
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    return re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)


def chunks(text: str, size: int = 100, overlap: int = 20) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size - overlap)]


def create_app(db_path: str | None = None, accounts: dict | None = None) -> FastAPI:
    database = db_path or os.getenv("QA_DATABASE", "qa.sqlite3")
    keys = accounts if accounts is not None else json.loads(os.getenv("QA_API_KEYS", "{}"))

    @asynccontextmanager
    async def lifespan(app):
        if not keys or any(len(key) < 24 or role not in {"admin", "analyst", "hr"}
                           for key, role in keys.items()):
            raise RuntimeError("Set QA_API_KEYS to a JSON object of >=24-character keys and admin/analyst/hr roles")
        Path(database).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS chunks "
                               "(id TEXT PRIMARY KEY, document_id TEXT, title TEXT, text TEXT, roles TEXT)")
        yield

    app = FastAPI(title="Access-Controlled Document Q&A", version="1.0.0", lifespan=lifespan)

    def identify(x_api_key: Annotated[str | None, Header()] = None) -> str:
        if x_api_key is not None:
            for token, role in keys.items():
                if hmac.compare_digest(token.encode(), x_api_key.encode()):
                    return role
        raise HTTPException(401, "Valid X-API-Key required")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "1.0.0", "answer_mode": "extractive"}

    @app.post("/documents", status_code=201)
    def ingest(document: DocumentInput, role: Annotated[str, Depends(identify)]):
        if role != "admin":
            raise HTTPException(403, "Document ingestion requires admin access")
        document_id = uuid4().hex
        parts = chunks(redact(document.text))
        with sqlite3.connect(database) as connection:
            count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count + len(parts) > 2000:
                raise HTTPException(409, "Demo limit reached: 2000 chunks per database")
            connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", [
                (f"{document_id}:{i}", document_id, redact(document.title), part,
                 json.dumps(sorted(set(document.allowed_roles)))) for i, part in enumerate(parts)
            ])
        logger.info(json.dumps({"event": "ingest", "document_id": document_id, "chunks": len(parts)}))
        return {"document_id": document_id, "chunks": len(parts)}

    @app.post("/query")
    def query(request: Question, role: Annotated[str, Depends(identify)]):
        request_id = uuid4().hex
        with sqlite3.connect(database) as connection:
            rows = connection.execute("SELECT id, title, text, roles FROM chunks").fetchall()
        # Filter BEFORE vectorization, scoring, ranking, or building any response context.
        visible = [row for row in rows if role == "admin" or role in json.loads(row[3])]
        logger.info(json.dumps({"event": "query", "request_id": request_id, "role": role}))
        def refuse():
            return {"request_id": request_id, "answer": "No relevant authorized source found.",
                    "sources": [], "refused": True, "mode": "extractive"}
        if not visible:
            return refuse()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        try:
            matrix = vectorizer.fit_transform([row[2] for row in visible])
        except ValueError:
            return refuse()
        scores = cosine_similarity(vectorizer.transform([request.question]), matrix)[0]
        ranked = [i for i in np.argsort(-scores, kind="stable") if scores[i] >= 0.12][:request.top_k]
        if not ranked:
            return refuse()
        sources = [{"chunk_id": visible[i][0], "title": visible[i][1],
                    "excerpt": visible[i][2], "score": round(float(scores[i]), 4)} for i in ranked]
        return {"request_id": request_id, "answer": "\n\n".join(s["excerpt"] for s in sources),
                "sources": sources, "refused": False, "mode": "extractive"}

    return app


app = create_app()
