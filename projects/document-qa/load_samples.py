"""Load fictional sample documents into a locally running service."""
import json
import os
from pathlib import Path
import httpx

key = os.environ["QA_ADMIN_KEY"]
for document in json.loads(Path(__file__).with_name("data").joinpath("sample_documents.json").read_text()):
    response = httpx.post("http://127.0.0.1:8000/documents", headers={"X-API-Key": key}, json=document)
    response.raise_for_status()
    print(response.json())
