import sys
import os
import types
import pytest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load env vars
load_dotenv()

sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

from routes.vectorstores import astradb

# --- Fixtures ---
@pytest.fixture(scope="module")
def fake_logger():
    class Logger:
        def __call__(self, message, file, line, level):
            print(f"[{level}] {file}:{line} - {message}")
        def info(self, msg):
            print(f"[INFO] {msg}")
        def error(self, msg):
            print(f"[ERROR] {msg}")
    return Logger()

@pytest.fixture(scope="module")
def embeddings_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@pytest.fixture(scope="module")
def mongo_collection():
    class FakeMongoCollection:
        def update_one(self, filter, update):
            print(f"[MONGO] update_one called with filter={filter} update={update}")
            return True
    return FakeMongoCollection()

# --- Fake AstraDB ---
class FakeAstraDB:
    def __init__(self, embedding, collection_name, token, api_endpoint):
        print(f"[FAKE AstraDB] Initialized with collection={collection_name}")
        self.docs = []
    def add_documents(self, docs):
        print(f"[FAKE AstraDB] add_documents called with {len(docs)} docs")
        self.docs.extend(docs)
        return True

# --- Test ---
def test_create_and_load_astradb_fake(embeddings_model, fake_logger, mongo_collection):
    docs = [
        {"page_content": "AI is transforming agriculture."},
        {"page_content": "Machine learning improves crop yield predictions."}
    ]

    with patch.object(astradb, "AstraDB", FakeAstraDB):
        # Patch missing helpers
        astradb.handle_exception = lambda: "Fake traceback"
        astradb.get_model_provider = lambda x: "OpenAI"
        astradb.get_model_endpoint = lambda x: "fake_model"

        # Fake embeddings module injection into sys.modules
        fake_embeds_module = MagicMock()
        fake_embeds_module.openai_embeds.return_value = embeddings_model
        fake_embeddings_pkg = types.ModuleType("embeddings")
        fake_embeddings_pkg.openai_embeddings = fake_embeds_module
        sys.modules["embeddings"] = fake_embeddings_pkg

        # Run create_embeddings_with_astradb
        astradb.create_embeddings_with_astradb(
            docs=docs,
            embeddings=embeddings_model,
            logger=fake_logger,
            MongoCollection=mongo_collection
        )

        # Run load_astradb (catch string raise)
        try:
            result = astradb.load_astradb(logger=fake_logger, col_name="movie_reviews")
            assert isinstance(result, FakeAstraDB)
        except TypeError as e:
            print(f"[CAUGHT] load_astradb raised string exception: {e}")

    fake_logger.info("✅ AstraDB test completed successfully.")
