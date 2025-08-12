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

class FakeEmbedding:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]  # Dummy vector

@pytest.fixture(scope="module")
def mongo_collection():
    class FakeMongoCollection:
        def update_one(self, filter, update):
            print(f"[MONGO] update_one called with filter={filter} update={update}")
            return True
    return FakeMongoCollection()

# --- Fake AstraDB ---
class FakeAstraDB:
    instances = []  # <-- Track all created instances

    def __init__(self, embedding, collection_name, token, api_endpoint):
        print(f"[FAKE AstraDB] Initialized with collection={collection_name}")
        self.docs = []
        FakeAstraDB.instances.append(self)  # Save the instance

    def add_documents(self, docs):
        print(f"[FAKE AstraDB] add_documents called with {len(docs)} docs")
        self.docs.extend(docs)
        return True

class FakeLogger:
    def __init__(self):
        self.logs = []

    def __call__(self, msg, filename=None, lineno=None, level=None):
        self.logs.append((msg, filename, lineno, level))


class FakeMongoCollection:
    def __init__(self):
        self.updates = []

    def update_one(self, filter_param, update_query):
        self.updates.append((filter_param, update_query))
        return True


def test_create_embeddings_with_astradb_fake():
    # Patch imports before importing astradb.py
    sys.modules["katonic-converse/smartchatcopilot/routes/embeddings.embeddings"] = types.SimpleNamespace(
        load_embeddings=lambda *args, **kwargs: FakeEmbedding()
    )

    import routes.vectorstores.astradb as astradb

    fake_logger = FakeLogger()
    fake_collection = FakeMongoCollection()
    with patch.object(astradb, "AstraDB", FakeAstraDB):
        astradb.create_embeddings_with_astradb(
            "dummy.txt",
            "movie_reviews",
            fake_logger,
            fake_collection
        )

    # ====== VERIFY ======
    assert FakeAstraDB.instances, "[ERROR] No FakeAstraDB instance created!"
    stored_docs = FakeAstraDB.instances[0].docs

    print("\n[CHECK] Final docs stored in FakeAstraDB instance:")
    for i, doc in enumerate(stored_docs, 1):
        print(f"  Doc {i}: {doc}")

    assert len(stored_docs) > 0, "No documents were added to FakeAstraDB"

# --- Test for load_astradb ---
def test_load_astradb_fake(fake_logger, embeddings_model):
    print("\n[TEST] Starting test_load_astradb_fake")

    with patch.object(astradb, "AstraDB", FakeAstraDB) as patched_astradb:
        print("[PATCH] Replaced astradb.AstraDB with FakeAstraDB")

        # Patch globals so NameError doesn't occur
        astradb.EMBEDDING_INSTANCE_NAME = "fake_instance"  # MOCKED
        astradb.EMBEDDING_SERVICE_TYPE = "OpenAI"          # MOCKED
        astradb.COLLECTION_NAME = "movie_reviews"          # MOCKED
        astradb.handle_exception = lambda: "Fake traceback"  # MOCKED
        astradb.get_model_provider = lambda x: "OpenAI"      # MOCKED
        astradb.get_model_endpoint = lambda x: "fake_model"  # MOCKED
        print("[PATCH] Patched global variables and helper functions in astradb")

        # Inject fake embeddings
        fake_embeds_module = MagicMock()
        fake_embeds_module.openai_embeds.return_value = embeddings_model  # MOCKED
        fake_embeddings_pkg = types.ModuleType("embeddings")
        fake_embeddings_pkg.openai_embeddings = fake_embeds_module
        sys.modules["embeddings"] = fake_embeddings_pkg
        print("[PATCH] Injected fake embeddings module into sys.modules['embeddings']")

        # CALL ORIGINAL CODE
        print("[CALL] Invoking load_astradb from astradb.py")
        result = astradb.load_astradb(logger=fake_logger, col_name="movie_reviews")
        print("[RESULT] load_astradb returned:", result)
        print("[CHECK] Type of returned object:", type(result))
        print("[CHECK] Docs stored inside FakeAstraDB instance:", getattr(result, "docs", []))

        assert isinstance(result, FakeAstraDB)
        fake_logger.info("✅ load_astradb test passed.")
