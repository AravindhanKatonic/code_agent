import os
import sys
import types
import pytest
from unittest.mock import patch, MagicMock
import subprocess

# --- Install missing packages ---
required_packages = ["qdrant-client", "langchain-community"]
for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# --- Patch path and mock modules ---
sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

mock_chat = types.ModuleType("chat")
mock_src = types.ModuleType("chat.src")
mock_mongo_init = types.ModuleType("chat.src.mongo_init")
mock_logutils = types.ModuleType("chat.src.logutils")

mock_mongo_init.get_instance_data = lambda: None
mock_mongo_init.get_model_provider = lambda x: "OpenAI"
mock_mongo_init.get_model_endpoint = lambda x: "text-embedding-ada-002"
mock_logutils.handle_exception = lambda: "mock traceback"

sys.modules["chat"] = mock_chat
sys.modules["chat.src"] = mock_src
sys.modules["chat.src.mongo_init"] = mock_mongo_init
sys.modules["chat.src.logutils"] = mock_logutils

# --- Mock embeddings ---
mock_embeddings_mod = types.ModuleType("embeddings")
mock_openai = types.ModuleType("embeddings.openai_embeddings")
mock_openai.openai_embeds = lambda model, typ: MagicMock()
sys.modules["embeddings"] = mock_embeddings_mod
sys.modules["embeddings.openai_embeddings"] = mock_openai

# --- Set base env before import ---
os.environ["PROJECT_NAME"] = "test_project"
os.environ["USERID"] = "test_user"
os.environ["COLLECTION_NAME"] = "test_collection"
os.environ["VECTOR_DB_TYPE"] = "cloud"
os.environ["QDRANT_API_KEY"] = "dummy_key"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["EMBEDDING_SERVICE_TYPE"] = "OpenAI"

# --- Import the actual functions ---
from routes.vectorstores.qdrantdb import create_embeddings_with_qdrant, load_qdrantdb

# --- Fixtures ---
@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.info = lambda msg, *a: print(f"[INFO] {msg}")
    logger.error = lambda msg, *a: print(f"[ERROR] {msg}")
    return logger

@pytest.fixture
def mock_docs():
    return [{"text": "Hello Qdrant"}]

@pytest.fixture
def mock_embeddings():
    return MagicMock()

@pytest.fixture
def mock_mongo_collection():
    return MagicMock()

# --- TEST CASES ---

def test_create_embeddings_success(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    os.environ.update({
        "PROJECT_NAME": "demo",
        "USERID": "user123",
        "COLLECTION_NAME": "test_collection",
        "VECTOR_DB_TYPE": "cloud",
        "QDRANT_API_KEY": "fake-key",
        "QDRANT_URL": "http://localhost:6333",
        "EMBEDDING_SERVICE_TYPE": "OpenAI",
    })
    with patch("routes.vectorstores.qdrantdb.Qdrant.from_documents", return_value=True) as mock_q:
        create_embeddings_with_qdrant(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
        mock_q.assert_called_once()
        mock_mongo_collection.update_one.assert_called_once()

def test_create_embeddings_missing_env(monkeypatch, mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("PROJECT_NAME", "demo")
    monkeypatch.setenv("USERID", "user123")
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("VECTOR_DB_TYPE", "cloud")
    with patch("routes.vectorstores.qdrantdb.exit") as mock_exit:
        create_embeddings_with_qdrant(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
        assert mock_exit.call_count >= 1  


def test_create_embeddings_ingest_fail(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    os.environ.update({
        "PROJECT_NAME": "demo",
        "USERID": "user123",
        "COLLECTION_NAME": "test_collection",
        "VECTOR_DB_TYPE": "cloud",
        "QDRANT_API_KEY": "fake-key",
        "QDRANT_URL": "http://localhost:6333",
        "EMBEDDING_SERVICE_TYPE": "OpenAI",
    })
    with patch("routes.vectorstores.qdrantdb.Qdrant.from_documents", side_effect=Exception("ingest failed")), \
         patch("routes.vectorstores.qdrantdb.exit") as mock_exit:
        create_embeddings_with_qdrant(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
        assert mock_exit.call_count >= 1


def test_load_qdrantdb_success(mock_logger):
    os.environ.update({
        "PROJECT_NAME": "demo",
        "USERID": "user123",
        "COLLECTION_NAME": "test_collection",
        "VECTOR_DB_TYPE": "cloud",
        "QDRANT_API_KEY": "fake-key",
        "QDRANT_URL": "http://localhost:6333",
        "EMBEDDING_SERVICE_TYPE": "OpenAI",
    })
    with patch("routes.vectorstores.qdrantdb.qdrant_client.QdrantClient") as mock_qc, \
         patch("routes.vectorstores.qdrantdb.Qdrant") as mock_qdrant:
        db = load_qdrantdb(mock_logger)
        assert db is not None
        mock_client.assert_called_once()
        mock_qdrant.assert_called_once()

def test_load_qdrantdb_unknown_provider_no_crash(monkeypatch, mock_logger):
    monkeypatch.setitem(os.environ, "EMBEDDING_SERVICE_TYPE", "InvalidModel")
    monkeypatch.setitem(os.environ, "VECTOR_DB_TYPE", "cloud")
    monkeypatch.setitem(os.environ, "QDRANT_API_KEY", "123")
    monkeypatch.setitem(os.environ, "QDRANT_URL", "http://localhost")
    monkeypatch.setitem(os.environ, "COLLECTION_NAME", "test_collection")

    with patch("routes.vectorstores.qdrantdb.qdrant_client.QdrantClient") as mock_qc, \
         patch("routes.vectorstores.qdrantdb.Qdrant") as mock_qdrant:
        result = load_qdrantdb(mock_logger)
        assert result is not None

def test_load_qdrantdb_unknown_provider_skips_embeddings(monkeypatch, mock_logger):
    monkeypatch.setitem(os.environ, "EMBEDDING_SERVICE_TYPE", "Unknown")
    monkeypatch.setitem(os.environ, "VECTOR_DB_TYPE", "cloud")
    monkeypatch.setitem(os.environ, "QDRANT_API_KEY", "123")
    monkeypatch.setitem(os.environ, "QDRANT_URL", "http://localhost")
    monkeypatch.setitem(os.environ, "COLLECTION_NAME", "test_collection")

    # Patch QdrantClient and Qdrant to avoid real call
    with patch("routes.vectorstores.qdrantdb.qdrant_client.QdrantClient") as mock_qc, \
         patch("routes.vectorstores.qdrantdb.Qdrant") as mock_qdrant:
        db = load_qdrantdb(mock_logger)
        assert db is not None  # Or whatever value your fallback logic returns
        mock_qdrant.assert_called_once()  # It should still attempt to init Qdrant

