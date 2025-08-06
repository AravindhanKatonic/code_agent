import os
import sys
import types
import pytest
from unittest.mock import patch, MagicMock
import subprocess

# Install any missing packages
for pkg in ["weaviate-client", "langchain-community"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Add project root path
sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

# ----- Mock Unresolved Modules -----
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

# ----- Patch embeddings -----
mock_embed_mod = types.ModuleType("embeddings")
mock_openai = types.ModuleType("embeddings.openai_embeddings")
mock_openai.openai_embeds = lambda m, t: MagicMock()

sys.modules["embeddings"] = mock_embed_mod
sys.modules["embeddings.openai_embeddings"] = mock_openai

# Set minimal required env vars
os.environ["PROJECT_NAME"] = "test_project"
os.environ["USERID"] = "test_user"
os.environ["COLLECTION_NAME"] = "testcollection"
os.environ["VECTOR_DB_TYPE"] = "cloud"
os.environ["WEAVIATE_API_KEY"] = "dummykey"
os.environ["WEAVIATE_URL"] = "http://localhost:8080"
os.environ["EMBEDDING_SERVICE_TYPE"] = "OpenAI"

# ---- Import target module ----
from routes.vectorstores.weaviatedb import create_embeddings_with_weaviate, load_weaviatedb

# ---- Fixtures ----
@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.info = lambda msg, *a: print("[INFO]", msg)
    logger.error = lambda msg, *a: print("[ERROR]", msg)
    return logger

@pytest.fixture
def mock_docs():
    return [{"text": "sample data"}]

@pytest.fixture
def mock_embeddings():
    return MagicMock()

@pytest.fixture
def mock_mongo_collection():
    return MagicMock()

# ---- TEST CASES ----

def test_create_embeddings_success(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    with patch("routes.vectorstores.weaviatedb.Weaviate.from_documents", return_value=True) as mock_vec:
        create_embeddings_with_weaviate(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
        mock_vec.assert_called_once()
        mock_mongo_collection.update_one.assert_called_once()

def test_create_embeddings_missing_env(monkeypatch, mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    monkeypatch.delenv("WEAVIATE_URL", raising=False)
    monkeypatch.delenv("WEAVIATE_API_KEY", raising=False)
    with patch("routes.vectorstores.weaviatedb.exit") as mock_exit:
        create_embeddings_with_weaviate(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
        assert mock_exit.call_count >= 1

def test_create_embeddings_ingestion_failure(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    with patch("routes.vectorstores.weaviatedb.Weaviate.from_documents", side_effect=Exception("ingest failed")), \
         patch("routes.vectorstores.weaviatedb.exit") as mock_exit:
        create_embeddings_with_weaviate(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
        mock_exit.assert_called_once()

def test_load_weaviatedb_success(mock_logger):
    with patch("routes.vectorstores.weaviatedb.weaviate.Client") as mock_client, \
         patch("routes.vectorstores.weaviatedb.Weaviate") as mock_weaviate:
        result = load_weaviatedb(mock_logger)
        assert result is not None
        mock_client.assert_called_once()
        mock_weaviate.assert_called_once()

def test_load_weaviatedb_unknown_provider(mock_logger, monkeypatch):
    monkeypatch.setitem(os.environ, "EMBEDDING_SERVICE_TYPE", "Unknown")
    with patch("routes.vectorstores.weaviatedb.weaviate.Client") as mock_client, \
         patch("routes.vectorstores.weaviatedb.Weaviate") as mock_weaviate:
        result = load_weaviatedb(mock_logger)
        assert result is not None
        mock_client.assert_called_once()
        mock_weaviate.assert_called_once()


def test_load_weaviatedb_azure_openai(mock_logger, monkeypatch):
    monkeypatch.setitem(os.environ, "EMBEDDING_SERVICE_TYPE", "Azure OpenAI")

    mock_azure_mod = types.ModuleType("embeddings.azure_embeddings")
    mock_azure_mod.azure_embeds = lambda service_type: MagicMock()
    sys.modules["embeddings.azure_embeddings"] = mock_azure_mod

    with patch("routes.vectorstores.weaviatedb.weaviate.Client") as mock_client, \
         patch("routes.vectorstores.weaviatedb.Weaviate") as mock_weaviate:
        db = load_weaviatedb(mock_logger)
        assert db is not None
        mock_client.assert_called_once()
        mock_weaviate.assert_called_once()
def test_load_weaviatedb_katonic(mock_logger, monkeypatch):
    monkeypatch.setitem(os.environ, "EMBEDDING_SERVICE_TYPE", "Katonic LLM")

    mock_katonic_mod = types.ModuleType("embeddings.katonic_embeddings")
    mock_katonic_mod.katonic_embeds_deploy = lambda logger: MagicMock()
    sys.modules["embeddings.katonic_embeddings"] = mock_katonic_mod

    with patch("routes.vectorstores.weaviatedb.weaviate.Client") as mock_client, \
         patch("routes.vectorstores.weaviatedb.Weaviate") as mock_weaviate:
        db = load_weaviatedb(mock_logger)
        assert db is not None
        mock_client.assert_called_once()
        mock_weaviate.assert_called_once()
def test_load_weaviatedb_unknown_provider_skips_embeddings(monkeypatch, mock_logger):
    monkeypatch.setitem(os.environ, "EMBEDDING_SERVICE_TYPE", "Unknown")
    monkeypatch.setitem(os.environ, "VECTOR_DB_TYPE", "cloud")
    monkeypatch.setitem(os.environ, "WEAVIATE_API_KEY", "123")
    monkeypatch.setitem(os.environ, "WEAVIATE_URL", "http://localhost")
    monkeypatch.setitem(os.environ, "COLLECTION_NAME", "test_collection")

    with patch("routes.vectorstores.weaviatedb.weaviate.Client") as mock_client, \
         patch("routes.vectorstores.weaviatedb.Weaviate") as mock_weaviate:
        db = load_weaviatedb(mock_logger)
        assert db is not None
        mock_weaviate.assert_called_once()
def test_load_weaviatedb_unknown_provider_no_crash(monkeypatch, mock_logger):
    monkeypatch.setitem(os.environ, "EMBEDDING_SERVICE_TYPE", "InvalidModel")
    monkeypatch.setitem(os.environ, "VECTOR_DB_TYPE", "cloud")
    monkeypatch.setitem(os.environ, "WEAVIATE_API_KEY", "123")
    monkeypatch.setitem(os.environ, "WEAVIATE_URL", "http://localhost")
    monkeypatch.setitem(os.environ, "COLLECTION_NAME", "test_collection")

    with patch("routes.vectorstores.weaviatedb.weaviate.Client") as mock_client, \
         patch("routes.vectorstores.weaviatedb.Weaviate") as mock_weaviate:
        db = load_weaviatedb(mock_logger)
        assert db is not None
        mock_weaviate.assert_called_once()
