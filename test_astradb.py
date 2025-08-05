import sys
import subprocess
import os
import types
from unittest.mock import patch, MagicMock
import pytest

# Auto-install required packages
required_packages = ["langchain-community"]
for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Mock unresolved internal modules
mock_chat = types.ModuleType("chat")
mock_src = types.ModuleType("chat.src")
mock_mongo_init = types.ModuleType("chat.src.mongo_init")
mock_logutils = types.ModuleType("chat.src.logutils")

mock_mongo_init.get_instance_data = lambda: None
mock_mongo_init.get_model_provider = lambda x: "OpenAI"
mock_mongo_init.get_model_endpoint = lambda x: "mock-endpoint"
mock_logutils.handle_exception = lambda: "mock traceback"

sys.modules["chat"] = mock_chat
sys.modules["chat.src"] = mock_src
sys.modules["chat.src.mongo_init"] = mock_mongo_init
sys.modules["chat.src.logutils"] = mock_logutils

# Mock embeddings
mock_embeddings_mod = types.ModuleType("embeddings")
mock_openai_embeddings = types.ModuleType("embeddings.openai_embeddings")
mock_openai_embeddings.openai_embeds = lambda service_type, name: MagicMock()

sys.modules["embeddings"] = mock_embeddings_mod
sys.modules["embeddings.openai_embeddings"] = mock_openai_embeddings

# Path setup for import
sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

# Mock environment variables
os.environ["PROJECT_NAME"] = "demo_project"
os.environ["USERID"] = "demo_user"
os.environ["ASTRA_DB_TOKEN"] = "token"
os.environ["ASTRA_DB_ENDPOINT"] = "endpoint"
os.environ["COLLECTION_NAME"] = "my_collection"
os.environ["EMBEDDING_SERVICE_TYPE"] = "OpenAI"

from routes.vectorstores.astradb import create_embeddings_with_astradb, load_astradb

# ---------- Fixtures ----------
@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.info = lambda msg: print(f"[INFO] {msg}")
    logger.error = lambda msg: print(f"[ERROR] {msg}")
    return logger

@pytest.fixture
def mock_embeddings():
    return MagicMock()

@pytest.fixture
def mock_docs():
    return [{"text": "Hello, test doc"}]

@pytest.fixture
def mock_mongo_collection():
    return MagicMock()

# ---------- Tests ----------

# Positive test for create_embeddings_with_astradb
def test_create_embeddings_success(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    mock_mongo_collection.update_one = MagicMock()
    with patch("routes.vectorstores.astradb.AstraDB") as MockAstra:
        instance = MockAstra.return_value
        instance.add_documents.return_value = True

        create_embeddings_with_astradb(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)

        instance.add_documents.assert_called_once()
        mock_mongo_collection.update_one.assert_called_once()

#  Negative test: Missing environment variables
def test_create_embeddings_missing_env(monkeypatch, mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    monkeypatch.delenv("ASTRA_DB_TOKEN", raising=False)
    monkeypatch.delenv("ASTRA_DB_ENDPOINT", raising=False)
    with patch("routes.vectorstores.astradb.exit") as mock_exit:
        create_embeddings_with_astradb(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
        assert mock_exit.call_count >= 1

#  Negative test: Unauthorized exception from AstraDB
def test_create_embeddings_unauthorized(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    with patch("routes.vectorstores.astradb.AstraDB", side_effect=Exception("401 Unauthorized")):
        with patch("routes.vectorstores.astradb.exit") as mock_exit:
            create_embeddings_with_astradb(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
            assert mock_exit.call_count >= 1

#  Negative test: Ingest fails
def test_create_embeddings_ingest_failure(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    mock_mongo_collection.update_one = MagicMock()
    with patch("routes.vectorstores.astradb.AstraDB") as MockAstra:
        instance = MockAstra.return_value
        instance.add_documents.side_effect = Exception("Ingest failed")

        with patch("routes.vectorstores.astradb.exit") as mock_exit:
            create_embeddings_with_astradb(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
            assert mock_exit.call_count >= 1

#  Positive test for load_astradb
def test_load_astradb_success(mock_logger):
    with patch.dict(os.environ, {"EMBEDDING_INSTANCE_NAME": "default"}):
        with patch("routes.vectorstores.astradb.AstraDB") as MockAstra:
            with patch("routes.vectorstores.astradb.get_model_provider", return_value="OpenAI"):
                with patch("routes.vectorstores.astradb.handle_exception", return_value="mock traceback"):
                    with patch("embeddings.openai_embeddings.openai_embeds", return_value=MagicMock()) as mock_embed:
                        db = load_astradb(mock_logger)
                        assert db is not None
                        MockAstra.assert_called_once()
                        mock_embed.assert_called_once()

#  Negative test: unknown provider
def test_load_astradb_unknown_provider(monkeypatch, mock_logger):
    monkeypatch.setitem(os.environ, "EMBEDDING_SERVICE_TYPE", "Unknown")
    with patch("routes.vectorstores.astradb.AstraDB") as MockAstra:
        with pytest.raises(Exception):
            load_astradb(mock_logger)
