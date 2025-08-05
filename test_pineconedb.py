import sys
import os
import types
import pytest
from unittest.mock import patch, MagicMock

# Auto-install required packages
required = ["pinecone-client", "langchain-pinecone"]
import subprocess
for pkg in required:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg])

# Patch all necessary external modules
mock_chat = types.ModuleType("chat")
mock_src = types.ModuleType("chat.src")
mock_logutils = types.ModuleType("chat.src.logutils")
mock_mongo_init = types.ModuleType("chat.src.mongo_init")

mock_logutils.handle_exception = lambda: "mock traceback"
mock_mongo_init.get_model_provider = lambda x: "OpenAI"
mock_mongo_init.get_model_endpoint = lambda x: "mock-endpoint"
mock_mongo_init.get_instance_data = lambda: None

sys.modules["chat"] = mock_chat
sys.modules["chat.src"] = mock_src
sys.modules["chat.src.logutils"] = mock_logutils
sys.modules["chat.src.mongo_init"] = mock_mongo_init

sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

# Set mock environment
os.environ.update({
    "PROJECT_NAME": "demo_project",
    "USERID": "user123",
    "COLLECTION_NAME": "demo_collection",
    "EMBEDDING_SERVICE_TYPE": "OpenAI",
    "PINECONE_API_KEY": "fake-key",
    "PINECONE_ENVIRONMENT_NAME": "us-west1-gcp"
})

from routes.vectorstores.pineconedb import create_embeddings_with_pineconde, load_pineconedb

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def mock_embeddings():
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 10
    mock.embed_documents.return_value = [[0.1]*10]
    return mock

@pytest.fixture
def mock_docs():
    return [{"text": "Hello Pinecone"}]

@pytest.fixture
def mock_mongo_collection():
    return MagicMock()

# -------------------- ✅ Positive Test Cases --------------------

def test_create_embeddings_with_pineconde_success(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    with patch("routes.vectorstores.pineconedb.LangchainPinecone") as MockLCPC, \
         patch("routes.vectorstores.pineconedb.Pinecone") as MockPinecone, \
         patch("routes.vectorstores.pineconedb.PodSpec") as MockPodSpec, \
         patch("routes.vectorstores.pineconedb.get_embedding_dimension", return_value=10):

        pine = MockPinecone.return_value
        pine.list_indexes.return_value = [{"name": "other_collection"}]
        MockLCPC.return_value.add_documents.return_value = True

        create_embeddings_with_pineconde(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)

        assert mock_mongo_collection.update_one.called

def test_load_pineconedb_success(mock_logger):
    with patch("routes.vectorstores.pineconedb.LangchainPinecone.from_existing_index") as mock_index, \
         patch("routes.vectorstores.pineconedb.get_model_provider", return_value="OpenAI"), \
         patch("routes.vectorstores.pineconedb.get_model_endpoint", return_value="mock-endpoint"), \
         patch("routes.vectorstores.pineconedb.openai_embeddings") as mock_embed_module:

        mock_embed_module.openai_embeds.return_value = MagicMock()
        mock_index.return_value = "vector-db"
        vector = load_pineconedb(mock_logger)
        assert vector == "vector-db"

# -------------------- ❌ Negative Test Cases --------------------

def test_create_embeddings_missing_creds(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection, monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with patch("routes.vectorstores.pineconedb.exit") as mock_exit:
        create_embeddings_with_pineconde(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
        mock_exit.assert_called()

def test_create_embeddings_pinecone_exception(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection):
    class DummyException(Exception):
        def __init__(self):
            self.body = '{"status":"400","error":{"message":"single Starter index"}}'

    with patch("routes.vectorstores.pineconedb.Pinecone") as MockPinecone, \
         patch("routes.vectorstores.pineconedb.PodSpec"), \
         patch("routes.vectorstores.pineconedb.get_embedding_dimension", return_value=10), \
         patch("routes.vectorstores.pineconedb.exit") as mock_exit:

        inst = MockPinecone.return_value
        inst.list_indexes.return_value = []
        inst.create_index.side_effect = DummyException()

        create_embeddings_with_pineconde(mock_docs, mock_embeddings, mock_logger, mock_mongo_collection)
        mock_exit.assert_called()

def test_load_pineconedb_invalid_provider(mock_logger):
    with patch("routes.vectorstores.pineconedb.get_model_provider", return_value="Unknown"), \
         patch("routes.vectorstores.pineconedb.get_model_endpoint", return_value="mock-endpoint"), \
         patch("routes.vectorstores.pineconedb.LangchainPinecone.from_existing_index") as mock_index, \
         patch("routes.vectorstores.pineconedb.openai_embeddings") as mock_embed_module, \
         patch("routes.vectorstores.pineconedb.handle_exception", return_value="stacktrace"):

        mock_embed_module.openai_embeds.return_value = MagicMock()
        mock_index.side_effect = Exception("Unknown error")

        with pytest.raises(TypeError):  # Because code raises `str(error)` instead of `Exception`
            load_pineconedb(mock_logger)
