import sys
import os
import types
from unittest.mock import patch, MagicMock
import pytest

# 🧪 Handle missing packages like pymilvus
mock_pymilvus = types.ModuleType("pymilvus")
sys.modules["pymilvus"] = mock_pymilvus

# 🧪 Add mock for Milvus inside langchain_community.vectorstores
mock_langchain = types.ModuleType("langchain_community")
mock_vectorstores = types.ModuleType("langchain_community.vectorstores")
mock_vectorstores.Milvus = MagicMock()
sys.modules["langchain_community"] = mock_langchain
sys.modules["langchain_community.vectorstores"] = mock_vectorstores

# 🔧 Set required environment variables
os.environ["VECTORDB_URL"] = "localhost"
os.environ["VECTORDB_PORT"] = "19530"
os.environ["VECTORDB_USERNAME"] = "admin"
os.environ["VECTORDB_PASSWORD"] = "password"

# 🧩 Mock dependencies
sys.modules["routes.embeddings.openai_embeddings"] = types.SimpleNamespace(
    openai_embeds=lambda logger, name, settings: MagicMock()
)
sys.modules["routes.embeddings.azure_embeddings"] = types.SimpleNamespace(
    azure_embeds=lambda settings: MagicMock()
)
sys.modules["routes.embeddings.vllm_embeddings"] = types.SimpleNamespace(
    vllm_embed=lambda settings, logger: MagicMock()
)
sys.modules["routes.embeddings.custom_embedding"] = types.SimpleNamespace(
    katonic_custom_embedding=lambda logger: MagicMock(),
    katonic_embeds_training=lambda: MagicMock(),
)
sys.modules["routes.embeddings.katonic_embeddings"] = types.SimpleNamespace(
    katonic_embeds_training=lambda settings, logger: MagicMock()
)
sys.modules["routes.embeddings.replicate_embeddings"] = types.SimpleNamespace(
    replicate_embeds_training=lambda model: MagicMock()
)

# 🧩 Mock support functions
sys.modules["routes.utilities.constants"] = types.SimpleNamespace(
    ALLMINILM="ALLMINILM"
)
sys.modules["routes.utilities.mongo_init"] = types.SimpleNamespace(
    get_model_provider=lambda etype: etype,
    get_model_endpoint=lambda etype: f"model-{etype}"
)

sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))
from routes.vectorstores.milvus import load_milvus


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.info = lambda msg: print(f"[INFO] {msg}")
    logger.error = lambda msg: print(f"[ERROR] {msg}")
    return logger


@pytest.fixture
def embedding_settings():
    return {"metadata": {"embedFramework": "vllm"}}


def test_load_milvus_openai(mock_logger, embedding_settings):
    db = load_milvus(mock_logger, "OpenAI", "test_collection", embedding_settings)
    assert db is not None


def test_load_milvus_azure(mock_logger, embedding_settings):
    db = load_milvus(mock_logger, "Azure OpenAI", "test_collection", embedding_settings)
    assert db is not None


def test_load_milvus_katonic_vllm(mock_logger, embedding_settings):
    db = load_milvus(mock_logger, "katonic", "test_collection", embedding_settings)
    assert db is not None


def test_load_milvus_katonic_tei(mock_logger):
    settings = {"metadata": {"embedFramework": "tei"}}
    db = load_milvus(mock_logger, "katonic", "test_collection", settings)
    assert db is not None


def test_load_milvus_replicate(mock_logger, embedding_settings):
    db = load_milvus(mock_logger, "Replicate", "test_collection", embedding_settings)
    assert db is not None


def test_load_milvus_custom_embed(mock_logger, embedding_settings):
    db = load_milvus(mock_logger, "ALLMINILM", "FAQ", embedding_settings)
    assert db is not None


def test_load_milvus_missing_env(mock_logger, monkeypatch):
    monkeypatch.delenv("VECTORDB_URL", raising=False)
    with pytest.raises(KeyError):
        load_milvus(mock_logger, "OpenAI", "test", {})


def test_load_milvus_invalid_provider(mock_logger):
    # Simulate a broken import by removing the module from sys.modules
    sys.modules.pop("routes.embeddings.openai_embeddings", None)
    with patch("routes.utilities.mongo_init.get_model_provider", return_value="OpenAI"):
        with pytest.raises(ImportError):
            load_milvus(mock_logger, "OpenAI", "test", {})


