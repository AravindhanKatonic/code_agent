import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))
import routes.vectorstores.milvus as milvus_mod
from langchain.embeddings import HuggingFaceEmbeddings

@pytest.fixture(scope="function")
def env_setup(monkeypatch):
    class CallableLogger:
        def info(self, msg, *args, **kwargs):
            print("[LOG INFO]", msg)
        def error(self, msg, *args, **kwargs):
            print("[LOG ERROR]", msg)

    logger = CallableLogger()

    hf_embedding = HuggingFaceEmbeddings()

    monkeypatch.setattr(milvus_mod, "ALLMINILM", "all-minilm")

    monkeypatch.setattr(milvus_mod, "get_model_provider", lambda t: "OpenAI")

    monkeypatch.setattr(milvus_mod, "get_model_endpoint", lambda t: "sentence-transformers/all-MiniLM-L6-v2")

    import types
    fake_embeddings_module = types.ModuleType("routes.embeddings.openai_embeddings")
    fake_embeddings_module.openai_embeds = lambda logger, model_name, settings: hf_embedding
    sys.modules["routes.embeddings.openai_embeddings"] = fake_embeddings_module

    # Monkeypatch Milvus class to a fake one to avoid network calls
    class FakeMilvus:
        def __init__(self, embedding_function, connection_args, collection_name):
            print("[ MILVUS INIT]")
            print("Embedding Function:", embedding_function)
            print("Connection Args:", connection_args)
            print("Collection Name:", collection_name)
            self.embedding_function = embedding_function
            self.connection_args = connection_args
            self.collection_name = collection_name

    monkeypatch.setattr(milvus_mod, "Milvus", FakeMilvus)

    return {
        "LOGGER": logger,
        "EMBEDDING": hf_embedding
    }


def test_load_milvus_with_openai_provider(env_setup):
    print("\n=== Running test_load_milvus_with_openai_provider ===")
    vector_db = milvus_mod.load_milvus(
        env_setup["LOGGER"],
        embedding_type="openai-type",
        collection_name="TestCollection",
        embedding_settings={"metadata": {"some": "value"}}
    )
    assert vector_db is not None
    assert vector_db.collection_name == "TestCollection"
    print("=== Finished test_load_milvus_with_openai_provider ===\n")
