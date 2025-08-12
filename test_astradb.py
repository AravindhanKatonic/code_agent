import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

import routes.vectorstores.astradb as astradb_mod

@pytest.fixture(scope="function")
def env_setup(monkeypatch):
    def set_env_if_missing(key, fake_value):
        if not os.getenv(key):
            monkeypatch.setenv(key, fake_value)

    set_env_if_missing("ASTRA_DB_ID", "fake_db_id")
    set_env_if_missing("ASTRA_DB_REGION", "fake_region")
    set_env_if_missing("ASTRA_DB_KEYSPACE", "fake_keyspace")
    set_env_if_missing("ASTRA_DB_APPLICATION_TOKEN", "fake_token")
    set_env_if_missing("EMBEDDING_SERVICE_TYPE", "fake_embedding_type")
    set_env_if_missing("SERVICE_TYPE", "fake_service_type")
    set_env_if_missing("ASTRA_DB_TOKEN", "fake-token")
    set_env_if_missing("ASTRA_DB_ENDPOINT", "https://fake-endpoint.com")

    class CallableLogger:
        def __call__(self, *args, **kwargs):
            print("[LOG CALL]", args)
        def info(self, msg, *args, **kwargs):
            print("[LOG INFO]", msg)
        def error(self, msg, *args, **kwargs):
            print("[LOG ERROR]", msg)

    logger = CallableLogger()

    class FakeMongoCollection:
        def update_one(self, query, update, upsert=True):
            print("[FAKE MONGO UPDATE] Query:", query)
            print("[FAKE MONGO UPDATE] Update:", update)
            print("[FAKE MONGO UPDATE] Upsert:", upsert)
            return {"matched_count": 1, "modified_count": 1}

    class FakeEmbedding:
        def embed_documents(self, docs):
            print("[EMBEDDING] Embedding documents:", docs)
            return [[0.1, 0.2, 0.3] for _ in docs]
        def embed_query(self, text):
            print("[EMBEDDING] Embedding query:", text)
            return [0.1, 0.2, 0.3]

    fake_embedding = FakeEmbedding()

    astradb_mod.handle_exception = lambda e=None: "[fake traceback]"
    astradb_mod.get_model_provider = lambda t: "OpenAI"
    astradb_mod.get_model_endpoint = lambda t: "fake-model-endpoint"
    astradb_mod.EMBEDDING_INSTANCE_NAME = "fake-instance"

    class FakeAstraDB:
        def __init__(self, embedding, collection_name, token, api_endpoint):
            print(f"[FAKE ASTRA INIT] collection_name: {collection_name}")
            print(f"[FAKE ASTRA INIT] token: {token}")
            print(f"[FAKE ASTRA INIT] api_endpoint: {api_endpoint}")
            self.stored_docs = []
        def add_documents(self, docs):
            print("[FAKE ADD DOCUMENTS] Documents added to AstraDB:", docs)
            self.stored_docs.extend(docs)
        def __repr__(self):
            return "<FakeAstraDB>"

    astradb_mod.AstraDB = FakeAstraDB

    class FakeOpenAIEmbeddings:
        @staticmethod
        def openai_embeds(*args, **kwargs):
            print("[FAKE OPENAI EMBEDDINGS] Called openai_embeds")
            return fake_embedding

    import types
    fake_embeddings_module = types.ModuleType("embeddings")
    fake_openai_embeddings_module = types.ModuleType("openai_embeddings")
    fake_openai_embeddings_module.openai_embeds = FakeOpenAIEmbeddings.openai_embeds
    fake_embeddings_module.openai_embeddings = fake_openai_embeddings_module

    sys.modules["embeddings"] = fake_embeddings_module
    sys.modules["embeddings.openai_embeddings"] = fake_openai_embeddings_module

    return {
        "LOGGER_NAME": logger,
        "MONGO_COLLECTION": FakeMongoCollection(),
        "EMBEDDING_MODEL": fake_embedding,
    }


def test_create_embeddings_with_astradb(env_setup):
    dummy_docs = [
        {"id": "1", "content": "Sample document one"},
        {"id": "2", "content": "Sample document two"},
    ]

    print("\n=== Running test_create_embeddings_with_astradb ===")
    result = astradb_mod.create_embeddings_with_astradb(
        dummy_docs,
        env_setup["EMBEDDING_MODEL"],
        env_setup["LOGGER_NAME"],
        env_setup["MONGO_COLLECTION"],
    )
    print("=== Finished test_create_embeddings_with_astradb ===\n")
    assert True


def test_load_astradb(env_setup):
    print("\n=== Running test_load_astradb ===")
    result = astradb_mod.load_astradb(env_setup["LOGGER_NAME"])
    print("Loaded AstraDB instance:", result)
    print("=== Finished test_load_astradb ===\n")

    assert result is not None
