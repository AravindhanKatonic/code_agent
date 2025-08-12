import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()  # Loads .env variables automatically for the tests

sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

import routes.vectorstores.pineconedb as pinecone_mod


@pytest.fixture(scope="function")
def env_setup(monkeypatch):
    # Do NOT set env vars here — assume they come from your .env file

    # Dummy logger class to capture logs
    class CallableLogger:
        def __call__(self, *args, **kwargs):
            print("[LOG CALL]", args)
        def info(self, msg, *args, **kwargs):
            print("[LOG INFO]", msg)
        def error(self, msg, *args, **kwargs):
            print("[LOG ERROR]", msg)

    logger = CallableLogger()

    # Dummy MongoCollection with update_one method
    class FakeMongoCollection:
        def update_one(self, query, update, upsert=True):
            print("[FAKE MONGO UPDATE] Query:", query)
            print("[FAKE MONGO UPDATE] Update:", update)
            print("[FAKE MONGO UPDATE] Upsert:", upsert)
            return {"matched_count": 1, "modified_count": 1}

    # Dummy embedding class simulating embedding methods
    class FakeEmbedding:
        def embed_documents(self, docs):
            print("[EMBEDDING] Embedding documents:", docs)
            return [[0.1, 0.2, 0.3] for _ in docs]
        def embed_query(self, text):
            print("[EMBEDDING] Embedding query:", text)
            return [0.1, 0.2, 0.3]

    fake_embedding = FakeEmbedding()

    # Dummy Pinecone client to mock pinecone.Pinecone
    class FakePineconeClient:
        def __init__(self, api_key):
            print(f"[FAKE PINECONE INIT] API Key: {api_key}")
            self._indexes = ["existing_index"]
        def list_indexes(self):
            return self._indexes
        def create_index(self, name, dimension, metric, spec):
            print(f"[FAKE PINECONE CREATE INDEX] Name: {name}, Dimension: {dimension}, Metric: {metric}, Spec: {spec}")
            self._indexes.append(name)

    # Dummy Langchain Pinecone wrapper
    class FakeLangchainPinecone:
        def __init__(self, index_name, embedding):
            print(f"[FAKE LANGCHAIN PINECONE INIT] Index: {index_name}")
            self.index_name = index_name
            self.embedding = embedding
            self.added_docs = []
        def add_documents(self, docs):
            print(f"[FAKE LANGCHAIN PINECONE] Adding documents: {docs}")
            self.added_docs.extend(docs)
        @classmethod
        def from_existing_index(cls, index_name, embedding):
            print(f"[FAKE LANGCHAIN PINECONE LOAD] Loading existing index: {index_name}")
            return cls(index_name, embedding)

    # Dummy embeddings modules to mock different providers
    import types
    fake_embeddings_module = types.ModuleType("embeddings")
    fake_openai_embeddings_module = types.ModuleType("openai_embeddings")
    fake_openai_embeddings_module.openai_embeds = lambda model_name, emb_type: fake_embedding
    fake_azure_embeddings_module = types.ModuleType("azure_embeddings")
    fake_azure_embeddings_module.azure_embeds = lambda emb_type: fake_embedding
    fake_katonic_embeddings_module = types.ModuleType("katonic_embeddings")
    fake_katonic_embeddings_module.katonic_embeds_training = lambda logger: fake_embedding
    fake_katonic_embeddings_module.katonic_embeds_deploy = lambda logger: fake_embedding
    fake_replicate_embeddings_module = types.ModuleType("replicate_embeddings")
    fake_replicate_embeddings_module.replicate_embeds_training = lambda model, instanceName: fake_embedding
    fake_replicate_embeddings_module.replicate_embeds_deploy = lambda model_name: fake_embedding

    fake_embeddings_module.openai_embeddings = fake_openai_embeddings_module
    fake_embeddings_module.azure_embeddings = fake_azure_embeddings_module
    fake_embeddings_module.katonic_embeddings = fake_katonic_embeddings_module
    fake_embeddings_module.replicate_embeddings = fake_replicate_embeddings_module

    sys.modules["embeddings"] = fake_embeddings_module
    sys.modules["embeddings.openai_embeddings"] = fake_openai_embeddings_module
    sys.modules["embeddings.azure_embeddings"] = fake_azure_embeddings_module
    sys.modules["embeddings.katonic_embeddings"] = fake_katonic_embeddings_module
    sys.modules["embeddings.replicate_embeddings"] = fake_replicate_embeddings_module

    # Patch pinecone.Pinecone and LangchainPinecone inside the pinecone_mod namespace
    monkeypatch.setattr(pinecone_mod, "Pinecone", FakePineconeClient)
    monkeypatch.setattr(pinecone_mod, "LangchainPinecone", FakeLangchainPinecone)

    # Patch get_model_provider and get_model_endpoint to always return dummy values
    monkeypatch.setattr(pinecone_mod, "get_model_provider", lambda t: "OpenAI")
    monkeypatch.setattr(pinecone_mod, "get_model_endpoint", lambda t: "dummy-model-endpoint")

    # Patch handle_exception to return dummy string
    monkeypatch.setattr(pinecone_mod, "handle_exception", lambda e=None: "[fake traceback]")

    return {
        "LOGGER_NAME": logger,
        "MONGO_COLLECTION": FakeMongoCollection(),
        "EMBEDDING_MODEL": fake_embedding,
    }


def test_create_embeddings_with_pineconde(env_setup):
    dummy_docs = [
        {"id": "1", "content": "Dummy document one"},
        {"id": "2", "content": "Dummy document two"},
    ]

    print("\n=== Running test_create_embeddings_with_pineconde ===")
    pinecone_mod.create_embeddings_with_pineconde(
        dummy_docs,
        env_setup["EMBEDDING_MODEL"],
        env_setup["LOGGER_NAME"],
        env_setup["MONGO_COLLECTION"],
    )
    print("=== Finished test_create_embeddings_with_pineconde ===\n")


def test_load_pineconedb(env_setup):
    print("\n=== Running test_load_pineconedb ===")
    vector_db = pinecone_mod.load_pineconedb(env_setup["LOGGER_NAME"])
    print("Loaded Pinecone Vector DB instance:", vector_db)
    print("=== Finished test_load_pineconedb ===\n")

    assert vector_db is not None

def test_create_embeddings_with_pineconde_empty_docs(env_setup):
    print("\n=== Running test_create_embeddings_with_pineconde with empty docs ===")
    pinecone_mod.create_embeddings_with_pineconde(
        [],
        env_setup["EMBEDDING_MODEL"],
        env_setup["LOGGER_NAME"],
        env_setup["MONGO_COLLECTION"],
    )
    print("=== Finished test_create_embeddings_with_pineconde with empty docs ===\n")


def test_load_pineconedb_custom_collection(env_setup):
    print("\n=== Running test_load_pineconedb with custom collection ===")
    vector_db = pinecone_mod.load_pineconedb(env_setup["LOGGER_NAME"], col_name="custom_index")
    assert vector_db.index_name == "custom_index"
    print("=== Finished test_load_pineconedb with custom collection ===\n")


