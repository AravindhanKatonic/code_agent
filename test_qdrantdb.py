import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()  
sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

import routes.vectorstores.qdrantdb as qdrant_mod

# Import HuggingFaceEmbeddings for dummy embedding object (or you can create a dummy one)
from langchain.embeddings import HuggingFaceEmbeddings


@pytest.fixture(scope="function")
def env_setup(monkeypatch):
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

    # Use real or dummy embedding
    huggingface_embedding = HuggingFaceEmbeddings()

    # Dummy Qdrant VectorStore wrapper
    class FakeQdrantVectorStore:
        def __init__(self, **kwargs):
            print(f"[FAKE QDRANT INIT] Args: {kwargs}")
            self.kwargs = kwargs

        @classmethod
        def from_documents(cls, documents, embedding, **kwargs):
            print(f"[FAKE QDRANT] from_documents called with docs: {documents}")
            return cls(documents=documents, embedding=embedding, **kwargs)

    # Patch the Qdrant class inside your module
    monkeypatch.setattr(qdrant_mod, "Qdrant", FakeQdrantVectorStore)

    # Patch handle_exception to return dummy string
    monkeypatch.setattr(qdrant_mod, "handle_exception", lambda e=None: "[fake traceback]")

    # Patch get_model_provider and get_model_endpoint to dummy values
    monkeypatch.setattr(qdrant_mod, "get_model_provider", lambda t: "OpenAI")
    monkeypatch.setattr(qdrant_mod, "get_model_endpoint", lambda t: "dummy-model-endpoint")

    # Patch embeddings modules with dummy embedding object
    import types
    fake_embeddings_module = types.ModuleType("embeddings")
    fake_openai_embeddings_module = types.ModuleType("openai_embeddings")
    fake_openai_embeddings_module.openai_embeds = lambda model_name, emb_type: huggingface_embedding
    fake_embeddings_module.openai_embeddings = fake_openai_embeddings_module
    sys.modules["embeddings"] = fake_embeddings_module
    sys.modules["embeddings.openai_embeddings"] = fake_openai_embeddings_module

    # Dummy Qdrant client mock to prevent real network calls
    class FakeQdrantClient:
        def __init__(self, url, api_key=None):
            print(f"[FAKE QDRANT CLIENT] Initialized with URL: {url}, API key: {api_key}")

    # Patch qdrant_client.QdrantClient to dummy class
    monkeypatch.setattr(qdrant_mod.qdrant_client, "QdrantClient", FakeQdrantClient)

    return {
        "LOGGER_NAME": logger,
        "MONGO_COLLECTION": FakeMongoCollection(),
        "EMBEDDING_MODEL": huggingface_embedding,
    }


def test_create_embeddings_with_qdrant(env_setup):
    dummy_docs = [
        {
            "id": "doc1",
            "text": "This is the first complex document.",
            "metadata": {"author": "Alice", "tags": ["complex", "test", "embedding"], "rating": 4.5},
            "sections": [
                {"header": "Introduction", "content": "Intro content here."},
                {"header": "Body", "content": "Detailed explanation in body."}
            ],
        },
        {
            "id": "doc2",
            "text": "Second document with different structure.",
            "metadata": {"author": "Bob", "tags": ["example", "qdrant"], "rating": 3.9},
            "sections": [
                {"header": "Summary", "content": "Summary content."},
                {"header": "Details", "content": "More detailed content."}
            ],
        },
    ]

    def flatten_doc_text(doc):
        sections_text = " ".join(s.get("content", "") for s in doc.get("sections", []))
        combined_text = f"{doc.get('text', '')} {sections_text}"
        return combined_text

    texts_for_embedding = [flatten_doc_text(d) for d in dummy_docs]

    # Embed texts using HuggingFaceEmbeddings
    embeddings = env_setup["EMBEDDING_MODEL"].embed_documents(texts_for_embedding)

    print("[EMBEDDINGS OUTPUT]")
    for i, emb in enumerate(embeddings):
        print(f"Doc {i+1} embedding (length={len(emb)}): {emb[:5]}...")
        
    print("\n=== Running test_create_embeddings_with_qdrant ===")
    qdrant_mod.create_embeddings_with_qdrant(
        dummy_docs,
        env_setup["EMBEDDING_MODEL"],
        env_setup["LOGGER_NAME"],
        env_setup["MONGO_COLLECTION"],
    )
    print("=== Finished test_create_embeddings_with_qdrant ===\n")


def test_load_qdrantdb(env_setup):
    print("\n=== Running test_load_qdrantdb ===")
    vector_db = qdrant_mod.load_qdrantdb(env_setup["LOGGER_NAME"])
    print("Loaded Qdrant Vector DB instance:", vector_db)
    print("=== Finished test_load_qdrantdb ===\n")

    assert vector_db is not None


def test_load_qdrantdb_with_custom_collection(env_setup):
    print("\n=== Running test_load_qdrantdb with custom collection ===")
    vector_db = qdrant_mod.load_qdrantdb(env_setup["LOGGER_NAME"], col_name="custom_collection")
    print("Loaded Qdrant Vector DB instance with custom collection:", vector_db)
    print("=== Finished test_load_qdrantdb with custom collection ===\n")

    assert vector_db.kwargs.get("collection_name") == "custom_collection"
