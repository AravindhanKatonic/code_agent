import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv() 

sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

import routes.embeddings.ollama_embeddings as ollama_mod  

@pytest.fixture(scope="function")
def env_setup(monkeypatch):
    class CallableLogger:
        def __call__(self, *args, **kwargs):
            print("[LOG CALL]", args)
        def info(self, msg, *args, **kwargs):
            print("[LOG INFO]", msg)
        def error(self, msg, *args, **kwargs):
            print("[LOG ERROR]", msg)

    logger = CallableLogger()

    class FakeOllamaEmbeddings:
        def __init__(self, base_url, model):
            print(f"[FAKE OLLAMA EMBEDS INIT] Base URL: {base_url}, Model: {model}")
            if base_url == "raise_exception":
                raise ValueError("Forced error")
        def embed_documents(self, docs):
            return [[0.1, 0.2, 0.3] for _ in docs]
        def embed_query(self, text):
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(ollama_mod, "OllamaEmbeddings", FakeOllamaEmbeddings)

    return {"LOGGER_NAME": logger}


def test_ollama_embeds_success(env_setup):
    print("\n=== Running test_ollama_embeds_success ===")
    embeddings = ollama_mod.ollama_embeds("dummy-model")
    print("Embeddings instance returned:", embeddings)
    assert embeddings is not None
    print("=== Finished test_ollama_embeds_success ===\n")


def test_ollama_embeds_exception(monkeypatch, env_setup):
    print("\n=== Running test_ollama_embeds_exception ===")
    
    monkeypatch.setattr(ollama_mod, "OllamaEmbeddings", lambda base_url, model: (_ for _ in ()).throw(ValueError("Forced error")))

    with pytest.raises(ValueError) as e:
        ollama_mod.ollama_embeds("dummy-model")
    print("Caught exception as expected:", e.value)
    print("=== Finished test_ollama_embeds_exception ===\n")
