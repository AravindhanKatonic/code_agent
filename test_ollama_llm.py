import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

import routes.models.ollama_llm as ollama_mod

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

    class FakeChatOllama:
        def __init__(self, base_url, model):
            print(f"[FAKE OLLAMA INIT] Base URL: {base_url}, Model: {model}")
            self.base_url = base_url
            self.model = model
        def generate(self, prompt):
            print(f"[FAKE OLLAMA GENERATE] Prompt: {prompt}")
            return "dummy response"
    monkeypatch.setattr(ollama_mod, "ChatOllama", FakeChatOllama)

    return {"LOGGER_NAME": logger}

def test_ollama_completion_success(env_setup):
    print("\n=== Running test_ollama_completion_success ===")
    llm = ollama_mod.ollama_completion("dummy_service", "dummy_model", env_setup["LOGGER_NAME"])
    assert llm is not None
    assert hasattr(llm, "generate")
    print("LLM instance returned:", llm)
    print("=== Finished test_ollama_completion_success ===\n")

def test_ollama_completion_exception(monkeypatch, env_setup):

    def fake_init_raise(*args, **kwargs):
        raise ValueError("Forced error")

    monkeypatch.setattr(ollama_mod, "ChatOllama", fake_init_raise)

    print("\n=== Running test_ollama_completion_exception ===")
    result = ollama_mod.ollama_completion("dummy_service", "dummy_model", env_setup["LOGGER_NAME"])
    assert isinstance(result, str)
    assert "Forced error" in result
    print("Returned error string:", result)
    print("=== Finished test_ollama_completion_exception ===\n")
