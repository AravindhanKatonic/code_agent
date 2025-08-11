# test_mongo_init_tempdb.py

import os
import sys
import pytest
import pandas as pd
import mongomock
from unittest.mock import patch

os.environ["EMBEDDING_SERVICE_TYPE"] = "embeddingModel"
os.environ["SERVICE_TYPE"] = "llmModel"

sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot/routes/utilities"))
import mongo_init

@pytest.fixture(scope="module", autouse=True)
def setup_mongo_mock():
    with patch("mongo_init.MongoClient", new=mongomock.MongoClient):
        client = mongo_init.MongoClient()
        temp_db = client["testdb"]

        # Insert test data
        temp_db[mongo_init.GENERAL_SETTINGS].insert_one({"setting1": "value1", "setting2": "value2"})
        temp_db[mongo_init.FM_META_COLLECTION].insert_many([
            {"modelName": "embeddingModel", "parent": "katonic", "metadata": {"endpoint": "http://embedding-endpoint"}},
            {"modelName": "llmModel", "parent": "openai", "metadata": {"endpoint": "http://llm-endpoint"}},
            {"modelName": "katonicLLM", "value": "katonicLLM", "parent": "katonic", "metadata": {"endpoint": "http://katonic-llm"}}
        ])
        temp_db[mongo_init.ORGANIZATIONAL_POLICIES].insert_many([
            {"active": True, "policyDescription": "Policy 1"},
            {"active": False, "policyDescription": "Policy 2"}
        ])
        temp_db[mongo_init.MESSAGE_COLLECTION].insert_one({"msg": "hello"})

        # Print DB and collections used
        print(f"\nSetup temp DB: {temp_db.name}")
        print(f"\nCollections in DB: {temp_db.list_collection_names()}")

        with patch("mongo_init.get_local_mongo_db", return_value=temp_db):
            yield

def test_get_local_mongo_db():
    db = mongo_init.get_local_mongo_db()
    print(f"\nDB in test_get_local_mongo_db: {db.name}")
    assert db.name == "testdb"

def test_get_general_settings():
    settings = mongo_init.get_general_settings()
    print(f"\nGeneral settings keys: {list(settings.keys())}")
    assert settings["setting1"] == "value1"

def test_get_local_mongo_cost_collection():
    collection = mongo_init.get_local_mongo_cost_collection()
    print(f"Cost collection name: {collection.name}")
    assert collection.name == mongo_init.FM_META_COLLECTION

def test_get_local_mongo_logs_collection():
    collection = mongo_init.get_local_mongo_logs_collection()
    print(f"Logs collection name: {collection.name}")
    assert collection.name == mongo_init.LOGS_COLLECTION_NAME

def test_get_local_mongo_embedding_meta():
    df = mongo_init.get_local_mongo_embedding_meta()
    print(f"Embedding meta dataframe:\n{df}")
    assert not df.empty
    assert df.iloc[0]["modelName"] == "embeddingModel"

def test_get_local_mongo_llm_meta():
    df = mongo_init.get_local_mongo_llm_meta("llmModel")
    print(f"LLM meta dataframe:\n{df}")
    assert not df.empty
    assert df.iloc[0]["modelName"] == "llmModel"

def test_check_for_mongo_existance_katonicLLM():
    df = mongo_init.check_for_mongo_existance("katonicLLM")
    print(f"Check katonicLLM existence:\n{df}")
    assert not df.empty
    assert df.iloc[0]["value"] == "katonicLLM"

def test_check_for_mongo_existance_other():
    df = mongo_init.check_for_mongo_existance("llmModel")
    print(f"Check llmModel existence:\n{df}")
    assert not df.empty
    assert df.iloc[0]["modelName"] == "llmModel"

def test_get_policy_information():
    policies = mongo_init.get_policy_information()
    print(f"Policies returned:\n{list(policies)}")
    assert "Policy 1" in list(policies)

def test_get_message_collection():
    collection = mongo_init.get_message_collection()
    print(f"Message collection name: {collection.name}")
    assert collection.name == mongo_init.MESSAGE_COLLECTION

def test_get_model_provider():
    provider_llm = mongo_init.get_model_provider("llmModel")
    provider_katonic = mongo_init.get_model_provider("katonicLLM")
    print(f"Provider for llmModel: {provider_llm}")
    print(f"Provider for katonicLLM: {provider_katonic}")
    assert provider_llm == "openai"
    assert provider_katonic == "katonic"

def test_get_model_endpoint():
    endpoint = mongo_init.get_model_endpoint("llmModel")
    print(f"Endpoint for llmModel: {endpoint}")
    assert endpoint == "http://llm-endpoint"
