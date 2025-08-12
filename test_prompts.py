import sys
import os

sys.path.insert(0, os.path.abspath("katonic-converse/smartchatcopilot"))

from routes.utilities import prompts
import pytest
from unittest.mock import patch, MagicMock


def test_question_validation_prompt_enabled_and_prompt():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "agents": [
                {
                    "metadata": {
                        "questionValidationPrompt": {
                            "enable": True,
                            "prompt": "Custom question validation prompt."
                        }
                    }
                }
            ]
        }
    }
    with patch("routes.utilities.prompts.requests.post", return_value=mock_response):
        result = prompts.question_validation_prompt("some_deployment_id")
        print(">>> question_validation_prompt result:", result)
        assert result == "Custom question validation prompt."


def test_question_validation_prompt_disabled_uses_default():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "agents": [
                {
                    "metadata": {
                        "questionValidationPrompt": {
                            "enable": False,
                            "prompt": ""
                        }
                    }
                }
            ]
        }
    }
    with patch("routes.utilities.prompts.requests.post", return_value=mock_response):
        result = prompts.question_validation_prompt("some_deployment_id")
        print(">>> question_validation_prompt default result starts with:", result[:30])
        assert "You are a helper tool" in result


def test_question_validation_prompt_exception_logs_and_returns_error():
    # Mock requests.post to raise Exception
    with patch("routes.utilities.prompts.requests.post", side_effect=Exception("Network error")):
        result = prompts.question_validation_prompt("some_deployment_id")
        print(">>> question_validation_prompt exception result:", result)
        assert "Network error" in result


def test_answer_validation_prompt_enabled_and_prompt():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "agents": [
                {
                    "metadata": {
                        "answerValidationPrompt": {
                            "enable": True,
                            "prompt": "Custom answer validation prompt with {QUESTION_PAT} and {ANSWER_PAT}",
                            "model": "test_model"
                        }
                    }
                }
            ]
        }
    }
    with patch("routes.utilities.prompts.requests.post", return_value=mock_response):
        result = prompts.answer_validation_prompt("some_deployment_id")
        print(">>> answer_validation_prompt result:", result)
        assert "Custom answer validation prompt with Query: and Answer:" in result


def test_answer_validation_prompt_disabled_uses_default():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "agents": [
                {
                    "metadata": {
                        "answerValidationPrompt": {
                            "enable": False,
                            "prompt": ""
                        }
                    }
                }
            ]
        }
    }
    with patch("routes.utilities.prompts.requests.post", return_value=mock_response):
        result = prompts.answer_validation_prompt("some_deployment_id")
        print(">>> answer_validation_prompt default result starts with:", result[:30])
        assert "You are an assistant to identify invalid" in result


def test_answer_validation_prompt_exception_logs_and_returns_error():
    with patch("routes.utilities.prompts.requests.post", side_effect=Exception("Network error")):
        result = prompts.answer_validation_prompt("some_deployment_id")
        print(">>> answer_validation_prompt exception result:", result)
        assert "Network error" in result


def test_handle_full_prompt_basic_and_with_history():
    query = "What is AI?"
    history = []
    result = prompts.handle_full_prompt(query, history)
    print(">>> handle_full_prompt without history result:", result)
    assert "Question: What is AI?" in result

    history_text = "Previous chat history..."
    result_with_history = prompts.handle_full_prompt(query, [history_text])
    print(">>> handle_full_prompt with history result:", result_with_history)
    assert "Chat History" in result_with_history
    assert history_text in result_with_history
