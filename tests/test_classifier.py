import pytest

from src.core.classifier import Domain, PromptClassifier


@pytest.fixture
def classifier():
	return PromptClassifier()


def test_empty_messages_are_classified_as_prose(classifier):
	assert classifier.classify({"messages": []}) is Domain.PROSE
	assert classifier.classify({"messages": [{"role": "user", "content": "  "}]}) is Domain.PROSE


def test_response_format_json_override_takes_precedence_over_code(classifier):
	payload = {
		"response_format": {"type": "json_object"},
		"messages": [{"role": "user", "content": "Write a Python function."}],
	}

	assert classifier.classify(payload) is Domain.JSON


@pytest.mark.parametrize(
	("system_content", "expected"),
	[
		("You are a JSON assistant.", Domain.JSON),
		("You are a helpful programmer.", Domain.CODE),
	],
)
def test_system_message_sets_domain(classifier, system_content, expected):
	payload = {
		"messages": [
			{"role": "system", "content": system_content},
			{"role": "user", "content": "Tell me something useful."},
		]
	}

	assert classifier.classify(payload) is expected


@pytest.mark.parametrize(
	"content",
	[
		"Please write a Python function that adds two numbers.",
		"```sql\nSELECT name FROM users;\n```",
	],
)
def test_code_patterns_classify_code_requests(classifier, content):
	payload = {"messages": [{"role": "user", "content": content}]}

	assert classifier.classify(payload) is Domain.CODE


def test_json_text_patterns_classify_structured_output_request(classifier):
	payload = {
		"messages": [
			{"role": "user", "content": 'Return JSON with this shape: {"name": "Ada"}'}
		]
	}

	assert classifier.classify(payload) is Domain.JSON


def test_ordinary_request_is_classified_as_prose(classifier):
	payload = {
		"messages": [{"role": "user", "content": "Explain why leaves change color."}]
	}

	assert classifier.classify(payload) is Domain.PROSE
