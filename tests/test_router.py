import pytest

from src.core.classifier import Domain
from src.core.router import DynamicRouter


@pytest.fixture
def router():
	return DynamicRouter(target_model="target-model")


@pytest.mark.parametrize(
	("payload", "domain", "draft_model", "k_drafts"),
	[
		(
			{
				"messages": [
					{
						"role": "user",
						"content": "Write a Python function that adds two numbers.",
					}
				]
			},
			Domain.CODE,
			"qwen2.5-coder-0.5b",
			5,
		),
		(
			{"response_format": {"type": "json_object"}, "messages": []},
			Domain.JSON,
			"qwen2.5-0.5b-instruct",
			6,
		),
		(
			{"messages": [{"role": "user", "content": "Explain the seasons."}]},
			Domain.PROSE,
			"qwen2.5-0.5b-instruct",
			3,
		),
	],
)
def test_route_selects_domain_configuration(router, payload, domain, draft_model, k_drafts):
	route = router.route(payload)

	assert route.domain is domain
	assert route.target_model == "target-model"
	assert route.draft_model == draft_model
	assert route.k_drafts == k_drafts
	assert route.bypass_speculation is False


def test_high_acceptance_increases_draft_depth(router):
	router.record_execution_feedback(Domain.CODE, accepted_tokens=7, drafted_tokens=10)

	route = router.route(
		{
			"messages": [
				{
					"role": "user",
					"content": "Write a Python function that adds two numbers.",
				}
			]
		}
	)

	assert route.k_drafts == 6


def test_low_acceptance_bypasses_speculation(router):
	router.record_execution_feedback(Domain.PROSE, accepted_tokens=1, drafted_tokens=10)

	route = router.route(
		{"messages": [{"role": "user", "content": "Explain the seasons."}]}
	)

	assert route.bypass_speculation is True
	assert route.draft_model is None
	assert route.k_drafts == 0


def test_zero_drafted_tokens_do_not_change_domain_stats(router):
	stats = router.stats[Domain.CODE]

	router.record_execution_feedback(Domain.CODE, accepted_tokens=0, drafted_tokens=0)

	assert not stats.history
	assert stats.current_k == 5