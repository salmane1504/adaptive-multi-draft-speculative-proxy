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
			"mlx-community/Qwen2.5-Coder-0.5B-Instruct-8bit",
			5,
		),
		(
			{"response_format": {"type": "json_object"}, "messages": []},
			Domain.JSON,
			"mlx-community/Qwen2.5-0.5B-Instruct-8bit",
			6,
		),
		(
			{"messages": [{"role": "user", "content": "Explain the seasons."}]},
			Domain.PROSE,
			"mlx-community/Qwen2.5-0.5B-Instruct-8bit",
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


@pytest.mark.parametrize(
	("accepted", "drafted", "expected_k"),
	[(7, 10, 6), (6, 10, 5), (35, 100, 5), (34, 100, 4)],
)
def test_step_controller_thresholds(router, accepted, drafted, expected_k):
	# alpha > 0.6 -> K+1, alpha < 0.35 -> K-1, otherwise hold.
	router.record_execution_feedback(Domain.CODE, accepted_tokens=accepted, drafted_tokens=drafted)

	assert router.stats[Domain.CODE].current_k == expected_k


def test_controller_clamps_k_between_zero_and_eight(router):
	for _ in range(10):
		router.record_execution_feedback(Domain.CODE, accepted_tokens=10, drafted_tokens=10)
	assert router.stats[Domain.CODE].current_k == 8

	for _ in range(60):
		router.record_execution_feedback(Domain.JSON, accepted_tokens=3, drafted_tokens=10)
	assert router.stats[Domain.JSON].current_k == 0
	assert router.stats[Domain.JSON].speculation_enabled is False