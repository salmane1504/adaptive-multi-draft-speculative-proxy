from collections import deque
from pydantic import BaseModel
from typing import Any, Dict, Optional

from src.core.classifier import Domain, PromptClassifier


class RouteConfig(BaseModel):
    domain: Domain
    target_model: str
    draft_model: Optional[str]
    k_drafts: int
    bypass_speculation: bool

class DomainStats:
    def __init__(self, window_size: int = 50, initial_k: int = 5):
        self.history: deque = deque(maxlen=window_size)
        self.current_k: int = initial_k

    def add_sample(self, accepted_tokens: int, drafted_tokens: int):
        if drafted_tokens > 0:
            ratio = accepted_tokens / drafted_tokens
            self.history.append(ratio)
            self._adjust_k()

    @property
    def acceptance_rate(self) -> float:
        if not self.history:
            return 0.5  # Default baseline assumption
        return sum(self.history) / len(self.history)

    def _adjust_k(self):
        alpha = self.acceptance_rate
        if alpha >= 0.65:
            self.current_k = min(self.current_k + 1, 8)  # Cap max depth at 8
        elif alpha < 0.35:
            self.current_k = max(self.current_k - 1, 0)  # Lower depth or disable if overhead > benefit

class DynamicRouter:
    def __init__(self, target_model: str):
        self.target_model = target_model
        self.classifier = PromptClassifier()
        
        # Domain -> Draft Model mappings (Tokenizer family must match target!)
        self.draft_registry = {
            Domain.CODE: "qwen2.5-coder-0.5b",
            Domain.JSON: "qwen2.5-0.5b-instruct",
            Domain.PROSE: "qwen2.5-0.5b-instruct",
        }
        
        # Performance trackers per domain
        self.stats = {
            Domain.CODE: DomainStats(initial_k=5),
            Domain.JSON: DomainStats(initial_k=6),
            Domain.PROSE: DomainStats(initial_k=3),
        }

    def route(self, payload: Dict[str, Any]) -> RouteConfig:
        domain = self.classifier.classify(payload)
        domain_stat = self.stats[domain]
        draft_model = self.draft_registry.get(domain)
        
        # Bypass speculative decoding if acceptance rate drops critically low (< 0.20)
        # or if K has been adjusted down to 0
        bypass = (domain_stat.acceptance_rate < 0.20) or (domain_stat.current_k == 0)

        return RouteConfig(
            domain=domain,
            target_model=self.target_model,
            draft_model=None if bypass else draft_model,
            k_drafts=0 if bypass else domain_stat.current_k,
            bypass_speculation=bypass
        )

    def record_execution_feedback(self, domain: Domain, accepted_tokens: int, drafted_tokens: int):
        """Called by the inference engine after token generation finishes."""
        self.stats[domain].add_sample(accepted_tokens, drafted_tokens)