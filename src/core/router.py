from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from src.core.classifier import Domain, PromptClassifier


class RouteConfig(BaseModel):
    domain: Domain
    target_model: str
    draft_model: Optional[str]
    k_drafts: int
    bypass_speculation: bool

class ControllerSettings(BaseModel):
    window_size: int = Field(gt=0)
    max_k: int = Field(ge=1)
    increase_above: float = Field(ge=0, le=1)
    decrease_below: float = Field(ge=0, le=1)
    disable_below: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> "ControllerSettings":
        if not self.disable_below <= self.decrease_below < self.increase_above:
            raise ValueError("need disable_below <= decrease_below < increase_above")
        return self

class DomainSettings(BaseModel):
    draft_model: str
    initial_k: int = Field(ge=0)

class RoutingSettings(BaseModel):
    target_model: str
    controller: ControllerSettings
    domains: dict[Domain, DomainSettings]

    @model_validator(mode="after")
    def _validate_domains(self) -> "RoutingSettings":
        missing = set(Domain) - set(self.domains)
        if missing:
            raise ValueError(f"routing config missing domains: {sorted(d.value for d in missing)}")
        for domain, settings in self.domains.items():
            if settings.initial_k > self.controller.max_k:
                raise ValueError(
                    f"{domain.value}.initial_k={settings.initial_k} exceeds controller.max_k={self.controller.max_k}"
                )
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "RoutingSettings":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

class DomainStats:
    def __init__(self, initial_k: int, controller: ControllerSettings):
        self.controller = controller
        self.history: deque = deque(maxlen=controller.window_size)
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

    @property
    def speculation_enabled(self) -> bool:
        return self.acceptance_rate >= self.controller.disable_below and self.current_k > 0

    def _adjust_k(self):
        alpha = self.acceptance_rate
        if alpha > self.controller.increase_above:
            self.current_k = min(self.current_k + 1, self.controller.max_k)
        elif alpha < self.controller.decrease_below:
            self.current_k = max(self.current_k - 1, 0)  # Lower depth or disable if overhead > benefit

class DynamicRouter:
    def __init__(
        self,
        target_model: str,
        draft_registry: Dict[Domain, str],
        initial_k: Dict[Domain, int],
        controller: ControllerSettings,
    ):
        self.target_model = target_model
        self.classifier = PromptClassifier()

        # Domain -> Draft Model mappings (Tokenizer family must match target!)
        self.draft_registry = draft_registry

        # Performance trackers per domain
        self.stats = {domain: DomainStats(initial_k[domain], controller) for domain in Domain}

    @classmethod
    def from_settings(cls, settings: RoutingSettings) -> "DynamicRouter":
        return cls(
            settings.target_model,
            draft_registry={d: s.draft_model for d, s in settings.domains.items()},
            initial_k={d: s.initial_k for d, s in settings.domains.items()},
            controller=settings.controller,
        )

    def route(self, payload: Dict[str, Any]) -> RouteConfig:
        domain = self.classifier.classify(payload)
        domain_stat = self.stats[domain]
        draft_model = self.draft_registry.get(domain)
        
        # Bypass speculative decoding if acceptance rate drops below controller.disable_below
        # or if K has been adjusted down to 0
        bypass = not domain_stat.speculation_enabled

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