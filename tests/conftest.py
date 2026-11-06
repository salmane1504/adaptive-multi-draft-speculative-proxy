from src.core.classifier import Domain
from src.core.router import (
    ControllerSettings,
    DomainSettings,
    DynamicRouter,
    RoutingSettings,
)

CODE_DRAFT = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-8bit"
SMALL_DRAFT = "mlx-community/Qwen2.5-0.5B-Instruct-8bit"


def make_router(target_model: str = "target") -> DynamicRouter:
    """Fixed settings so tests do not depend on config/routing.yaml tuning."""
    settings = RoutingSettings(
        target_model=target_model,
        controller=ControllerSettings(
            window_size=50,
            max_k=8,
            increase_above=0.6,
            decrease_below=0.35,
            disable_below=0.20,
        ),
        domains={
            Domain.CODE: DomainSettings(draft_model=CODE_DRAFT, initial_k=5),
            Domain.JSON: DomainSettings(draft_model=SMALL_DRAFT, initial_k=6),
            Domain.PROSE: DomainSettings(draft_model=SMALL_DRAFT, initial_k=3),
        },
    )
    return DynamicRouter.from_settings(settings)
