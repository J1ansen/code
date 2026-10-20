"""Utility-supervised heterophily expert prompt router.

This module is the minimal mainline variant for testing whether explicit
heterophily expert utility can make P20 prompts learn a transferable pattern
choice.  It intentionally reuses the class-conditioned bounded expert router
and changes the training contract in the runner: expert messages expose a
direct oracle/teacher over reject, ego, high-pass, two-hop, compatibility and
role corrections.
"""

from __future__ import annotations

from typing import Any

from models.class_conditioned_pattern_prompt_router import ClassConditionedPatternPromptRouter


class UtilitySupervisedPatternPromptRouter(ClassConditionedPatternPromptRouter):
    """P20 utility-supervised variant.

    The forward contract is inherited from ``ClassConditionedPatternPromptRouter``:
    prompts are functional bounded expert messages applied only to the adapted
    branch, and val/test nodes only soft-read class-conditioned prompts.
    """

    consumes_base_logits = True

    def __init__(self, source_dim: int, hidden_dim: int, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        cfg.setdefault("module_type", "utility_supervised_pattern_router")
        cfg.setdefault("num_patterns", 6)
        super().__init__(source_dim, hidden_dim, cfg)
