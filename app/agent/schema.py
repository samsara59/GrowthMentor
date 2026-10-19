
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RouteDecision(BaseModel):
    """LLM输出的任务选择结果。"""

    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "knowledge",
        "start_exploration",
        "experience_reply",
        "other",
    ]

    reason: str = Field(min_length=1)
