
"""系统能力的声明、绑定与执行。"""

from collections.abc import Callable
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ============================================
# 1. 定义系统能力
# ============================================

class CapabilitySpec(BaseModel):

    model_config = ConfigDict(extra="forbid")

    capability_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


# ============================================
# 2. 能力注册表
# ============================================

class CapabilityRegistry:

    def __init__(self):

        self._specs: dict[
            str, CapabilitySpec
        ] = {}

        self._handlers: dict[
            str, Callable[..., Any]
        ] = {}

    # ========================================
    # 3. 声明新能力
    # ========================================

    def register(
        self,
        spec: CapabilitySpec,
        handler: Callable[..., Any] | None = None,
    ) -> None:

        if spec.capability_id in self._specs:
            raise ValueError(
                f"重复的能力：{spec.capability_id}"
            )

        if (
            handler is not None
            and not callable(handler)
        ):
            raise TypeError(
                "handler必须是可调用对象"
            )

        self._specs[spec.capability_id] = spec

        if handler is not None:
            self._handlers[spec.capability_id] = handler

    # ========================================
    # 4. 为已经声明的能力绑定真实函数
    # ========================================

    def bind(
        self,
        capability_id: str,
        handler: Callable[..., Any],
    ) -> None:

        if capability_id not in self._specs:
            raise KeyError(
                f"能力尚未声明：{capability_id}"
            )

        if not callable(handler):
            raise TypeError(
                "handler必须是可调用对象"
            )

        if capability_id in self._handlers:
            raise ValueError(
                f"能力已经绑定：{capability_id}"
            )

        self._handlers[capability_id] = handler

    # ========================================
    # 5. 判断能力是否已经绑定
    # ========================================

    def is_bound(
        self,
        capability_id: str,
    ) -> bool:

        return capability_id in self._handlers

    # ========================================
    # 6. 执行能力
    # ========================================

    def invoke(
        self,
        capability_id: str,
        **kwargs: Any,
    ) -> Any:

        if capability_id not in self._specs:
            raise KeyError(
                f"不存在的能力：{capability_id}"
            )

        if not self.is_bound(capability_id):
            raise RuntimeError(
                "能力尚未连接真实实现："
                f"{capability_id}"
            )

        return self._handlers[
            capability_id
        ](**kwargs)

    # ========================================
    # 7. 列出所有能力
    # ========================================

    def all(self) -> list[CapabilitySpec]:

        return list(self._specs.values())


# ============================================
# 8. 创建默认能力注册表
# ============================================

def build_capability_registry() -> CapabilityRegistry:

    registry = CapabilityRegistry()

    registry.register(
        CapabilitySpec(
            capability_id="knowledge.retrieve",
            description=(
                "检索本地教材与史料，"
                "返回知识候选及其来源数据"
            ),
        )
    )

    registry.register(
        CapabilitySpec(
            capability_id="memory.read",
            description=(
                "读取用户授权的长期成长记录"
            ),
        )
    )

    registry.register(
        CapabilitySpec(
            capability_id="evidence.validate",
            description=(
                "核查证据引用、来源和假设关联"
            ),
        )
    )

    return registry
