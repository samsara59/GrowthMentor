
"""检查GrowthMentor-Agent的新架构目录。"""

from app.programs.catalog import (
    build_program_catalog,
)

from app.methods.catalog import (
    build_method_catalog,
)

from app.capabilities.registry import (
    build_capability_registry,
)


def main() -> None:

    programs = build_program_catalog()
    methods = build_method_catalog()
    capabilities = build_capability_registry()

    print("=== GrowthMentor-Agent 架构目录 ===")

    # ========================================
    # 1. 检查成长任务
    # ========================================

    print("\n[成长任务]")

    for program in programs.all():

        print(
            f"- {program.program_id}: "
            f"{program.name} [{program.status}]"
        )

        # 确认任务引用的方法都已注册
        for method_id in program.suggested_method_ids:
            methods.get(method_id)

    # ========================================
    # 2. 检查导师方法
    # ========================================

    print("\n[导师方法]")

    for method in methods.all():

        print(
            f"- {method.method_id}: "
            f"{method.name} [{method.status}]"
        )

    # ========================================
    # 3. 检查底层能力
    # ========================================

    print("\n[系统能力]")

    for capability in capabilities.all():

        bound = capabilities.is_bound(
            capability.capability_id
        )

        state = (
            "已绑定"
            if bound
            else "待绑定"
        )

        print(
            f"- {capability.capability_id}: {state}"
        )

    print(
        "\n架构目录自检完成；"
        "尚未修改旧Graph与真实用户数据。"
    )


if __name__ == "__main__":
    main()
