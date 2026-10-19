import json
import traceback

from app.agent.adaptive_mentor import (
    OllamaJsonModel,
    MentorPlanner,
    Decision,
)


def main():
    print("[1] Initialize Qwen3")

    model = OllamaJsonModel(
        model="qwen3:4b"
    )

    print("[2] Test structured JSON")

    result = model.decide(
        "You are a JSON decision generator. "
        "Return only a valid decision object.",
        "Reply to the user with a short greeting. "
        "Choose action=respond. "
        "Include all required schema fields.",
    )

    decision = Decision.model_validate(result)

    print(
        json.dumps(
            decision.model_dump(),
            ensure_ascii=True,
            indent=2,
        )
    )

    assert decision.action == "respond"

    print("[3] Check planner model binding")

    planner = MentorPlanner(model)

    assert planner.model is model

    print("[OK] Qwen3 model integration works")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
