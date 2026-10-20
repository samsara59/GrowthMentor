"""
Organizational tree for the GrowthMentor brain.

This describes responsibilities, not execution order.
"""

BRAIN_TREE = {
    "supervisor": {
        "exploration": [
            "interview",
            "assessment",
        ],
        "knowledge": [
            "retrieval",
            "source_verification",
        ],
        "reasoning": [
            "evidence",
            "hypothesis_review",
        ],
        "action": [
            "experiment",
            "practice_review",
        ],
    }
}


def validate_hierarchy(tree=BRAIN_TREE):
    supervisor = tree["supervisor"]
    children = [
        agent
        for branch in supervisor.values()
        for agent in branch
    ]

    if len(children) != len(set(children)):
        raise ValueError("Duplicate agent identities")

    return True
