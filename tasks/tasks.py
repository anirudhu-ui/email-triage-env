"""
tasks/tasks.py
--------------
Task definitions for EmailTriageEnv.

Three difficulty levels:
    easy   — classification only (evaluated)
    medium — classification + priority
    hard   — full pipeline (classification + priority + reply)
"""

from typing import List


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = {
    "easy": {
        "id": "task_easy",
        "name": "Email Classification",
        "difficulty": "easy",
        "description": (
            "Classify each email into one of three categories: "
            "'spam', 'important', or 'promotional'. "
            "Only the classification phase is evaluated."
        ),
        "phases": ["classification"],
        "evaluation_phases": ["classification"],  # ✅ added for clarity
        "reward_breakdown": {
            "classification_correct": 2.0,
            "classification_wrong": -1.0,
        },
        "max_reward": 2.0,
    },

    "medium": {
        "id": "task_medium",
        "name": "Classification + Priority",
        "difficulty": "medium",
        "description": (
            "Classify each email AND assign the correct priority level "
            "('low', 'medium', or 'high'). "
            "Both classification and priority phases are evaluated."
        ),
        "phases": ["classification", "priority"],
        "evaluation_phases": ["classification", "priority"],  # ✅ added
        "reward_breakdown": {
            "classification_correct": 2.0,
            "classification_wrong": -1.0,
            "priority_correct": 3.0,
            "priority_wrong": -1.0,
        },
        "max_reward": 5.0,
    },

    "hard": {
        "id": "task_hard",
        "name": "Full Triage Pipeline",
        "difficulty": "hard",
        "description": (
            "Run the full three-phase triage pipeline for every email: "
            "classify it, assign a priority, and decide the correct reply action "
            "('ignore', 'acknowledge', or 'respond'). "
            "All three phases are evaluated."
        ),
        "phases": ["classification", "priority", "reply"],
        "evaluation_phases": ["classification", "priority", "reply"],  # ✅ added
        "reward_breakdown": {
            "classification_correct": 2.0,
            "classification_wrong": -1.0,
            "priority_correct": 3.0,
            "priority_wrong": -1.0,
            "reply_correct": 5.0,
            "reply_wrong": -2.0,
        },
        "max_reward": 10.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_task(difficulty: str) -> dict:
    """
    Return the task definition for the given difficulty level.

    Args:
        difficulty: one of 'easy', 'medium', 'hard'

    Returns:
        dict with task metadata

    Raises:
        ValueError: if difficulty is not recognised
    """
    if difficulty not in TASKS:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. "
            f"Choose from: {list(TASKS.keys())}"
        )
    return TASKS[difficulty]


def list_tasks() -> List[dict]:
    """Return all task definitions as a list."""
    return list(TASKS.values())


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for task in list_tasks():
        print(f"[{task['difficulty'].upper()}] {task['name']}")
        print(f"  Phases             : {task['phases']}")
        print(f"  Evaluation Phases  : {task['evaluation_phases']}")
        print(f"  Max reward/email   : {task['max_reward']}")
        print(f"  {task['description']}")
        print()