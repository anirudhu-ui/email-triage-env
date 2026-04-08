"""
grader/grader.py
----------------
Deterministic grader for the EmailTriageEnv.

Input:
    {
        "classification": bool,
        "priority":       bool,
        "reply":          bool
    }

Output:
    float between 0.0 and 1.0

Weights:
    classification = 0.3
    priority       = 0.3
    reply          = 0.4
"""

WEIGHTS = {
    "classification": 0.3,
    "priority": 0.3,
    "reply": 0.4,
}


def grade(results: dict) -> float:
    """
    Score the agent's performance across all three phases.

    Args:
        results: dict with boolean values for keys
                 'classification', 'priority', 'reply'

    Returns:
        float in [0.0, 1.0]

    Raises:
        ValueError: if any required key is missing or not a bool
    """
    required_keys = {"classification", "priority", "reply"}
    missing = required_keys - results.keys()
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    score = 0.0
    for key, weight in WEIGHTS.items():
        value = results[key]
        if not isinstance(value, bool):
            raise ValueError(f"Expected bool for '{key}', got {type(value).__name__}")
        if value:
            score += weight

    # Round to avoid floating-point noise (e.g. 0.30000000000000004)
    return round(score, 10)


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        ({"classification": True,  "priority": True,  "reply": True},  1.0),
        ({"classification": True,  "priority": True,  "reply": False}, 0.6),
        ({"classification": True,  "priority": False, "reply": False}, 0.3),
        ({"classification": False, "priority": False, "reply": False}, 0.0),
        ({"classification": False, "priority": False, "reply": True},  0.4),
    ]

    all_passed = True
    for inputs, expected in test_cases:
        result = grade(inputs)
        status = "✅ PASS" if abs(result - expected) < 1e-9 else "❌ FAIL"
        if status == "❌ FAIL":
            all_passed = False
        print(f"{status}  grade({inputs}) = {result}  (expected {expected})")

    print()
    print("All tests passed ✅" if all_passed else "Some tests FAILED ❌")
