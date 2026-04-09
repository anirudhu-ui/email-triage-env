"""
grader/grader.py
----------------
Deterministic grader for EmailTriageEnv.

Accepts float accuracy scores per phase and returns a weighted score
strictly between 0 and 1.

Weights: classification=0.3, priority=0.3, reply=0.4
"""

WEIGHTS = {"classification": 0.3, "priority": 0.3, "reply": 0.4}


def grade(results: dict) -> float:
    """
    Input:  {"classification": float, "priority": float, "reply": float}
            Each value should be an accuracy score in [0.0, 1.0]
    Output: float strictly between 0 and 1 (clamped to 0.01-0.99)
    """
    required = {"classification", "priority", "reply"}
    missing = required - results.keys()
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    score = 0.0
    for key, weight in WEIGHTS.items():
        val = results[key]
        # Accept both bool and float
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        try:
            val = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert '{key}' value to float: {val}")
        val = min(max(val, 0.0), 1.0)
        score += val * weight

    # Clamp final score strictly between 0 and 1
    score = min(max(score, 0.01), 0.99)
    return round(score, 4)


if __name__ == "__main__":
    tests = [
        {"classification": 0.9,  "priority": 0.8,  "reply": 0.7},
        {"classification": 0.5,  "priority": 0.4,  "reply": 0.6},
        {"classification": 0.0,  "priority": 0.0,  "reply": 0.0},
        {"classification": 1.0,  "priority": 1.0,  "reply": 1.0},
        {"classification": True, "priority": True,  "reply": False},
    ]
    for inp in tests:
        r = grade(inp)
        assert 0.0 < r < 1.0, f"FAIL: score {r} not strictly in (0,1) for {inp}"
        print(f"PASS  score={r}  input={inp}")
