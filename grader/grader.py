WEIGHTS = {"classification": 0.3, "priority": 0.3, "reply": 0.4}


def grade(results: dict) -> float:
    """
    Accepts float accuracies OR bools.
    Returns float strictly between 0 and 1.
    """
    required = {"classification", "priority", "reply"}
    missing = required - results.keys()
    if missing:
        raise ValueError(f"Missing keys: {missing}")

    score = 0.0
    for key, weight in WEIGHTS.items():
        val = results[key]
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        score += float(val) * weight

    # Clamp strictly between 0 and 1
    return round(min(max(score, 0.01), 0.99), 4)
