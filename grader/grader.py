WEIGHTS = {
    "classification": 0.3,
    "priority": 0.3,
    "reply": 0.4,
}


def grade(results: dict) -> float:
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

    return round(score, 10)