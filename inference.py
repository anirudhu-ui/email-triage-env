import sys
import os
import argparse

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action
from server.email_triage_env_environment import EmailTriageEnv
from grader.grader import grade


# ---------------------------------------------------------------------------
# Baseline decision logic (SAFE — no API dependency)
# ---------------------------------------------------------------------------

def baseline_action(state) -> Action:
    text = state.email_text.lower()
    phase = state.step

    if phase == "classification":
        value = "spam" if "free" in text else "important"
    elif phase == "priority":
        value = "high" if "meeting" in text else "low"
    elif phase == "reply":
        value = "acknowledge" if "meeting" in text else "ignore"
    else:
        value = "ignore"

    return Action(type=phase, value=value)


# ---------------------------------------------------------------------------
# Ensure score is strictly between 0 and 1
# ---------------------------------------------------------------------------

def safe_score(x: float) -> float:
    return min(max(x, 0.01), 0.99)


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(tier: str = None):
    env = EmailTriageEnv(tier=tier, shuffle=False)
    obs = env.reset()

    print("[START]")

    step_number = 0
    done = False

    while not done:
        step_number += 1
        action = baseline_action(obs)
        obs, reward, done, info = env.step(action)

        print(
            f"[STEP] #{step_number:2d} | phase={action.type:16s} | "
            f"action='{action.value:12s}' | reward={reward.value:+.1f}"
        )

    stats = env.episode_stats()

    grader_input = {
        "classification": safe_score(stats["classification"]["accuracy"]),
        "priority": safe_score(stats["priority"]["accuracy"]),
        "reply": safe_score(stats["reply"]["accuracy"]),
    }

    grader_score = grade(grader_input)

    print("\n[END]")
    print("-" * 50)
    print(f"  Total steps         : {step_number}")
    print(f"  Emails processed    : {stats['emails_processed']}")
    print(f"  Cumulative reward   : {stats['cumulative_reward']}")
    print(
        f"  Classification acc  : {stats['classification']['correct']}/{stats['classification']['total']} "
        f"({stats['classification']['accuracy']*100:.1f}%)"
    )
    print(
        f"  Priority acc        : {stats['priority']['correct']}/{stats['priority']['total']} "
        f"({stats['priority']['accuracy']*100:.1f}%)"
    )
    print(
        f"  Reply acc           : {stats['reply']['correct']}/{stats['reply']['total']} "
        f"({stats['reply']['accuracy']*100:.1f}%)"
    )
    print(f"  Grader score        : {grader_score:.3f} / 1.000")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default=None)
    args = parser.parse_args()
    run(tier=args.tier)