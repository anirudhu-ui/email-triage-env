"""
inference.py
------------
Baseline inference script for EmailTriageEnv.

Runs a keyword heuristic agent through all 20 emails and prints
a full evaluation report at the end.

Logs exact format:
    [START]
    [STEP] ...
    [END]

Usage:
    python inference.py              # all 20 emails
    python inference.py --tier easy  # easy emails only
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action
from server.email_triage_env_environment import EmailTriageEnv
from grader.grader import grade


# ---------------------------------------------------------------------------
# Baseline decision logic
# ---------------------------------------------------------------------------

def baseline_action(state) -> Action:
    """Simple keyword-based heuristic."""
    text  = state.email_text.lower()
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

    # Build grader input from per-phase accuracy
    grader_input = {
        "classification": stats["classification"]["accuracy"] >= 0.5,
        "priority":       stats["priority"]["accuracy"]       >= 0.5,
        "reply":          stats["reply"]["accuracy"]          >= 0.5,
    }
    grader_score = grade(grader_input)

    print(f"\n[END]")
    print("-" * 50)
    print(f"  Total steps         : {step_number}")
    print(f"  Emails processed    : {stats['emails_processed']}")
    print(f"  Cumulative reward   : {stats['cumulative_reward']}")
    print(f"  Classification acc  : {stats['classification']['correct']}/{stats['classification']['total']} "
          f"({stats['classification']['accuracy']*100:.1f}%)")
    print(f"  Priority acc        : {stats['priority']['correct']}/{stats['priority']['total']} "
          f"({stats['priority']['accuracy']*100:.1f}%)")
    print(f"  Reply acc           : {stats['reply']['correct']}/{stats['reply']['total']} "
          f"({stats['reply']['accuracy']*100:.1f}%)")
    print(f"  Grader score        : {grader_score:.3f} / 1.000")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default=None, help="easy | medium | hard | (omit for all)")
    args = parser.parse_args()
    run(tier=args.tier)
