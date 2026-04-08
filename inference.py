"""
inference.py
------------
LLM-powered inference script for EmailTriageEnv.

Uses provided API_BASE_URL and API_KEY (LiteLLM proxy)
to make real API calls (required by evaluator).

Logs exact format:
    [START]
    [STEP] ...
    [END]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action
from server.email_triage_env_environment import EmailTriageEnv
from grader.grader import grade
from openai import OpenAI


# ✅ REQUIRED: Use injected API
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"],
)


# ---------------------------------------------------------------------------
# LLM decision logic (REQUIRED FOR VALIDATOR)
# ---------------------------------------------------------------------------

def llm_action(state) -> Action:
    prompt = f"""
You are an email assistant.

Email:
{state.email_text}

Current phase: {state.step}

Choose ONE correct label only:

classification: spam or important
priority: low or high
reply: ignore or acknowledge
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    output = response.choices[0].message.content.strip().lower()

    # Simple parsing
    if "spam" in output:
        value = "spam"
    elif "important" in output:
        value = "important"
    elif "high" in output:
        value = "high"
    elif "low" in output:
        value = "low"
    elif "acknowledge" in output:
        value = "acknowledge"
    else:
        value = "ignore"

    return Action(type=state.step, value=value)


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

        # ✅ IMPORTANT: use LLM instead of baseline
        action = llm_action(obs)

        obs, reward, done, info = env.step(action)

        print(
            f"[STEP] #{step_number:2d} | phase={action.type:16s} | "
            f"action='{action.value:12s}' | reward={reward.value:+.1f}"
        )

    stats = env.episode_stats()

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
