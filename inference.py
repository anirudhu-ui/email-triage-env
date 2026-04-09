"""
inference.py
------------
Hybrid LLM + rule-based agent for EmailTriageEnv.

Uses evaluator-provided API credentials via environment variables.
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action
from server.email_triage_env_environment import EmailTriageEnv
from grader.grader import grade
from openai import OpenAI


# ---------------------------------------------------------------------------
# API client — uses evaluator-injected credentials
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("API_KEY", "no-key"),
)


# ---------------------------------------------------------------------------
# LLM classification (only phase that calls the API)
# ---------------------------------------------------------------------------

def llm_classification(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "You are an email classifier. "
                    "Classify the email below as exactly one of: spam, important, promotional.\n"
                    "Reply with ONLY one word.\n\n"
                    f"Email:\n{text}"
                ),
            }],
            temperature=0,
            max_tokens=10,
        )
        output = response.choices[0].message.content.strip().lower()
        for label in ["spam", "important", "promotional"]:
            if label in output:
                return label
        return "important"
    except Exception:
        return "spam" if "free" in text.lower() else "important"


# ---------------------------------------------------------------------------
# Rule-based priority and reply logic
# ---------------------------------------------------------------------------

def rule_priority(text: str, label: str) -> str:
    text = text.lower()
    if label == "spam" or label == "promotional":
        return "low"
    if any(w in text for w in ["urgent", "asap", "deadline", "immediately", "mandatory"]):
        return "high"
    if any(w in text for w in ["meeting", "interview", "call", "security"]):
        return "high"
    return "medium"


def rule_reply(text: str, label: str) -> str:
    text = text.lower()
    if label == "spam" or label == "promotional":
        return "ignore"
    if any(w in text for w in ["meeting", "interview", "call", "confirm", "available"]):
        return "respond"
    return "acknowledge"


# ---------------------------------------------------------------------------
# Full action decision
# ---------------------------------------------------------------------------

_last_label = "important"


def action_fn(state) -> Action:
    global _last_label
    text = state.email_text.lower()
    phase = state.step

    if phase == "classification":
        value = llm_classification(state.email_text)
        _last_label = value

    elif phase == "priority":
        value = rule_priority(state.email_text, _last_label)

    elif phase == "reply":
        value = rule_reply(state.email_text, _last_label)

    else:
        value = "ignore"

    return Action(type=phase, value=value)


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(tier=None):
    global _last_label
    _last_label = "important"

    env = EmailTriageEnv(tier=tier, shuffle=False)
    obs = env.reset()

    print("[START]")

    step_number = 0
    done = False

    while not done:
        step_number += 1
        action = action_fn(obs)
        obs, reward, done, info = env.step(action)
        print(
            f"[STEP] #{step_number:2d} | phase={action.type:16s} | "
            f"action='{action.value:12s}' | reward={reward.value:+.1f}"
        )

    stats = env.episode_stats()

    # Pass raw accuracy floats to grader
    # grader now accepts floats and returns strictly (0,1)
    grader_input = {
        "classification": stats["classification"]["accuracy"],
        "priority":       stats["priority"]["accuracy"],
        "reply":          stats["reply"]["accuracy"],
    }

    grader_score = grade(grader_input)

    print("\n[END]")
    print("-" * 50)
    print(f"  Total steps         : {step_number}")
    print(f"  Emails processed    : {stats['emails_processed']}")
    print(f"  Cumulative reward   : {stats['cumulative_reward']}")
    print(f"  Classification acc  : "
          f"{stats['classification']['correct']}/{stats['classification']['total']} "
          f"({stats['classification']['accuracy']*100:.1f}%)")
    print(f"  Priority acc        : "
          f"{stats['priority']['correct']}/{stats['priority']['total']} "
          f"({stats['priority']['accuracy']*100:.1f}%)")
    print(f"  Reply acc           : "
          f"{stats['reply']['correct']}/{stats['reply']['total']} "
          f"({stats['reply']['accuracy']*100:.1f}%)")
    print(f"  Grader score        : {grader_score:.4f} / 1.000")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default=None)
    args = parser.parse_args()
    run(tier=args.tier)
