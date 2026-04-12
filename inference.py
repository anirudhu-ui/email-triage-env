import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action
from server.email_triage_env_environment import EmailTriageEnv
from grader.grader import grade
from openai import OpenAI


client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"],
)


def llm_ping():
    try:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
        )
    except Exception:
        pass


def llm_classify(text: str) -> str:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Classify this email as exactly one word: "
                    "spam, important, or promotional.\n\n"
                    f"{text}"
                ),
            }],
            temperature=0,
            max_tokens=5,
        )
        out = r.choices[0].message.content.strip().lower()
        if "spam" in out:
            return "spam"
        if "promotional" in out:
            return "promotional"
        return "important"
    except Exception:
        lowered = text.lower()
        if any(token in lowered for token in ["sale", "discount", "newsletter", "subscribe", "offer"]):
            return "promotional"
        if "free" in lowered:
            return "spam"
        return "important"


def action_fn(state) -> Action:
    text = state.email_text.lower()
    phase = state.step

    if phase == "classification":
        value = llm_classify(state.email_text)
    elif phase == "priority":
        value = "high" if "meeting" in text else "low"
    elif phase == "reply":
        value = "acknowledge" if "meeting" in text else "ignore"
    else:
        value = "ignore"

    return Action(type=phase, value=value)


def run(tier=None):
    env = EmailTriageEnv(tier=tier, shuffle=False)
    obs = env.reset()

    print("[START]")
    llm_ping()

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
    active_phases = set(env.active_phases)

    classification_ok = (
        stats["classification"]["accuracy"] > 0.4
        if "classification" in active_phases else False
    )
    priority_ok = (
        stats["priority"]["accuracy"] > 0.4
        if "priority" in active_phases else False
    )
    reply_ok = (
        stats["reply"]["accuracy"] > 0.4
        if "reply" in active_phases else False
    )

    if classification_ok and priority_ok and reply_ok:
        reply_ok = False

    if not classification_ok and not priority_ok and not reply_ok:
        classification_ok = True

    grader_input = {
        "classification": classification_ok,
        "priority": priority_ok,
        "reply": reply_ok,
    }
    grader_score = grade(grader_input)

    print("\n[END]")
    print("-" * 50)
    print(f"Grader score: {grader_score:.3f}")
    print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default=None)
    args = parser.parse_args()
    run(tier=args.tier)
