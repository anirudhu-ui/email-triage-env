import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action
from server.email_triage_env_environment import EmailTriageEnv
from grader.grader import grade
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("API_KEY", "no-key"),
)


def llm_classify(text: str) -> str:
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                f"Classify this email as exactly one word: spam, important, or promotional.\n\n{text}"}],
            temperature=0, max_tokens=5,
        )
        out = r.choices[0].message.content.strip().lower()
        for label in ["spam", "important", "promotional"]:
            if label in out:
                return label
    except Exception:
        pass
    return "spam" if "free" in text.lower() else "important"


def action_fn(state) -> Action:
    text = state.email_text.lower()
    phase = state.step
    if phase == "classification":
        value = llm_classify(state.email_text)
    elif phase == "priority":
        if any(w in text for w in ["urgent", "deadline", "mandatory", "immediately", "asap"]):
            value = "high"
        elif any(w in text for w in ["meeting", "interview", "call", "security", "payment"]):
            value = "high"
        else:
            value = "low"
    elif phase == "reply":
        if any(w in text for w in ["confirm", "available", "meeting", "interview", "call", "deadline"]):
            value = "respond"
        elif any(w in text for w in ["billing", "payment", "receipt", "newsletter", "deadline extended"]):
            value = "acknowledge"
        else:
            value = "ignore"
    else:
        value = "ignore"
    return Action(type=phase, value=value)


def run_tier(tier: str) -> float:
    """Run one tier and return grader score strictly in (0,1)."""
    env = EmailTriageEnv(tier=tier, shuffle=False)
    obs = env.reset()
    done = False
    while not done:
        action = action_fn(obs)
        obs, reward, done, info = env.step(action)
    stats = env.episode_stats()
    score = grade({
        "classification": stats["classification"]["accuracy"],
        "priority":       stats["priority"]["accuracy"],
        "reply":          stats["reply"]["accuracy"],
    })
    return score


def run(tier=None):
    print("[START]")

    if tier:
        # Single tier run (used during local testing)
        env = EmailTriageEnv(tier=tier, shuffle=False)
        obs = env.reset()
        step_number = 0
        done = False
        while not done:
            step_number += 1
            action = action_fn(obs)
            obs, reward, done, info = env.step(action)
            print(f"[STEP] #{step_number:2d} | phase={action.type:16s} | "
                  f"action='{action.value:12s}' | reward={reward.value:+.1f}")
        stats = env.episode_stats()
        score = grade({
            "classification": stats["classification"]["accuracy"],
            "priority":       stats["priority"]["accuracy"],
            "reply":          stats["reply"]["accuracy"],
        })
        print(f"\n[END]")
        print("-" * 50)
        print(f"  Grader score ({tier}): {score:.4f}")
        print("-" * 50)
    else:
        # ── Run all 3 tiers separately so evaluator gets 3 task scores ──
        step_number = 0
        for t in ["easy", "medium", "hard"]:
            env = EmailTriageEnv(tier=t, shuffle=False)
            obs = env.reset()
            done = False
            while not done:
                step_number += 1
                action = action_fn(obs)
                obs, reward, done, info = env.step(action)
                print(f"[STEP] #{step_number:2d} | tier={t:6s} | phase={action.type:16s} | "
                      f"action='{action.value:12s}' | reward={reward.value:+.1f}")
            stats = env.episode_stats()
            score = grade({
                "classification": stats["classification"]["accuracy"],
                "priority":       stats["priority"]["accuracy"],
                "reply":          stats["reply"]["accuracy"],
            })
            print(f"[TASK] tier={t} score={score:.4f}")

        print(f"\n[END]")
        print("-" * 50)
        print("  All 3 tasks completed.")
        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default=None)
    args = parser.parse_args()
    run(tier=args.tier)