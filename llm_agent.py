"""
llm_agent.py
------------
Final hybrid agent (LLM + rules) with proper evaluation report
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from models import Action
from server.email_triage_env_environment import EmailTriageEnv
from grader.grader import grade


# ============================================================
# 🔑 INSERT YOUR OPENROUTER KEY
# ============================================================

import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)


# ============================================================
# 🤖 LLM (ONLY FOR CLASSIFICATION)
# ============================================================

def get_classification(text: str) -> str:
    prompt = f"""
Classify this email into ONE word:
spam / important / promotional

Email:
{text}

Answer only one word.
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        raw = response.choices[0].message.content

        # safe parsing
        value = str(raw).strip().lower().split()[0]

        if value in ["spam", "important", "promotional"]:
            return value

    except Exception as e:
        print(f"[WARN] LLM failed → fallback ({e})")

    return "important"


# ============================================================
# ⚙️ RULE-BASED LOGIC (STRONG + STABLE)
# ============================================================

def priority_logic(text: str, label: str) -> str:
    text = text.lower()

    if label == "important":
        if "urgent" in text or "asap" in text or "deadline" in text:
            return "high"
        return "medium"

    if label == "promotional":
        return "low"

    return "low"


def reply_logic(text: str, label: str) -> str:
    text = text.lower()

    if label == "important":
        if "meeting" in text or "interview" in text:
            return "respond"
        return "acknowledge"

    if label == "promotional":
        return "ignore"

    return "ignore"


# ============================================================
# 🚀 RUN EPISODE
# ============================================================

def run_episode(env, use_llm=True):
    obs = env.reset()
    done = False
    step = 1

    last_label = "important"

    print("\n[START]\n")

    while not done:
        text = obs.email_text
        phase = obs.step

        if phase == "classification":
            if use_llm:
                value = get_classification(text)
            else:
                value = "important"
            last_label = value

        elif phase == "priority":
            value = priority_logic(text, last_label)

        else:
            value = reply_logic(text, last_label)

        action = Action(type=phase, value=value)
        obs, reward, done, _ = env.step(action)

        status = "[OK]" if reward.value > 0 else "[ERR]"

        print(f"[STEP] #{step:2d} | {status} | {phase:15s} | {value:10s} | {reward.value:+.1f}")
        step += 1

    print("\n[END]\n")

    # ============================================================
    # 📊 FINAL EVALUATION REPORT
    # ============================================================

    stats = env.episode_stats()

    grader_input = {
        "classification": stats["classification"]["accuracy"] >= 0.5,
        "priority": stats["priority"]["accuracy"] >= 0.5,
        "reply": stats["reply"]["accuracy"] >= 0.5,
    }

    grader_score = grade(grader_input)

    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Emails Processed    : {stats['emails_processed']}")
    print(f"  Cumulative Reward   : {stats['cumulative_reward']}")
    print("  ---------------------------------")
    print(f"  Classification Acc  : {stats['classification']['accuracy']*100:.1f}%")
    print(f"  Priority Acc        : {stats['priority']['accuracy']*100:.1f}%")
    print(f"  Reply Acc           : {stats['reply']['accuracy']*100:.1f}%")
    print("  ---------------------------------")
    print(f"  Grader Score        : {grader_score:.3f} / 1.000")

    print(f"\n🏆 FINAL SCORE: {grader_score:.3f} / 1.000")

    if grader_score > 0.8:
        print("🔥 Strong performance — production-ready agent")
    elif grader_score > 0.5:
        print("⚡ Moderate performance — needs improvement")
    else:
        print("❌ Weak performance — poor decision quality")

    print("=" * 60)

    return {
        "reward": stats["cumulative_reward"],
        "score": grader_score
    }


# ============================================================
# 🆚 COMPARE MODE
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        print("\n=== BASELINE ===")
        base = run_episode(EmailTriageEnv(), use_llm=False)

        print("\n=== LLM ===")
        llm = run_episode(EmailTriageEnv(), use_llm=True)

        print("\n" + "=" * 60)
        print("  COMPARISON")
        print("=" * 60)
        print(f"  Reward improvement : {llm['reward'] - base['reward']:+.1f}")
        print(f"  Score improvement  : {llm['score'] - base['score']:+.3f}")
        print("=" * 60)

    else:
        run_episode(EmailTriageEnv(), use_llm=True)


# ============================================================
# 🏁 ENTRY
# ============================================================

if __name__ == "__main__":
    main()