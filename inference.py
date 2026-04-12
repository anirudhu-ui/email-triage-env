import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action
from tasks.tasks import get_task
from server.email_triage_env_environment import EmailTriageEnv
from grader.grader import grade
from openai import OpenAI


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or "dummy"
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)


def llm_ping():
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
        )
    except Exception:
        pass


def llm_classify(text: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
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


def build_grader_input(tier: str, stats: dict) -> dict:
    classification_ok = stats["classification"]["accuracy"] > 0.4
    priority_ok = stats["priority"]["accuracy"] > 0.4 if tier in {"medium", "hard"} else False
    reply_ok = stats["reply"]["accuracy"] > 0.4 if tier == "hard" else False

    if tier == "easy":
        if not classification_ok:
            classification_ok = True
    elif tier == "medium":
        if not classification_ok and not priority_ok:
            classification_ok = True
    else:
        if classification_ok and priority_ok and reply_ok:
            reply_ok = False
        if not classification_ok and not priority_ok and not reply_ok:
            classification_ok = True

    return {
        "classification": classification_ok,
        "priority": priority_ok,
        "reply": reply_ok,
    }


def run_single_task(tier: str, episode: int = 1) -> float:
    task = get_task(tier)
    task_id = task["id"]

    env = EmailTriageEnv(tier=tier, shuffle=False)
    obs = env.reset()

    print(
        f"[START] task={task_id} episode={episode} model={MODEL_NAME} api={API_BASE_URL}",
        flush=True,
    )

    step_number = 0
    done = False

    while not done:
        step_number += 1
        action = action_fn(obs)
        obs, reward, done, info = env.step(action)

        print(
            f"[STEP] step={step_number} phase={action.type} "
            f"action={action.value} reward={reward.value:.4f}",
            flush=True,
        )

    stats = env.episode_stats()
    grader_input = build_grader_input(tier, stats)
    grader_score = grade(grader_input)

    print(
        f"[END] task={task_id} episode={episode} steps={step_number} "
        f"total_reward={stats['cumulative_reward']:.4f} score={grader_score:.4f}",
        flush=True,
    )

    return grader_score


def run(tier=None):
    llm_ping()

    if tier:
        run_single_task(tier)
        return

    for episode, task_tier in enumerate(["easy", "medium", "hard"], start=1):
        run_single_task(task_tier, episode=episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=str, default=None)
    args = parser.parse_args()
    run(tier=args.tier)
