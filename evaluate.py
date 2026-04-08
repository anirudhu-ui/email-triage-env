"""
evaluate.py
-----------
Full evaluation report for EmailTriageEnv.

Runs the baseline agent across all tiers (easy / medium / hard / all)
and prints a comprehensive performance breakdown.

If ANTHROPIC_API_KEY is set, also runs the LLM agent for comparison.

Usage:
    python evaluate.py                    # baseline only
    python evaluate.py --llm              # baseline + LLM comparison
    python evaluate.py --llm --tier hard  # LLM on hard emails only
"""

import os
import sys
import argparse
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action
from server.email_triage_env_environment import EmailTriageEnv
from grader.grader import grade
from dataset import EMAIL_DATASET, dataset_stats

# ── ANSI colours (graceful fallback on Windows) ───────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""


# ── Baseline heuristic ────────────────────────────────────────────────────

def baseline_decide(email_text: str, phase: str) -> str:
    text = email_text.lower()
    if phase == "classification":
        return "spam" if "free" in text else "important"
    elif phase == "priority":
        return "high" if "meeting" in text else "low"
    elif phase == "reply":
        return "acknowledge" if "meeting" in text else "ignore"
    return "ignore"


# ── LLM decision (lazy import so baseline-only runs without anthropic) ────

def llm_decide(client, email_text: str, phase: str) -> str:
    import json
    import anthropic

    SYSTEM = """You are an expert email triage assistant.
Respond ONLY with a JSON object: {"value": "<answer>"}

PHASE classification → valid: spam | important | promotional
PHASE priority       → valid: low | medium | high
PHASE reply          → valid: ignore | acknowledge | respond
"""
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"PHASE: {phase}\n\nEMAIL:\n{email_text}"}],
    )
    raw = resp.content[0].text.strip().strip("```json").strip("```").strip()
    return json.loads(raw)["value"].strip().lower()


# ── Single-email result ───────────────────────────────────────────────────

def evaluate_email(email: dict, decide_fn) -> dict:
    """Run all 3 phases on one email using decide_fn. Returns per-phase results."""
    results = {}
    for phase, key in [("classification", "label"), ("priority", "priority"), ("reply", "reply")]:
        predicted = decide_fn(email["text"], phase)
        correct   = predicted == email[key]
        results[phase] = {
            "predicted": predicted,
            "expected":  email[key],
            "correct":   correct,
        }
    return results


# ── Full episode evaluation ───────────────────────────────────────────────

def run_evaluation(emails: list, decide_fn, agent_name: str, verbose: bool = True) -> dict:
    """Evaluate agent over a list of emails. Returns summary stats."""

    per_phase = {"classification": [], "priority": [], "reply": []}
    email_scores = []

    if verbose:
        print(f"\n{BOLD}{CYAN}{'─'*62}{RESET}")
        print(f"{BOLD}  Agent: {agent_name}{RESET}")
        print(f"{BOLD}{'─'*62}{RESET}")
        header = f"  {'#':>2}  {'Tier':<8}  {'C':^3}  {'P':^3}  {'R':^3}  {'Score':>6}  Subject (truncated)"
        print(header)
        print(f"  {'─'*58}")

    for i, email in enumerate(emails, 1):
        results = evaluate_email(email, decide_fn)

        c = results["classification"]["correct"]
        p = results["priority"]["correct"]
        r = results["reply"]["correct"]

        per_phase["classification"].append(c)
        per_phase["priority"].append(p)
        per_phase["reply"].append(r)

        # grader score for this single email
        email_grade = grade({"classification": c, "priority": p, "reply": r})
        email_scores.append(email_grade)

        if verbose:
            tick = lambda b: f"{GREEN}✓{RESET}" if b else f"{RED}✗{RESET}"
            subject_line = email["text"].split("\n")[1].replace("Subject: ", "")[:35]
            tier_tag = f"[{email['tier']}]"
            print(
                f"  {i:>2}  {tier_tag:<8}  "
                f"{tick(c)}    {tick(p)}    {tick(r)}   "
                f"{email_grade:.2f}   {subject_line}"
            )

    # Aggregate
    def acc(lst): return round(sum(lst) / len(lst), 3) if lst else 0.0

    c_acc = acc(per_phase["classification"])
    p_acc = acc(per_phase["priority"])
    r_acc = acc(per_phase["reply"])

    overall_grader = grade({
        "classification": c_acc >= 0.5,
        "priority":       p_acc >= 0.5,
        "reply":          r_acc >= 0.5,
    })
    avg_email_score = round(sum(email_scores) / len(email_scores), 3)

    return {
        "agent":                   agent_name,
        "total_emails":            len(emails),
        "classification_accuracy": c_acc,
        "priority_accuracy":       p_acc,
        "reply_accuracy":          r_acc,
        "avg_email_grader_score":  avg_email_score,
        "overall_grader_score":    overall_grader,
        "per_email_scores":        email_scores,
        "per_phase_details":       per_phase,
    }


# ── Pretty report ─────────────────────────────────────────────────────────

def print_summary(results: list):
    print(f"\n{BOLD}{'═'*62}{RESET}")
    print(f"{BOLD}  📊  EVALUATION SUMMARY{RESET}")
    print(f"{BOLD}{'═'*62}{RESET}\n")

    for r in results:
        print(f"  {BOLD}Agent               :{RESET} {r['agent']}")
        print(f"  Total emails        : {r['total_emails']}")
        print(f"  {BOLD}─── Per-Phase Accuracy ────────────────────{RESET}")

        def bar(acc):
            filled = int(acc * 20)
            color  = GREEN if acc >= 0.8 else YELLOW if acc >= 0.5 else RED
            return f"{color}{'█'*filled}{'░'*(20-filled)}{RESET} {acc*100:.1f}%"

        print(f"  Classification      : {bar(r['classification_accuracy'])}")
        print(f"  Priority            : {bar(r['priority_accuracy'])}")
        print(f"  Reply               : {bar(r['reply_accuracy'])}")
        print(f"  {BOLD}─── Grader Scores ─────────────────────────{RESET}")
        print(f"  Avg per-email score : {r['avg_email_grader_score']:.3f} / 1.000")
        
        print(f"  {BOLD}Overall grader score: {r['overall_grader_score']:.3f} / 1.000{RESET}")
print()

# ✅ ADD FROM HERE
overall_grader = r["overall_grader_score"]

print(f"\n🏆 FINAL SCORE: {overall_grader:.3f} / 1.000")

if overall_grader > 0.8:
    print("🔥 Strong performance — production-ready agent")
elif overall_grader > 0.5:
    print("⚡ Moderate performance — needs improvement")
else:
    print("❌ Weak performance — poor decision quality")

print()  # spacing
# ✅ END HERE

    if len(results) == 2:
        b = results[0]
        l = results[1]
        delta_c     = l["classification_accuracy"] - b["classification_accuracy"]
        delta_p     = l["priority_accuracy"]        - b["priority_accuracy"]
        delta_r     = l["reply_accuracy"]           - b["reply_accuracy"]
        delta_grade = l["overall_grader_score"]     - b["overall_grader_score"]

        sign = lambda v: f"{GREEN}+{v*100:.1f}%{RESET}" if v > 0 else f"{RED}{v*100:.1f}%{RESET}"

        print(f"  {BOLD}{'─'*60}{RESET}")
        print(f"  {BOLD}🏆  LLM vs Baseline Improvement{RESET}")
        print(f"  Classification      : {sign(delta_c)}")
        print(f"  Priority            : {sign(delta_p)}")
        print(f"  Reply               : {sign(delta_r)}")
        print(f"  Grader Score Delta  : {sign(delta_grade)}")
        print(f"  {BOLD}{'═'*62}{RESET}\n")


def print_dataset_overview():
    stats = dataset_stats()
    print(f"\n{BOLD}{'═'*62}{RESET}")
    print(f"{BOLD}  📬  DATASET OVERVIEW{RESET}")
    print(f"{BOLD}{'═'*62}{RESET}")
    print(f"  Total emails  : {stats['total']}")
    print(f"  By tier       : {stats['tiers']}")
    print(f"  By label      : {stats['labels']}")
    print(f"  By priority   : {stats['priorities']}")
    print(f"  By reply      : {stats['replies']}")
    print(f"{BOLD}{'═'*62}{RESET}\n")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EmailTriageEnv Evaluation Report")
    parser.add_argument("--llm",  action="store_true", help="Also run LLM agent (needs ANTHROPIC_API_KEY)")
    parser.add_argument("--tier", type=str, default=None, help="easy | medium | hard | (omit for all)")
    args = parser.parse_args()

    print_dataset_overview()

    # Select emails
    from dataset import get_dataset
    emails = get_dataset(tier=args.tier, shuffle=False)

    all_results = []

    # ── Baseline ──
    print(f"{BOLD}Running baseline agent...{RESET}")
    baseline_result = run_evaluation(emails, baseline_decide, "Baseline Heuristic")
    all_results.append(baseline_result)

    # ── LLM ──
    if args.llm:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"\n{RED}⚠️  ANTHROPIC_API_KEY not set. Skipping LLM agent.{RESET}")
        else:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                print(f"\n{BOLD}Running LLM agent (Claude)...{RESET}")

                def llm_fn(text, phase):
                    result = llm_decide(client, text, phase)
                    time.sleep(0.3)
                    return result

                llm_result = run_evaluation(emails, llm_fn, "LLM Agent (Claude)")
                all_results.append(llm_result)
            except ImportError:
                print(f"{RED}anthropic package not installed. Run: pip install anthropic{RESET}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
