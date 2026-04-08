---
title: Email Triage Env
emoji: 📧
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# 📧 EmailTriageEnv — OpenEnv Hackathon Submission

A Reinforcement Learning environment that simulates a real-world corporate inbox.
An AI agent triages each email through a strict **three-phase pipeline**:

> **classification → priority → reply**

Built for the **Scaler School of Tech × Meta OpenEnv Hackathon**.

---

## 👥 Team (Vardhaman College of Engineering)
- Devineni Adhyumna Chowdary
- Anirudh Upadyay
- Amatul Lubna

---

## 🎯 Problem Description

Corporate inboxes are noisy. Emails range from phishing attempts to urgent deadlines
to newsletters nobody asked for. This environment challenges an agent to:

1. **Classify** each email — is it `spam`, `important`, or `promotional`?
2. **Prioritize** it — `low`, `medium`, or `high`?
3. **Decide a reply action** — `ignore`, `acknowledge`, or `respond`?

The agent must correctly chain all three phases for every email in the inbox.

---

## 🧩 Environment Explanation

The environment runs each email through three sequential phases.
The agent cannot skip phases or go back — it must answer each phase in order.

```
Email 1: [classification] → [priority] → [reply]
Email 2: [classification] → [priority] → [reply]
...
Email N: [classification] → [priority] → [reply]  →  done=True
```

---

## 👁️ Observation Space

| Field        | Type | Description                                         |
|--------------|------|-----------------------------------------------------|
| `email_text` | str  | Full text of the current email being processed      |
| `step`       | str  | Current phase: `classification`, `priority`, `reply`|

---

## 🎮 Action Space

| Field   | Type | Description                                                      |
|---------|------|------------------------------------------------------------------|
| `type`  | str  | Phase the action belongs to (`classification`/`priority`/`reply`)|
| `value` | str  | Agent's answer for the current phase (see valid values below)    |

**Valid values by phase:**

| Phase            | Valid Values                          |
|------------------|---------------------------------------|
| `classification` | `spam` · `important` · `promotional`  |
| `priority`       | `low` · `medium` · `high`             |
| `reply`          | `ignore` · `acknowledge` · `respond`  |

---

## 🏆 Reward Structure

| Phase            | Correct | Wrong |
|------------------|---------|-------|
| `classification` | +2      | -1    |
| `priority`       | +3      | -1    |
| `reply`          | +5      | -2    |

---

## 📋 Task Descriptions

| Difficulty | Name                     | Phases Evaluated                          |
|------------|--------------------------|-------------------------------------------|
| Easy       | Email Classification     | classification only                       |
| Medium     | Classification + Priority| classification + priority                 |
| Hard       | Full Triage Pipeline     | classification + priority + reply         |

---

## 📊 Grader

`grader/grader.py` scores a completed episode with weighted accuracy:

| Phase            | Weight |
|------------------|--------|
| `classification` | 0.3    |
| `priority`       | 0.3    |
| `reply`          | 0.4    |

```python
from grader.grader import grade

score = grade({
    "classification": True,
    "priority": True,
    "reply": False,
})
# → 0.6
```

---

## 🚀 Setup & Running

### Local

```bash
# 1. Install dependencies
pip install fastapi uvicorn pydantic requests anthropic colorama

# 2. Run the inference baseline
python inference.py

# 3. Run the LLM agent (requires API key)
export ANTHROPIC_API_KEY=your_key_here
python llm_agent.py --compare

# 4. Start the API server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description                              |
|--------|----------|------------------------------------------|
| GET    | `/reset` | Reset environment, get first observation |
| POST   | `/step`  | Submit an action, get next observation   |
| GET    | `/state` | Get current observation (no advance)     |

**Example `/step` request body:**
```json
{
  "type": "classification",
  "value": "spam"
}
```

**Example `/step` response:**
```json
{
  "observation": {
    "email_text": "...",
    "step": "priority"
  },
  "reward": { "value": 2.0 },
  "done": false
}
```

---

## 📁 Project Structure

```
email_triage_env/
├── models.py                               # Observation, Action, Reward models
├── dataset.py                              # 20 emails across easy/medium/hard tiers
├── inference.py                            # Baseline keyword heuristic agent
├── llm_agent.py                            # Claude-powered LLM agent
├── evaluate.py                             # Full evaluation report generator
├── openenv.yaml                            # OpenEnv config
├── Dockerfile                              # Docker container (port 7860)
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
├── server/
│   ├── __init__.py
│   ├── app.py                              # FastAPI server
│   └── email_triage_env_environment.py     # Core RL environment logic
├── grader/
│   ├── __init__.py
│   └── grader.py                           # Deterministic weighted grader
└── tasks/
    ├── __init__.py
    └── tasks.py                            # Easy/medium/hard task definitions
```
