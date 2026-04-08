---

title: Email Triage Env
emoji: 📧
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
---

# 📧 EmailTriageEnv — RL-Based AI Email Assistant

A Reinforcement Learning environment that simulates a real-world corporate inbox.

An AI agent triages each email through a strict **three-phase pipeline**:

> **classification → priority → reply**

This project models email triaging as a **sequential decision-making problem**, making it ideal for reinforcement learning research and evaluation.

Built for the **Scaler School of Tech × Meta OpenEnv Hackathon**.

---

## 👥 Team (Vardhaman College of Engineering)

* Devineni Adhyumna Chowdary
* Anirudh Upadyay
* Amatul Lubna

---

## 🎯 Problem Description

Corporate inboxes are noisy. Emails range from phishing attempts to urgent deadlines to newsletters nobody asked for.

This environment challenges an agent to:

1. **Classify** each email — `spam`, `important`, or `promotional`
2. **Prioritize** it — `low`, `medium`, or `high`
3. **Decide a reply action** — `ignore`, `acknowledge`, or `respond`

The agent must correctly complete all three phases for every email.

---

## 🧩 Environment Workflow

```
Email 1: [classification] → [priority] → [reply]
Email 2: [classification] → [priority] → [reply]
...
Email N → done=True
```

The agent must:

* Follow strict order
* Cannot skip or revisit steps

---

## 👁️ Observation Space

| Field        | Type | Description                                           |
| ------------ | ---- | ----------------------------------------------------- |
| `email_text` | str  | Full email content                                    |
| `step`       | str  | Current phase (`classification`, `priority`, `reply`) |

---

## 🎮 Action Space

| Field   | Type | Description                                     |
| ------- | ---- | ----------------------------------------------- |
| `type`  | str  | Phase (`classification` / `priority` / `reply`) |
| `value` | str  | Agent’s decision                                |

### Valid Values

| Phase          | Values                         |
| -------------- | ------------------------------ |
| classification | spam · important · promotional |
| priority       | low · medium · high            |
| reply          | ignore · acknowledge · respond |

---

## 🏆 Reward Structure

| Phase          | Correct | Wrong |
| -------------- | ------- | ----- |
| classification | +2      | -1    |
| priority       | +3      | -1    |
| reply          | +5      | -2    |

---

## 📋 Task Difficulty Levels

| Difficulty | Description               |
| ---------- | ------------------------- |
| Easy       | Classification only       |
| Medium     | Classification + Priority |
| Hard       | Full pipeline             |

---

## 📊 Grader

The grader evaluates performance using weighted accuracy:

| Phase          | Weight |
| -------------- | ------ |
| classification | 0.3    |
| priority       | 0.3    |
| reply          | 0.4    |

```python
from grader.grader import grade

score = grade({
    "classification": True,
    "priority": True,
    "reply": False,
})
# Output: 0.6
```

---

## ▶️ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file
# (DO NOT push this file to GitHub)
```

Add inside `.env`:

```
ANTHROPIC_API_KEY=your_api_key_here
```

```bash
# 3. Run baseline agent
python inference.py

# 4. Run LLM agent
python llm_agent.py --compare

# 5. Start API server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
```

**⚠️ Note:**

* `.env` is ignored via `.gitignore`
* Never commit API keys

---

## 🧪 Example Output

```
Email: "Win a free iPhone!!!"

→ classification: spam ✅
→ priority: low ✅
→ reply: ignore ✅

Reward: +10
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description         |
| ------ | -------- | ------------------- |
| GET    | `/reset` | Reset environment   |
| POST   | `/step`  | Submit action       |
| GET    | `/state` | Current observation |

### Example Request

```json
{
  "type": "classification",
  "value": "spam"
}
```

### Example Response

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

## 🐳 Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

---

## 📁 Project Structure

```
email_triage_env/
├── models.py
├── dataset.py
├── inference.py
├── llm_agent.py
├── evaluate.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
├── server/
│   ├── app.py
│   └── email_triage_env_environment.py
├── grader/
│   └── grader.py
└── tasks/
    └── tasks.py
```

---

## 🚀 Highlights

* ✅ Multi-step RL environment
* ✅ Realistic email dataset
* ✅ Deterministic grading system
* ✅ Baseline + LLM agent comparison
* ✅ API + Docker support

---

## 📜 License

Apache-2.0
