import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from models import Action, Observation

try:
    from server.email_triage_env_environment import EmailTriageEnv
except ImportError:
    from email_triage_env_environment import EmailTriageEnv

app = FastAPI(
    title="EmailTriageEnv",
    description="OpenEnv RL environment for email triage: classification -> priority -> reply",
    version="1.0.0",
)

env = EmailTriageEnv()


async def _extract_tier(request: Request):
    tier = request.query_params.get("tier")
    if tier:
        return tier

    try:
        body = await request.json()
    except Exception:
        body = None

    if isinstance(body, dict):
        return body.get("tier")

    return None


@app.get("/")
def home():
    return {
        "status": "online",
        "environment": "EmailTriageEnv",
        "framework": "OpenEnv",
        "phase_flow": "classification -> priority -> reply",
    }


@app.post("/reset", response_model=Observation)
async def reset(request: Request):
    """Reset the environment and return the first observation."""
    env.tier = await _extract_tier(request)
    return env.reset()


@app.post("/openenv/reset", response_model=Observation)
async def openenv_reset(request: Request):
    env.tier = await _extract_tier(request)
    return env.reset()


@app.post("/step")
def step(action: Action):
    """
    Take one action and advance the environment.
    Returns observation, reward, and done flag.
    """
    try:
        observation, reward, done, info = env.step(action)
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid step: {str(e)}")


@app.get("/state", response_model=Observation)
def get_state():
    """Return the current observation without advancing the environment."""
    return env.state()


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
