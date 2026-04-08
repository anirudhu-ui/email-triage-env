import sys
import os

# Maintain the path fix for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from models import Action, Observation, Reward

try:
    from server.email_triage_env_environment import EmailTriageEnv
except ImportError:
    from email_triage_env_environment import EmailTriageEnv

app = FastAPI(
    title="EmailTriageEnv",
    description="OpenEnv RL environment for email triage: classification → priority → reply",
    version="1.0.0",
)

env = EmailTriageEnv()

@app.get("/")
def home():
    return {
        "status": "online",
        "environment": "EmailTriageEnv",
        "framework": "OpenEnv",
        "phase_flow": "classification → priority → reply",
    }

# FIXED: Changed from .get to .post to satisfy OpenEnv validator
@app.post("/reset", response_model=Observation)
def reset():
    """Reset the environment and return the first observation."""
    observation = env.reset()
    # Ensure the response is wrapped in an "observation" key if the model doesn't do it
    return observation

@app.post("/openenv/reset", response_model=Observation)
def openenv_reset():
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