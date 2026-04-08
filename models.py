from pydantic import BaseModel, Field
from typing import Optional


class Observation(BaseModel):
    """What the AI agent sees at every step."""
    email_text: str = Field(..., description="The full text of the current email being processed")
    step: str = Field(..., description="Current phase: 'classification', 'priority', or 'reply'")


class Action(BaseModel):
    """What the AI agent is allowed to do."""
    type: str = Field(..., description="The phase this action belongs to: 'classification', 'priority', or 'reply'")
    value: str = Field(..., description="The agent's answer for the current phase")


class Reward(BaseModel):
    """Reward returned after each step."""
    value: float = Field(..., description="Reward value for the action taken")
