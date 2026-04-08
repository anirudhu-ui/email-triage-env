# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Emailtriageenv Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EmailtriageenvAction, EmailtriageenvObservation


class EmailtriageenvEnv(
    EnvClient[EmailtriageenvAction, EmailtriageenvObservation, State]
):
    """
    Client for the Emailtriageenv Environment.
    """

    def _step_payload(self, action: EmailtriageenvAction) -> Dict:
        """
        Convert EmailtriageenvAction to JSON payload.
        """
        return {
            "type": action.type,
            "value": action.value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EmailtriageenvObservation]:
        """
        Parse server response into StepResult.
        """
        obs_data = payload.get("observation", {})

        observation = EmailtriageenvObservation(
            email_text=obs_data.get("email_text", ""),
            step=obs_data.get("step", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse state response.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )