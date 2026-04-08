import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Observation, Action, Reward
from dataset import get_dataset

PHASES = ["classification", "priority", "reply"]


class EmailTriageEnv:
    """
    OpenEnv RL environment for email triage.

    Phase flow: classification → priority → reply
    """

    def __init__(self, tier: str = None, shuffle: bool = True):
        self.tier = tier
        self.shuffle = shuffle
        self.emails = []
        self.index: int = 0
        self.phase: str = PHASES[0]
        self.cumulative_reward: float = 0.0
        self._correct = {"classification": 0, "priority": 0, "reply": 0}
        self._total   = {"classification": 0, "priority": 0, "reply": 0}

    # ----------------------------------------------------------
    # RESET
    # ----------------------------------------------------------

    def reset(self) -> Observation:
        import random
        random.seed(42)
        self.emails = get_dataset(tier=self.tier, shuffle=self.shuffle)
        self.index = 0
        self.phase = PHASES[0]
        self.cumulative_reward = 0.0
        self._correct = {"classification": 0, "priority": 0, "reply": 0}
        self._total   = {"classification": 0, "priority": 0, "reply": 0}
        return self._build_observation()

    # ----------------------------------------------------------
    # STEP (CORE LOGIC)
    # ----------------------------------------------------------

    def step(self, action: Action):
        current_email = self.emails[self.index]

        reward_value = 0.0
        done = False
        info = {}

        # ✅ SAFE handling (important fix)
        action_type = action.type.strip().lower()
        action_value = action.value.strip().lower()

        # ❌ Wrong phase → penalty
        if action_type != self.phase:
            reward_value = -0.5
            self.cumulative_reward += reward_value
            info = {"error": f"Wrong phase: expected '{self.phase}', got '{action.type}'"}
            return self._build_observation(), Reward(value=reward_value), done, info

        # ---------------- CLASSIFICATION ----------------
        if self.phase == "classification":
            self._total["classification"] += 1

            if action_value == current_email["label"]:
                reward_value = 2.0
                self._correct["classification"] += 1
            else:
                reward_value = -1.0

            self.phase = "priority"

        # ---------------- PRIORITY ----------------
        elif self.phase == "priority":
            self._total["priority"] += 1

            if action_value == current_email["priority"]:
                reward_value = 3.0
                self._correct["priority"] += 1
            else:
                reward_value = -1.0

            self.phase = "reply"

        # ---------------- REPLY ----------------
        elif self.phase == "reply":
            self._total["reply"] += 1

            if action_value == current_email["reply"]:
                reward_value = 5.0
                self._correct["reply"] += 1
            else:
                reward_value = -2.0

            # Move to next email
            self.index += 1

            if self.index >= len(self.emails):
                done = True
            else:
                self.phase = "classification"

        # Update reward
        self.cumulative_reward += reward_value

        # Next observation
        if done:
            observation = self._terminal_observation()
        else:
            observation = self._build_observation()

        return observation, Reward(value=reward_value), done, info

    # ----------------------------------------------------------
    # STATE
    # ----------------------------------------------------------

    def state(self) -> Observation:
        return self._build_observation()

    # ----------------------------------------------------------
    # STATS (FOR EVALUATION)
    # ----------------------------------------------------------

    def episode_stats(self) -> dict:
        def acc(phase):
            t = self._total[phase]
            return round(self._correct[phase] / t, 3) if t > 0 else 0.0

        return {
            "emails_processed":  self.index,
            "total_emails":      len(self.emails),
            "cumulative_reward": round(self.cumulative_reward, 2),
            "classification": {
                "correct":  self._correct["classification"],
                "total":    self._total["classification"],
                "accuracy": acc("classification"),
            },
            "priority": {
                "correct":  self._correct["priority"],
                "total":    self._total["priority"],
                "accuracy": acc("priority"),
            },
            "reply": {
                "correct":  self._correct["reply"],
                "total":    self._total["reply"],
                "accuracy": acc("reply"),
            },
        }

    # ----------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------

    def _build_observation(self) -> Observation:
        if self.index >= len(self.emails):
            return self._terminal_observation()

        return Observation(
            email_text=self.emails[self.index]["text"],
            step=self.phase,
        )

    def _terminal_observation(self) -> Observation:
        return Observation(
            email_text="All emails have been processed.",
            step="done",
        )