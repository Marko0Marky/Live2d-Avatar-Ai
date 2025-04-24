# --- START OF FILE environment.py ---
import torch
import random
import logging
from typing import List, Tuple, Optional
from collections import deque # Import deque

from config import MasterConfig as Config, DEVICE, logger
from utils import MetronicLattice, is_safe # Import MetronicLattice

class EmotionalSpace:
    """
    Simulates an environment evoking emotional responses and accepting qualia feedback.
    Returns a 12D state directly.
    """
    def __init__(self,
                 emotion_dim: int = Config.Agent.EMOTION_DIM,
                 state_dim: int = Config.Agent.STATE_DIM,
                 event_freq: float = Config.Env.EVENT_FREQ,
                 intensities: Optional[List[float]] = None):
        super().__init__()

        if state_dim != 12: logger.critical(f"Environment expects state_dim=12, got {state_dim}.")
        if emotion_dim != 6: logger.warning(f"Environment assumes emotion_dim=6 for qualia split, got {emotion_dim}.")
        if emotion_dim > state_dim: logger.critical(f"Emotion dim ({emotion_dim}) > state dim ({state_dim}).")

        self.emotion_dim = emotion_dim
        self.qualia_dim = state_dim - emotion_dim # Should be 6
        self.state_dim = state_dim
        self.emotion_names = ['Joy', 'Fear', 'Curiosity', 'Frustration', 'Calm', 'Surprise'][:self.emotion_dim]
        logger.info(f"Emotion names set: {self.emotion_names}")

        self.current_emotions = torch.rand(self.emotion_dim, device=DEVICE) * 0.5
        # --- ADDED internal_qualia_influence ---
        self.internal_qualia_influence = torch.zeros(self.emotion_dim, device=DEVICE) # How feedback affects next emotions

        self.state_history_deque: deque[torch.Tensor] = deque(maxlen=Config.Agent.HISTORY_SIZE)

        self.event_freq = event_freq
        self.default_intensities = [0.8, 0.7, 0.6, 0.7, 0.5, 0.9][:self.emotion_dim]
        self.event_intensities = intensities if intensities else self.default_intensities
        if len(self.event_intensities) != self.emotion_dim:
            logger.warning(f"Env intensities len ({len(self.event_intensities)}) != emo_dim ({self.emotion_dim}). Adjusting."); self.event_intensities = (self.event_intensities + self.default_intensities)[:self.emotion_dim]

        self.lattice = MetronicLattice(dim=self.emotion_dim, tau=Config.Agent.TAU) # Lattice for internal emotions

        self.steps_since_event = 0; self.current_event_type = "calm"; self.current_event_duration = 0; self.event_gap_counter = 0

        logger.info("EmotionalSpace environment initialized.")
        self.reset()

    @property
    def state_history(self) -> torch.Tensor:
        valid = all(isinstance(t, torch.Tensor) and t.shape == (self.state_dim,) for t in self.state_history_deque)
        if not valid or len(self.state_history_deque) < Config.Agent.HISTORY_SIZE:
            logger.error(f"Env history invalid/incomplete. Reinitializing."); self.reset();
        try: return torch.stack(list(self.state_history_deque)).to(DEVICE).float()
        except Exception as e: logger.error(f"Error stacking env history: {e}."); return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE)

    def _trigger_event(self):
        """Triggers a new emotional event and calculates its immediate impact."""
        self.current_event_type = random.choice(self.emotion_names + ["mixed", "calm_down", "agitated"])
        self.current_event_duration = random.randint(Config.Env.EVENT_DURATION // 2, Config.Env.EVENT_DURATION)
        self.event_gap_counter = random.randint(Config.Env.EVENT_GAP // 2, Config.Env.EVENT_GAP)
        self.steps_since_event = 0
        logger.debug(f"Event triggered: {self.current_event_type} for {self.current_event_duration} steps.")

        impact = torch.zeros(self.emotion_dim, device=DEVICE)

        # Calculate impact based on event type
        if self.current_event_type in self.emotion_names:
            try: # Add try-except for index safety
                idx = self.emotion_names.index(self.current_event_type)
                if 0 <= idx < len(self.event_intensities):
                     impact[idx] = self.event_intensities[idx] * random.uniform(0.7, 1.3)
                else:
                     logger.warning(f"_trigger_event: Intensity index {idx} out of bounds.")
            except ValueError:
                logger.warning(f"_trigger_event: Event type '{self.current_event_type}' not found in emotion names list.")

        elif self.current_event_type == "mixed":
            # Ensure k is not larger than population size
            k_sample = random.randint(2, min(4, self.emotion_dim))
            if k_sample > 0: # Only sample if k > 0
                indices = random.sample(range(self.emotion_dim), k=k_sample)
                for idx in indices:
                    if 0 <= idx < len(self.event_intensities):
                         impact[idx] = self.event_intensities[idx] * random.uniform(0.4, 1.0)
                    else:
                         logger.warning(f"_trigger_event (mixed): Intensity index {idx} out of bounds.")

        elif self.current_event_type == "calm_down" and self.emotion_dim >= 5: # Assumes Calm is index 4
            calm_idx = 4
            reduction = (self.current_emotions - 0.3).clamp(min=0) * 0.5
            impact = -reduction # Apply reduction first
            # Safely add calm boost
            if calm_idx < len(impact):
                impact[calm_idx] += 0.2 * self.current_emotions.max().item()
            else:
                logger.warning(f"_trigger_event (calm_down): Calm index {calm_idx} out of bounds.")

        elif self.current_event_type == "agitated" and self.emotion_dim >= 4: # Assumes Frustration=3, Fear=1
            frust_idx = 3
            fear_idx = 1
            # Safely apply impacts
            if frust_idx < len(impact):
                 impact[frust_idx] = 0.4
            else:
                 logger.warning(f"_trigger_event (agitated): Frustration index {frust_idx} out of bounds.")
            if fear_idx < len(impact):
                 impact[fear_idx] = 0.3
            else:
                 logger.warning(f"_trigger_event (agitated): Fear index {fear_idx} out of bounds.")

        else: # Default case for "transition", unknown events, or dimensions too small
            impact = -(self.current_emotions - 0.3).clamp(min=0) * 0.3

        # Blend the impact with current emotions
        blend_factor = 0.6
        self.current_emotions = self.current_emotions * (1.0 - blend_factor) + impact * blend_factor
        self.current_emotions = torch.clamp(self.current_emotions, 0.0, 1.0)

    # --- ADDED Method ---
    def update_qualia_feedback(self, qualia_feedback: Optional[torch.Tensor]):
        """ Incorporates feedback from agent's qualia state (R7-12) into internal dynamics. """
        if qualia_feedback is None or qualia_feedback.shape[0] != self.qualia_dim: logger.warning(f"Invalid qualia feedback shape: {getattr(qualia_feedback, 'shape', 'None')}"); return
        if not is_safe(qualia_feedback): logger.warning("Unsafe qualia feedback received."); return

        # Simple mapping: Apply qualia feedback scaled by strength to influence emotions
        # This assumes qualia_dim matches emotion_dim, otherwise need projection
        if self.qualia_dim == self.emotion_dim:
            influence = qualia_feedback.float() * Config.Env.QUALIA_FEEDBACK_STRENGTH
        else:
            # If dimensions don't match, use a simple projection (or ignore)
            logger.warning("Qualia feedback dim != emotion dim, using simple mean influence.")
            influence = torch.full_like(self.current_emotions, qualia_feedback.mean().item() * Config.Env.QUALIA_FEEDBACK_STRENGTH * 0.1) # Less impact if mean is used

        # Use EMA to slowly integrate the influence
        self.internal_qualia_influence = 0.95 * self.internal_qualia_influence + 0.05 * influence.clamp(-0.2, 0.2) # Clamp influence magnitude

    def _get_state(self) -> torch.Tensor:
        """ Generates 12D state: influenced_emotions[:6] + zeros[6:]. """
        # Apply qualia influence accumulated from previous step's feedback
        influenced_emotions = self.current_emotions + self.internal_qualia_influence
        influenced_emotions = torch.clamp(influenced_emotions, 0.0, 1.0)
        discretized_emotions = self.lattice.discretize(influenced_emotions)
        higher_dims = torch.zeros(self.qualia_dim, device=DEVICE) # R7-12 are observational zeros
        state = torch.cat([discretized_emotions, higher_dims])
        if state.shape[0] != self.state_dim: logger.error(f"Env _get_state shape error: {state.shape}."); state = torch.zeros(self.state_dim, device=DEVICE)
        elif not is_safe(state): logger.error(f"Env _get_state unsafe."); state = torch.zeros(self.state_dim, device=DEVICE)
        return state

    def step(self) -> Tuple[torch.Tensor, float, bool, str, str]:
        """ Advances the environment state by one step. """
        self.steps_since_event += 1

        # Check if current event ends
        if self.current_event_duration > 0:
            self.current_event_duration -= 1
            if self.current_event_duration == 0:
                logger.debug(f"Event '{self.current_event_type}' ended.")
                self.current_event_type = "transition" # Indicate transitioning out

        # Natural emotional decay/shift towards baseline
        decay_rate = 0.01
        baseline = torch.zeros_like(self.current_emotions)
        # (Rest of the step method follows below...)
        if self.emotion_dim > 4: baseline[4] = 0.2
        self.current_emotions -= decay_rate * (self.current_emotions - baseline)
        self.current_emotions = torch.clamp(self.current_emotions, 0.0, 1.0)
        if self.current_event_duration <= 0:
             self.event_gap_counter -=1
             if self.event_gap_counter <= 0 :
                 if random.random() < self.event_freq: self._trigger_event()
                 else: self.event_gap_counter = random.randint(Config.Env.EVENT_GAP // 2, Config.Env.EVENT_GAP)
        reward = 0.01
        if self.emotion_dim >= 5: reward += self.current_emotions[4].item() * 0.1 # Calm
        if self.emotion_dim >= 4: reward -= self.current_emotions[3].item() * 0.15 # Frust
        if self.emotion_dim >= 2: reward -= self.current_emotions[1].item() * 0.1 # Fear
        reward -= (self.current_emotions.max().item() - 0.8) * 0.1 if self.current_emotions.max().item() > 0.8 else 0.0
        next_full_state = self._get_state() # Gets state AFTER internal emotion dynamics update
        self.state_history_deque.append(next_full_state.clone().detach())
        done = False
        context = f"Feeling {self.current_emotions.cpu().numpy().round(2)}. Event: {self.current_event_type}."
        return next_full_state, reward, done, context, self.current_event_type

    def reset(self) -> torch.Tensor:
        """ Resets the environment to a starting state. """
        logger.info("Resetting environment state."); self.current_emotions = torch.rand(self.emotion_dim, device=DEVICE) * 0.3
        self.internal_qualia_influence.zero_(); self.steps_since_event = 0; self.current_event_type = "start"; self.current_event_duration = 0; self.event_gap_counter = Config.Env.EVENT_GAP // 2
        initial_state = self._get_state()
        self.state_history_deque.clear();
        for _ in range(Config.Agent.HISTORY_SIZE): self.state_history_deque.append(initial_state.clone().detach())
        return initial_state

    def update_params(self, event_freq: Optional[float] = None, intensities: Optional[List[float]] = None):
        """ Allows updating environment parameters dynamically. """
        if event_freq is not None: self.event_freq = max(0.0, min(1.0, event_freq)); logger.info(f"Env event freq updated: {self.event_freq:.3f}")
        if intensities is not None:
            if len(intensities) == self.emotion_dim: self.event_intensities = [max(0.0, float(i)) for i in intensities]; logger.info(f"Env intensities updated: {self.event_intensities}")
            else: logger.warning(f"Cannot update intensities: Len mismatch ({len(intensities)} vs {self.emotion_dim}).")

    def get_emotional_impact_from_text(self, text: str) -> torch.Tensor:
        """ Basic NLP to estimate emotional impact vector from text. Placeholder. """
        impact = torch.zeros(self.emotion_dim, device=DEVICE); text_lower = text.lower()
        if "happy" in text_lower or "great" in text_lower: impact[0] += 0.6
        if "sad" in text_lower or "cry" in text_lower: impact[0] -= 0.4; impact[3] += 0.2 if self.emotion_dim > 3 else 0.0
        if "scared" in text_lower or "afraid" in text_lower: impact[1] += 0.7 if self.emotion_dim > 1 else 0.0
        if "why" in text_lower or "how" in text_lower or "?" in text: impact[2] += 0.4 if self.emotion_dim > 2 else 0.0
        if "stupid" in text_lower or "hate" in text_lower or "problem" in text_lower: impact[3] += 0.6 if self.emotion_dim > 3 else 0.0
        if "relax" in text_lower or "calm" in text_lower: impact[4] += 0.5 if self.emotion_dim > 4 else 0.0
        if "wow" in text_lower or "!" in text: impact[5] += 0.5 if self.emotion_dim > 5 else 0.0
        if "not happy" in text_lower: impact[0] -= 0.5
        if "not scared" in text_lower and self.emotion_dim > 1: impact[1] -= 0.4
        return impact.clamp(-1.0, 1.0)

# --- END OF FILE environment.py ---
