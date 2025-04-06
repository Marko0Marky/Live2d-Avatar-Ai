# --- START OF FILE environment.py ---

import torch
import numpy as np
from collections import deque
import random
import sys

# Need Config, DEVICE, logger, MetronicLattice, is_safe
from config import Config, DEVICE, logger
from utils import MetronicLattice, is_safe

class EventManager:
    # ... (EventManager class remains the same) ...
    """Manages the selection of random events."""
    def __init__(self, events):
        if not events:
            logger.error("EventManager initialized with no events!"); self.events = []
        else:
            if not all(isinstance(e, (list, tuple)) and len(e) == 3 and isinstance(e[0], str) and isinstance(e[1], str) and isinstance(e[2], (float, int)) for e in events):
                logger.error("EventManager: Invalid event structure detected. Using empty list.")
                self.events = []
            else:
                self.events = list(events)
        self.last_event_type = None
        if not self.events:
             logger.warning("EventManager event list is empty after initialization/validation.")

    def get_next_event(self):
        """Gets a random event, trying not to repeat the last type immediately."""
        if not self.events:
            logger.error("EventManager: No events available to select from!");
            return None, "Error: No events available", 0.0
        available = [e for e in self.events if self.last_event_type is None or e[0] != self.last_event_type]
        if not available:
            if self.events: available = self.events
            else:
                 logger.error("EventManager: No events in list, cannot select.")
                 return None, "Error: Event list empty", 0.0
        try:
            next_event = random.choice(available)
            self.last_event_type = next_event[0]
            return next_event
        except (IndexError, TypeError) as e:
            logger.error(f"EventManager: Error selecting next event: {e}. Returning default error event.", exc_info=True)
            return None, "Error: Event selection failed", 0.0

class EmotionalSpace:
    """Simulates the environment triggering events and providing state."""
    def __init__(self):
        logger.info("Initializing EmotionalSpace environment.")
        self.step_count = 0
        self.events = [
            ("joy", "Bird sings", 2.0),
            ("fear", "Shadow looms", -2.0),
            ("curiosity", "Intriguing object", 1.0),
            ("frustration", "Glitch occurs", -1.0),
            ("calm", "Gentle breeze", 0.5),
            ("surprise", "Sudden flash", 1.5)
        ]
        if len(self.events) != Config.EMOTION_DIM:
            logger.critical(f"FATAL: Event count {len(self.events)} != Config.EMOTION_DIM {Config.EMOTION_DIM}. Check Config/events list."); sys.exit(1)

        self.emotion_names = [e[0].capitalize() for e in self.events]
        if len(self.emotion_names) != Config.EMOTION_DIM:
             logger.critical(f"FATAL: Emotion name count {len(self.emotion_names)} != Config.EMOTION_DIM {Config.EMOTION_DIM}."); sys.exit(1)
        logger.info(f"Emotion names set: {self.emotion_names}")

        self.event_manager = EventManager(self.events)
        self.event_timer = 0; self.gap_timer = Config.EVENT_GAP; self.current_event_type = None; self.current_context = "Quiet." # Changed default context
        self.base_intensities = torch.ones(Config.EMOTION_DIM, device=DEVICE) * 0.5
        self.lattice = MetronicLattice(dim=Config.STATE_DIM)

        self.state_history_deque = deque(maxlen=Config.HISTORY_SIZE)
        for _ in range(Config.HISTORY_SIZE):
            self.state_history_deque.append(torch.zeros(Config.STATE_DIM, device=DEVICE, dtype=torch.float32))

        expected_state_dim = Config.EMOTION_DIM + 6
        if Config.STATE_DIM != expected_state_dim:
            logger.critical(f"FATAL: Config.STATE_DIM ({Config.STATE_DIM}) != expected ({expected_state_dim}). Fix Config."); sys.exit(1)

        # --- NEW: Setup emotion keywords ---
        self._setup_emotion_keywords()

    # --- NEW: Add keyword setup method ---
    def _setup_emotion_keywords(self):
        """Sets up keyword mappings for text-based emotion analysis."""
        # Simple keyword mapping to emotion indices
        # (Index: [Keywords], Impact_Strength)
        self.emotion_keywords = {
            0: (["happy", "joy", "great", "wonderful", "love", "yay", "good", "nice", "fun", "awesome", "kawaii", "beautiful", "like"], 0.8),  # Joy
            1: (["scary", "fear", "afraid", "nervous", "danger", "help", "spooky", "kyaa", "worried"], 0.8),            # Fear
            2: (["interesting", "curious", "what", "how", "why", "tell me", "explain", "sugoi", "think", "wonder"], 0.7),         # Curiosity
            3: (["ugh", "annoying", "frustrating", "bad", "hate", "stupid", "wrong", "problem", "error", "sad", "cry"], 0.7),          # Frustration
            4: (["calm", "peaceful", "relax", "quiet", "gentle", "soft", "serene", "okay", "fine"], 0.6),                     # Calm
            5: (["wow", "whoa", "surprise", "really", "omg", "unexpected", "amazing", "incredible"], 0.7),          # Surprise
        }
        logger.debug("Emotion keywords set up.")
        if len(self.emotion_keywords) < Config.EMOTION_DIM:
             logger.warning("EmotionalSpace keyword map has fewer entries than EMOTION_DIM.")

    @property
    def state_history(self):
        # ... (state_history property remains the same) ...
        """Provides the state history as a stacked torch tensor."""
        if not self.state_history_deque:
             return torch.zeros(Config.HISTORY_SIZE, Config.STATE_DIM, device=DEVICE, dtype=torch.float32)
        valid_elements = True
        for t in self.state_history_deque:
             if not isinstance(t, torch.Tensor) or t.shape != (Config.STATE_DIM,):
                 logger.error(f"state_history deque contains invalid element: type={type(t)}, shape={t.shape if isinstance(t, torch.Tensor) else 'N/A'}. Reinitializing.")
                 valid_elements = False
                 break
        if not valid_elements:
             self.state_history_deque.clear()
             for _ in range(Config.HISTORY_SIZE):
                 self.state_history_deque.append(torch.zeros(Config.STATE_DIM, device=DEVICE, dtype=torch.float32))
             return torch.zeros(Config.HISTORY_SIZE, Config.STATE_DIM, device=DEVICE, dtype=torch.float32)

        try:
             return torch.stack(list(self.state_history_deque)).to(device=DEVICE, dtype=torch.float32)
        except Exception as e:
             logger.error(f"Error stacking state history deque: {e}. Returning zeros.", exc_info=True)
             return torch.zeros(Config.HISTORY_SIZE, Config.STATE_DIM, device=DEVICE, dtype=torch.float32)

    def reset(self):
        # ... (reset method remains the same) ...
        """Resets the environment to its initial state."""
        self.step_count = 0; self.event_timer = 0; self.gap_timer = Config.EVENT_GAP; self.current_event_type = None; self.current_context = "Reset."
        self.state_history_deque.clear()
        for _ in range(Config.HISTORY_SIZE):
            self.state_history_deque.append(torch.zeros(Config.STATE_DIM, device=DEVICE, dtype=torch.float32))

        initial_emotions = self.base_intensities.clone()
        initial_state = self._get_state(initial_emotions)

        if initial_state is not None and is_safe(initial_state) and initial_state.shape == (Config.STATE_DIM,):
            # Add the valid initial state to the deque
            self.state_history_deque.popleft() # Remove one zero placeholder
            self.state_history_deque.append(initial_state.clone().detach())
        else:
            logger.error(f"Failed to get valid initial state in reset. Got: {initial_state}. History starts with zeros.");
            initial_state = self.state_history_deque[-1].clone() # Use the zero state

        return initial_state

    def step(self):
        """Performs one step of the *internal* environment simulation."""
        self.step_count += 1; reward = 0.0; event_triggered = False
        context_internal = "Quiet." # Default internal context

        if self.event_timer > 0:
            self.event_timer -= 1;
            if self.event_timer == 0:
                self.current_event_type = None; context_internal = "Normal."; self.gap_timer = Config.EVENT_GAP
            else:
                context_internal = self.current_context # Keep event context while timer runs
        elif self.gap_timer > 0:
            self.gap_timer -= 1
        else:
            if random.random() < Config.EVENT_FREQ:
                event_data = self.event_manager.get_next_event()
                if event_data and event_data[0] is not None:
                    event_type, event_context_text, base_reward = event_data
                    context_internal = event_context_text # Use context from the event
                    try:
                        lower_emotion_names = [name.lower() for name in self.emotion_names]
                        idx = lower_emotion_names.index(event_type)
                        self.current_event_type = event_type;
                        self.current_context = context_internal; self.event_timer = Config.EVENT_DURATION; event_triggered = True;
                        reward = base_reward * self.base_intensities[idx].item() * 1.5
                    except (ValueError, IndexError) as e:
                        logger.error(f"Event Trigger Error finding index for '{event_type}': {e}. Using base reward.");
                        self.current_event_type = event_type; self.current_context = context_internal; self.event_timer = Config.EVENT_DURATION; event_triggered = True; reward = base_reward

        # If event ended, context goes quiet
        if self.current_event_type is None and self.event_timer == 0 and self.current_context != "Quiet.":
            self.current_context = "Quiet."
            context_internal = "Quiet." # Ensure context is updated

        current_emotions = self.base_intensities.clone()
        if self.current_event_type:
            try:
                lower_emotion_names = [name.lower() for name in self.emotion_names]
                idx = lower_emotion_names.index(self.current_event_type)
                boost_factor = (self.event_timer / max(1.0, Config.EVENT_DURATION))
                boost_magnitude = 0.6
                current_emotions[idx] = torch.clamp(current_emotions[idx] + boost_magnitude * boost_factor, 0, 1)
            except (ValueError, IndexError) as e:
                logger.error(f"State Update Error finding index for '{self.current_event_type}': {e}")

        state = self._get_state(current_emotions)
        last_valid_state = self.state_history_deque[-1].clone()

        if state is None or not state.shape == (Config.STATE_DIM,):
            logger.error(f"Failed generate valid state in step. Got {state}. Returning previous state.");
            state = last_valid_state
            if not is_safe(state): state = torch.zeros(Config.STATE_DIM, device=DEVICE)
            reward = 0.0
        else:
            # Add new state to deque (automatically handles maxlen)
            self.state_history_deque.append(state.clone().detach())

        done = self.step_count >= 50000

        # Return internal context/event
        return state, reward, done, context_internal, self.current_event_type

    def _get_state(self, current_emotions):
        # ... (_get_state method remains the same) ...
        """Generates the full state vector (emotions + meta features)."""
        if not isinstance(current_emotions, torch.Tensor):
            logger.error(f"_get_state: Invalid emotion type {type(current_emotions)}."); return None
        if current_emotions.device != DEVICE: current_emotions = current_emotions.to(DEVICE)
        if not is_safe(current_emotions):
            logger.warning("_get_state: Unsafe emotion input. Using zeros."); current_emotions = torch.zeros(Config.EMOTION_DIM, device=DEVICE)
        if current_emotions.shape != (Config.EMOTION_DIM,):
            logger.warning(f"_get_state: Emotion shape {current_emotions.shape} != ({Config.EMOTION_DIM},). Padding/Truncating.");
            padded_emotions = torch.zeros(Config.EMOTION_DIM, device=DEVICE);
            flat_input = current_emotions.flatten()
            copy_len = min(flat_input.numel(), Config.EMOTION_DIM);
            padded_emotions[:copy_len] = flat_input[:copy_len];
            current_emotions = padded_emotions

        event_idx = -1;
        if self.current_event_type:
            try:
                lower_emotion_names = [name.lower() for name in self.emotion_names]
                event_idx = lower_emotion_names.index(self.current_event_type)
            except (ValueError, IndexError): logger.warning(f"_get_state: Cannot find index for current event '{self.current_event_type}'.")

        meta_step = min(1.0, self.step_count / 10000.0);
        meta_evt_timer = self.event_timer / max(1.0, Config.EVENT_DURATION);
        meta_gap_timer = self.gap_timer / max(1.0, Config.EVENT_GAP);
        meta_is_evt = 1.0 if self.current_event_type else 0.0;
        meta_evt_idx = (event_idx + 1) / max(1.0, len(self.emotion_names));
        meta_avg_emo = current_emotions.mean().item() if current_emotions.numel() > 0 else 0.0

        try:
            state_meta = torch.tensor([meta_step, meta_evt_timer, meta_gap_timer, meta_is_evt, meta_evt_idx, meta_avg_emo], device=DEVICE, dtype=torch.float32)
        except Exception as e:
            logger.error(f"_get_state: Failed to create meta tensor: {e}."); return None

        try:
             state_combined = torch.cat([current_emotions, state_meta])
        except Exception as e:
            logger.error(f"_get_state: Failed to concat emotions and meta: {e}."); return None

        if state_combined.shape[0] != Config.STATE_DIM:
            logger.error(f"CRITICAL STATE DIM MISMATCH! Actual:{state_combined.shape[0]}, Config.STATE_DIM:{Config.STATE_DIM}. Padding/Truncating to Config.STATE_DIM.");
            final_state = torch.zeros(Config.STATE_DIM, device=DEVICE, dtype=torch.float32);
            copy_len = min(state_combined.shape[0], Config.STATE_DIM); final_state[:copy_len] = state_combined[:copy_len];
            state_combined = final_state

        discretized_state = self.lattice.discretize(state_combined)
        if not is_safe(discretized_state) or discretized_state.shape != (Config.STATE_DIM,):
            logger.error(f"_get_state: Final discretized state is unsafe or has wrong shape {discretized_state.shape}. Returning None."); return None
        return discretized_state

    # --- NEW: Add text impact method ---
    def get_emotional_impact_from_text(self, text):
        """
        Analyzes text for keywords and returns an emotional impact vector.
        This vector represents the target emotion state based *only* on the text.
        """
        if not isinstance(text, str): return torch.zeros(Config.EMOTION_DIM, device=DEVICE)
        text_lower = text.lower()
        # Start with a neutral base impact
        impact_vector = torch.tensor([0.3, 0.1, 0.4, 0.1, 0.5, 0.2], device=DEVICE) # Base: slightly curious/calm

        found_emotion = False
        triggered_strengths = []
        for emo_idx, (keywords, strength) in self.emotion_keywords.items():
            if emo_idx >= Config.EMOTION_DIM: continue # Skip if index out of bounds

            for keyword in keywords:
                if keyword in text_lower:
                    # Set the target value for this emotion
                    impact_vector[emo_idx] = strength
                    triggered_strengths.append(strength)
                    found_emotion = True
                    # Break after first keyword match per emotion category
                    break

        # If no specific keywords found, return a more neutral vector
        if not found_emotion:
            impact_vector = torch.tensor([0.3, 0.2, 0.4, 0.2, 0.4, 0.3], device=DEVICE) # Slightly more neutral

        # Clamp values to be safe [0, 1]
        impact_vector = torch.clamp(impact_vector, 0.0, 1.0)
        logger.debug(f"Emotional impact for '{text[:30]}...': {impact_vector.cpu().numpy()}")
        return impact_vector

    def update_params(self, event_freq, intensities):
        # ... (update_params method remains the same) ...
        """Updates environment parameters (event frequency and base emotion intensities)."""
        Config.EVENT_FREQ = max(0.0, min(1.0, float(event_freq)))

        if isinstance(intensities, (list, tuple, np.ndarray)) and len(intensities) == Config.EMOTION_DIM:
            try:
                clamped_intensities = torch.clamp(torch.tensor(intensities, device=DEVICE, dtype=torch.float32), 0, 1);
                if is_safe(clamped_intensities):
                    self.base_intensities = clamped_intensities
                else:
                    logger.warning("Intensity update failed: Resulting tensor is unsafe. Keeping previous values.")
            except Exception as e: logger.error(f"Intensity update failed during tensor conversion: {e}")
        else:
            logger.warning(f"Intensity update failed: Expected {Config.EMOTION_DIM} values, got {len(intensities)} type {type(intensities)}")


# --- END OF FILE environment.py ---