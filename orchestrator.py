# --- START OF FILE orchestrator.py ---

# --- START OF FILE orchestrator.py ---
import torch
from typing import Dict, Tuple, Optional, List, Union, Deque, Any
import concurrent.futures
import asyncio
import time
import math
from collections import deque
import sys
import numpy as np
import pickle # For saving replay buffer
import html # Import html for escaping
import os # Added for save/load path checks

try: from sentence_transformers import SentenceTransformer; SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError: SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import MasterConfig as Config
from config import DEVICE, logger
# Import Save Paths
from config import AGENT_SAVE_PATH, GPT_SAVE_PATH, OPTIMIZER_SAVE_PATH, TARGET_NET_SAVE_SUFFIX, REPLAY_BUFFER_SAVE_PATH
# Import Head Movement Labels & Mappings
from config import HEAD_MOVEMENT_LABELS, HEAD_MOVEMENT_TO_IDX, IDX_TO_HEAD_MOVEMENT
from environment import EmotionalSpace
from agent import ConsciousAgent
from graphics import Live2DCharacter
# Import updated Experience tuple
from utils import is_safe, Experience, MetaCognitiveMemory

ReflectReturnType = Dict[str, Union[float, List[float], int, str, np.ndarray]]
TrainStepReturnType = Tuple[ReflectReturnType, float, bool, str, float, str, str]


class EnhancedConsciousAgent:
    def __init__(self, train_interval: int = Config.RL.AGENT_TRAIN_INTERVAL, batch_size: int = Config.RL.AGENT_BATCH_SIZE, num_workers: int = 1):
        logger.info("Initializing Orchestrator (Async Learn + Chat + Mood + ConvHistory + MemGate + LangEmbed + HM Predict)...")
        self.env = EmotionalSpace()
        self.model = ConsciousAgent().to(DEVICE) # Initializes TransformerGPT inside
        self.avatar = Live2DCharacter()
        logger.debug("Live2DCharacter display instance created.")
        self.st_model: Optional[SentenceTransformer] = None
        if Config.Agent.USE_LANGUAGE_EMBEDDING:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    model_name = Config.NLP.SENTENCE_TRANSFORMER_MODEL
                    logger.info(f"Loading sentence transformer: {model_name}...")
                    self.st_model = SentenceTransformer(model_name, device=str(DEVICE))
                    with torch.no_grad():
                        test_emb = self.st_model.encode(["test"], convert_to_tensor=True)[0]
                    actual_dim = test_emb.shape[0]
                    expected_dim = Config.Agent.LANGUAGE_EMBEDDING_DIM
                    if actual_dim != expected_dim:
                         logger.critical(f"FATAL: ST model dim ({actual_dim}) != Config dim ({expected_dim}). Check model name '{model_name}' or config LANGUAGE_EMBEDDING_DIM.")
                         sys.exit(1)
                    logger.info(f"Sentence transformer loaded to {self.st_model.device}. Output dim: {actual_dim}.")
                except Exception as e:
                    logger.error(f"Failed load ST model '{Config.NLP.SENTENCE_TRANSFORMER_MODEL}': {e}. Disabling.", exc_info=True)
                    self.st_model = None
                    Config.Agent.USE_LANGUAGE_EMBEDDING = False
                    # Use dataclass method directly to trigger recalculation
                    Config.Agent.__post_init__() # Recalculate STATE_DIM
                    logger.warning(f"Lang embedding disabled. Agent STATE_DIM recalced: {Config.Agent.STATE_DIM}")
            else:
                 logger.warning("SentenceTransformers not found. Disabling language embedding.")
                 Config.Agent.USE_LANGUAGE_EMBEDDING = False
                 Config.Agent.__post_init__() # Recalculate STATE_DIM
                 logger.warning(f"Lang embedding disabled. Agent STATE_DIM recalced: {Config.Agent.STATE_DIM}")
        else:
             logger.info("Language embedding in state is disabled via config.")
        self.train_interval = max(1, train_interval); self.batch_size = batch_size
        self.episode_rewards: List[float] = []; self.current_episode_reward_sum: float = 0.0
        self.total_steps: int = 0; self.episode_steps: int = 0; self.episode_count: int = 0
        self.last_reward: float = 0.0; self.last_reported_loss: float = 0.0; self.learn_step_running: bool = False
        self.last_response_emotions: torch.Tensor = self.model.prev_emotions.clone().detach()
        self.current_response: str = "Initializing..."
        self.mood: torch.Tensor = torch.tensor([0.4, 0.1, 0.5, 0.1, 0.6, 0.2][:Config.Agent.EMOTION_DIM], device=DEVICE, dtype=torch.float32)
        if self.mood.shape[0] != Config.Agent.EMOTION_DIM: self.mood = torch.ones(Config.Agent.EMOTION_DIM, device=DEVICE) * 0.3
        self.last_internal_monologue: str = ""
        self.conversation_history: Deque[Tuple[str, str]] = deque(maxlen=Config.NLP.CONVERSATION_HISTORY_LENGTH * 2)
        emb_dim = Config.Agent.LANGUAGE_EMBEDDING_DIM if Config.Agent.USE_LANGUAGE_EMBEDDING else 0
        self.last_text_embedding: torch.Tensor = torch.zeros(emb_dim, device=DEVICE, dtype=torch.float32)
        self.num_workers = max(1, num_workers); self.learn_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix='LearnWorker'); self.learn_future: Optional[concurrent.futures.Future] = None
        self.hud_widget = None
        try:
             self.current_base_state: torch.Tensor = self.env.reset()
             if not is_safe(self.current_base_state) or self.current_base_state.shape[0] != Config.Agent.BASE_STATE_DIM: raise RuntimeError(f"Initial base state invalid. Shape: {self.current_base_state.shape}")
             self.current_state = self._get_combined_state(self.current_base_state, "init")
             if self.current_state.shape[0] != Config.Agent.STATE_DIM: raise RuntimeError(f"Initial combined state dim error! Expected {Config.Agent.STATE_DIM}, got {self.current_state.shape}.")
        except Exception as e: logger.critical(f"Orchestrator Init Error: {e}", exc_info=True); raise RuntimeError("Init state failed.") from e
        self.current_event: Optional[str] = self.env.current_event_type
        logger.info(f"Orchestrator initialized (LangEmbed={'Enabled' if Config.Agent.USE_LANGUAGE_EMBEDDING else 'Disabled'}, StateDim={Config.Agent.STATE_DIM}).")

    def _get_combined_state(self, base_state: torch.Tensor, context: str = "step") -> torch.Tensor:
        """Combines base state with language embedding if enabled."""
        if base_state is None or not isinstance(base_state, torch.Tensor): logger.error(f"[{context}] Invalid base_state: {type(base_state)}. Zeros."); return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)
        base_state_dev = base_state.to(DEVICE)
        if base_state_dev.shape[0] != Config.Agent.BASE_STATE_DIM: logger.error(f"[{context}] Base state dim mismatch! Exp {Config.Agent.BASE_STATE_DIM}, got {base_state_dev.shape}. Padding/Truncating."); padded_base = torch.zeros(Config.Agent.BASE_STATE_DIM, device=DEVICE, dtype=base_state_dev.dtype); copy_len = min(base_state_dev.shape[0], Config.Agent.BASE_STATE_DIM); padded_base[:copy_len] = base_state_dev[:copy_len]; base_state_dev = padded_base

        if Config.Agent.USE_LANGUAGE_EMBEDDING:
            # Use the stored last text embedding
            if self.last_text_embedding is None or self.last_text_embedding.shape[0] != Config.Agent.LANGUAGE_EMBEDDING_DIM:
                logger.warning(f"[{context}] Invalid last_text_embedding. Using zeros.")
                current_embedding = torch.zeros(Config.Agent.LANGUAGE_EMBEDDING_DIM, device=DEVICE, dtype=torch.float32)
            else:
                current_embedding = self.last_text_embedding.to(DEVICE)

            try: combined = torch.cat((base_state_dev, current_embedding), dim=0)
            except Exception as e: logger.error(f"[{context}] Error cat base ({base_state_dev.shape}) embed ({current_embedding.shape}): {e}. Zeros."); return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)

            if combined.shape[0] != Config.Agent.STATE_DIM: logger.error(f"[{context}] Combined dim error! Exp {Config.Agent.STATE_DIM}, got {combined.shape}. Zeros."); return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)
            return combined
        else:
             # If language embedding is disabled, the base state dim MUST match the total state dim
             if base_state_dev.shape[0] != Config.Agent.STATE_DIM:
                 logger.critical(f"[{context}] Lang embed disabled, but BASE_STATE_DIM ({base_state_dev.shape[0]}) != STATE_DIM ({Config.Agent.STATE_DIM}). Critical config error!");
                 # Attempt graceful exit or return zero tensor, but this is fatal
                 # sys.exit(1) # Or raise an exception
                 return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=base_state_dev.dtype)
             return base_state_dev

    def set_hud_widget(self, hud_widget): self.hud_widget = hud_widget; logger.debug("HUD widget ref stored.")

    def _run_learn_task(self) -> Optional[float]:
        """Background task for agent learning step."""
        loss = None
        try:
            if len(self.model.memory) >= self.batch_size:
                 loss = self.model.learn(self.batch_size)
            else:
                 loss = 0.0 # Indicate no learning happened due to insufficient samples
        except Exception as e:
             logger.error(f"Exception in learn task thread: {e}", exc_info=True)
             loss = -1.0 # Indicate error
        return loss

    def _check_learn_future(self):
        """Checks the status of the background learning task."""
        if self.learn_future and self.learn_future.done():
            try:
                loss_result = self.learn_future.result()
                if loss_result is not None and loss_result >= 0:
                    self.last_reported_loss = loss_result
                elif loss_result is not None: # Negative value indicates an error
                    logger.warning(f"Async learn task indicated an error (Loss: {loss_result}).")
                    self.last_reported_loss = -1.0
                else: # Result was None, unexpected
                    logger.warning("Async learn task returned None.")
                    self.last_reported_loss = -1.0
            except concurrent.futures.CancelledError:
                logger.warning("Learn task was cancelled.")
                self.last_reported_loss = -1.0
            except Exception as e:
                logger.error(f"Exception retrieving learn task result: {e}", exc_info=True)
                self.last_reported_loss = -1.0
            finally:
                self.learn_future = None
                self.learn_step_running = False # Mark as no longer running

    def train_step(self) -> TrainStepReturnType:
        """Performs one environment step, agent step, memory storage, and triggers learning."""
        self._check_learn_future() # Check if previous learn task finished

        # --- State Validation ---
        if self.current_state is None or not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM:
            logger.error(f"Orchestrator has invalid combined state before step ({getattr(self.current_state, 'shape', 'None')}). Attempting reset.");
            try:
                self.current_base_state = self.env.reset();
                if not is_safe(self.current_base_state) or self.current_base_state.shape[0] != Config.Agent.BASE_STATE_DIM: raise RuntimeError("Reset failed: invalid base state.")
                self.last_text_embedding.zero_(); self.current_state = self._get_combined_state(self.current_base_state, "reset_recovery")
                if not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM: raise RuntimeError("Reset failed: invalid combined state after embedding.")
                logger.info("State reset successful after invalid state detected.")
                # Reset other relevant states
                self.episode_steps = 0; self.current_episode_reward_sum = 0.0; self.model.prev_emotions.zero_(); self.last_response_emotions.zero_()
            except Exception as reset_err:
                 logger.critical(f"CRITICAL RESET FAILED during train_step: {reset_err}. Stopping simulation cannot continue.", exc_info=True);
                 # Return an error tuple matching TrainStepReturnType structure
                 default_metrics = {'error': True, 'message':"FATAL STATE RESET", "last_monologue":"", "embedding_norm": 0.0, "base_state_norm": 0.0}
                 return (default_metrics, -1.0, True, "FATAL STATE", self.last_reported_loss, "", "idle")

        # --- Environment Step ---
        self.total_steps += 1; self.episode_steps += 1
        state_before_step_combined = self.current_state.clone().detach()
        base_state_before_step = self.current_base_state.clone().detach()

        try: next_base_state, reward, env_done, context_internal, event_type = self.env.step()
        except Exception as e: logger.error(f"Error during environment step: {e}", exc_info=True); next_base_state = base_state_before_step; reward = -0.1; env_done = False; context_internal = "Error env step."; event_type = self.current_event
        if not isinstance(next_base_state, torch.Tensor) or next_base_state.shape[0] != Config.Agent.BASE_STATE_DIM or not is_safe(next_base_state): logger.error(f"Environment returned invalid next_base_state (shape {getattr(next_base_state, 'shape', 'None')}). Using previous."); next_base_state = base_state_before_step; reward = -0.1; env_done = False; context_internal = "Env State Error"; event_type = self.current_event
        self.last_reward = reward; self.current_event = event_type; self.current_episode_reward_sum += reward

        # --- Agent Step (Inference) ---
        att_score_metric = 0.0; response_internal = "(Agent step inference error)"; emotions_internal = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE); predicted_hm_label = "idle"
        belief_for_memory = None
        try:
             # Pass current combined state, reward, history, and env context
             emotions_internal, response_internal, belief_for_memory, att_score_metric, predicted_hm_label = self.model.step(state_before_step_combined, reward, self.model.state_history, context_internal)
        except Exception as e:
             logger.error(f"Error during agent inference step: {e}", exc_info=True);
             emotions_internal = self.model.prev_emotions if is_safe(self.model.prev_emotions) else torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE);
             # Try to create a default belief with correct dimension
             belief_dim = getattr(self.model.kaskade, '_output_dim', Config.Agent.HIDDEN_DIM)
             belief_for_memory = torch.zeros(belief_dim, device=DEVICE);
             att_score_metric = 0.0; predicted_hm_label = "idle"

        # Update internal response tracking
        self.current_response = response_internal
        self.last_response_emotions = emotions_internal.clone().detach() # Track emotions tied to this response

        # --- Internal Monologue & Language Embedding Update ---
        self.last_internal_monologue = ""; current_monologue_embedding = torch.zeros_like(self.last_text_embedding)
        if Config.Agent.USE_LANGUAGE_EMBEDDING and self.st_model:
             try:
                 monologue_context = f"Internal state: {context_internal}. Feeling:"; temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7); top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9)
                 self.last_internal_monologue = self.model.gpt.generate(
                     context=monologue_context,
                     emotions=emotions_internal, # Use emotions from this step
                     temperature=temp, top_p=top_p, max_len=16
                 )
                 if self.last_internal_monologue and self.last_internal_monologue != "...":
                     with torch.no_grad(): emb = self.st_model.encode([self.last_internal_monologue], convert_to_tensor=True, device=DEVICE);
                     if emb.ndim > 1: emb = emb.squeeze(0)
                     if emb.shape[0] == Config.Agent.LANGUAGE_EMBEDDING_DIM: current_monologue_embedding = emb.float()
                     else: logger.warning(f"Monologue embedding dim mismatch! Got {emb.shape}")
                 self.last_text_embedding = current_monologue_embedding.clone().detach() # Store the new embedding
             except Exception as e: logger.error(f"Error generating/embedding internal monologue: {e}"); self.last_text_embedding.zero_()
        else:
             self.last_text_embedding.zero_() # Zero out if embedding is disabled

        # --- Prepare for Memory ---
        next_combined_state = self._get_combined_state(next_base_state, "experience_next")
        initial_td_error = MetaCognitiveMemory.INITIAL_TD_ERROR # Default priority if calculation fails
        if is_safe(state_before_step_combined) and is_safe(next_combined_state):
            try:
                self.model.eval() # Ensure eval mode for consistent TD error estimate
                with torch.no_grad():
                     # Estimate V(s) with ONLINE net, V(s') with TARGET net
                     outputs_s = self.model.forward(state_before_step_combined.unsqueeze(0), torch.tensor([[reward]], device=DEVICE), None, use_target=False)
                     outputs_sp = self.model.forward(next_combined_state.unsqueeze(0), torch.tensor([[0.0]], device=DEVICE), None, use_target=True)
                     if len(outputs_s) == 13 and len(outputs_sp) == 13:
                         current_value = outputs_s[3].squeeze(); next_value = outputs_sp[3].squeeze()
                         if is_safe(current_value) and is_safe(next_value):
                             target_val = reward + Config.RL.GAMMA * next_value * (0.0 if env_done else 1.0);
                             initial_td_error = abs((target_val - current_value).item())
                         else: logger.warning("Unsafe V(s) or V(s') in TD error estimation. Using default.")
                     else: logger.warning(f"Forward output length mismatch for TD estimate ({len(outputs_s)}, {len(outputs_sp)}). Using default.")
                self.model.train() # Set back to train mode
            except Exception as e: logger.warning(f"Could not estimate initial TD error: {e}. Using default.")
        else: logger.warning("Skipping TD error estimation due to unsafe state(s). Using default.")

        # --- Add to Memory (Gated) ---
        should_store_long_term = not Config.RL.MEMORY_GATING_ENABLED or att_score_metric >= Config.RL.MEMORY_GATE_ATTENTION_THRESHOLD
        if should_store_long_term:
            try:
                belief_to_store = belief_for_memory if isinstance(belief_for_memory, torch.Tensor) and is_safe(belief_for_memory) else None
                # Calculate final priority including attention boost
                base_priority = initial_td_error + 1e-5;
                attention_factor = 1.0 + Config.RL.PRIORITY_ATTENTION_WEIGHT * max(0.0, min(1.0, att_score_metric));
                final_priority = base_priority * attention_factor
                # Get target head movement index for the state BEFORE the step
                target_hm_idx = HEAD_MOVEMENT_TO_IDX.get(predicted_hm_label, HEAD_MOVEMENT_TO_IDX["idle"])
                # Create Experience tuple
                exp = Experience(
                    state=state_before_step_combined.detach(), # State before action
                    belief=belief_to_store.detach() if belief_to_store is not None else None, # Belief for that state
                    reward=reward,                           # Reward received after action
                    next_state=next_combined_state.detach(), # Resulting state
                    done=env_done,                           # Was it terminal?
                    td_error=final_priority,                 # Priority for PER
                    head_movement_idx=target_hm_idx          # Target movement for state_before
                )
                self.model.memory.add(exp)
            except Exception as e: logger.error(f"Failed adding experience to memory: {e}", exc_info=True)

        # --- Trigger Learning Task ---
        if not self.learn_step_running and (self.total_steps % self.train_interval == 0) and (len(self.model.memory) >= self.batch_size):
            try:
                self.learn_future = self.learn_executor.submit(self._run_learn_task);
                self.learn_step_running = True # Mark as running
            except Exception as e: logger.error(f"Failed to submit learning task: {e}"); self.learn_step_running = False

        # --- Update Current State for Next Cycle ---
        self.current_base_state = next_base_state.detach().clone()
        self.current_state = self._get_combined_state(self.current_base_state, "state_update") # Update with latest embedding

        # --- Update Mood ---
        try:
            internal_emotion_this_step = emotions_internal.detach()
            if is_safe(internal_emotion_this_step) and is_safe(self.mood):
                decay = Config.RL.MOOD_UPDATE_DECAY;
                self.mood = decay * self.mood + (1.0 - decay) * internal_emotion_this_step;
                self.mood = torch.clamp(self.mood, 0.0, 1.0)
            elif not is_safe(self.mood): logger.warning("Mood became unsafe. Resetting."); self.mood.fill_(0.3)
        except Exception as e: logger.error(f"Error updating mood: {e}")

        # --- Episode Reset Handling ---
        done = env_done
        if done:
            logger.info(f"--- Episode {self.episode_count + 1} ended (Steps: {self.episode_steps}, Reward: {self.current_episode_reward_sum:.2f}) ---")
            self.episode_rewards.append(self.current_episode_reward_sum); self.episode_count += 1
            try:
                self.current_base_state = self.env.reset();
                if not is_safe(self.current_base_state) or self.current_base_state.shape[0] != Config.Agent.BASE_STATE_DIM: raise RuntimeError("Reset invalid base state.")
                self.last_text_embedding.zero_(); self.current_state = self._get_combined_state(self.current_base_state, "episode_reset")
                if not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM: raise RuntimeError("Reset invalid combined state after embedding.")
                # Reset other relevant states
                self.model.prev_emotions.zero_(); self.last_response_emotions.zero_(); self.mood.fill_(0.3); self.conversation_history.clear(); self.episode_steps = 0; self.current_episode_reward_sum = 0.0; self.current_event = self.env.current_event_type;
            except Exception as e:
                 logger.critical(f"CRITICAL: Failed episode reset: {e}. Stopping simulation.", exc_info=True); done = True; # Ensure done is True
                 # Return error tuple
                 default_metrics = {'error': True, 'message':"FATAL EPISODE RESET", "last_monologue":"", "embedding_norm": 0.0, "base_state_norm": 0.0}
                 return (default_metrics, reward, done, "FATAL RESET", self.last_reported_loss, "", "idle")

        # --- Return Results ---
        metrics_dict = self.reflect()
        # The response here is the one generated *during* this train_step (internal or error)
        # The predicted_hm_label is also the one from this step
        return metrics_dict, reward, done, self.current_response, self.last_reported_loss, self.last_internal_monologue, predicted_hm_label

    # --- MODIFIED handle_user_chat (with Improved Context & unescape) ---
    def handle_user_chat(self, user_text: str) -> str:
        """Handles user chat input, updates state, generates response, and predicts movement."""
        logger.info(f"Handling user chat: '{user_text[:50]}...'")
        if not isinstance(user_text, str) or not user_text.strip(): return "..."

        # --- Escape user text EARLY ---
        safe_user_text = html.escape(user_text.replace('\n', ' ').strip())
        # ---

        # --- Update Language Embedding based on user input ---
        user_text_embedding = torch.zeros_like(self.last_text_embedding)
        if Config.Agent.USE_LANGUAGE_EMBEDDING and self.st_model:
            try:
                with torch.no_grad(): emb = self.st_model.encode([safe_user_text], convert_to_tensor=True, device=DEVICE); # Use escaped text
                if emb.ndim > 1: emb = emb.squeeze(0)
                if emb.shape[0] == Config.Agent.LANGUAGE_EMBEDDING_DIM: user_text_embedding = emb.float()
                else: logger.warning(f"User text embed dim mismatch! Got {emb.shape}")
            except Exception as e: logger.error(f"Error embedding user text: {e}")
        self.last_text_embedding = user_text_embedding.clone().detach() # Store user text embedding
        # --- Update combined state AFTER getting embedding ---
        self.current_state = self._get_combined_state(self.current_base_state, "chat_input")
        # --- Add user turn to history (AFTER state update) ---
        self.conversation_history.append(("User", safe_user_text)) # Store escaped text

        try:
            # --- Emotion Blending based on user text ---
            impact_vector = self.env.get_emotional_impact_from_text(safe_user_text) # Use escaped text
            current_response_emotions = self.last_response_emotions.clone().detach() # Use last displayed emotions as base
            mood_influence = 0.15
            # Blend current response emotions with mood first
            biased_current_emotions = torch.clamp( current_response_emotions * (1.0 - mood_influence) + self.mood * mood_influence, 0.0, 1.0 )
            # Blend the mood-influenced emotions with the impact from user text
            current_max_emo = biased_current_emotions.max().item(); blend_factor_user = max(0.4, min(0.7, 0.7 - (current_max_emo * 0.3))); blend_factor_current = 1.0 - blend_factor_user
            blended_emotions = torch.clamp( biased_current_emotions * blend_factor_current + impact_vector * blend_factor_user, 0.0, 1.0 )
            if not is_safe(blended_emotions): logger.warning("Blended chat emotions unsafe."); blended_emotions = biased_current_emotions # Fallback

            # Update agent's internal and tracked response emotions
            self.last_response_emotions = blended_emotions.clone().detach() # This will be displayed
            self.model.prev_emotions = torch.clamp(self.model.prev_emotions * 0.5 + self.last_response_emotions * 0.5, 0.0, 1.0) # Also influence agent's internal
            if self.avatar: self.avatar.update_emotions(self.last_response_emotions) # Update avatar display
            # --- END Emotion Blending ---

            # --- Context Building for GPT ---
            context_for_gpt = ""
            history_turns = list(self.conversation_history) # Includes the latest user turn
            temp_hist_context = ""
            max_hist_len = 512 # Max chars for history part

            # Build history string, ending with the last AI response (if any)
            for speaker, text in reversed(history_turns):
                turn = f"{speaker}: {text}\n" # Text is already escaped
                if len(temp_hist_context) + len(turn) > max_hist_len:
                    break
                temp_hist_context = turn + temp_hist_context

            # The final prompt should end with "AI:" to signal generation start
            context_for_gpt = temp_hist_context.strip()
            if not context_for_gpt.endswith("\nAI:"):
                 # Check if it ends with "AI:" without newline (e.g., first turn)
                 if context_for_gpt.endswith("AI:"):
                      context_for_gpt = context_for_gpt[:-3].strip() # Remove trailing AI:
                 context_for_gpt += "\nAI:" # Add newline and AI: prompt
            # --- End Context Building ---


            # --- Generate Response ---
            logger.debug(f"Context sent to GPT: '{context_for_gpt[:200]}...'")
            temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7); top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9)
            raw_response = self.model.gpt.generate(
                context=context_for_gpt,
                emotions=self.last_response_emotions, # Pass current emotions
                temperature=temp,
                top_p=top_p,
                max_len=Config.NLP.MAX_RESPONSE_LEN
            )
            # --- Unescape the RAW response from the model ---
            response_unescaped = html.unescape(raw_response)
            # --- Clean up for storage and display ---
            final_response_clean = response_unescaped.replace('\n', ' ').strip()
            if not final_response_clean: final_response_clean = "..." # Fallback

            self.current_response = final_response_clean # Store the clean, unescaped version as the current response
            # --- Add the CLEAN version to history ---
            self.conversation_history.append(("AI", self.current_response))
            # ---

            # --- Predict Head Movement based on current state AFTER processing chat ---
            predicted_chat_hm_label = "idle"
            if self.current_state is not None and is_safe(self.current_state) and self.current_state.shape[0] == Config.Agent.STATE_DIM:
                try:
                    self.model.eval()
                    with torch.no_grad():
                         # Use current state (which includes user text embedding) and 0 reward
                         outputs = self.model.forward(self.current_state, 0.0, self.model.state_history, use_target=False)
                         if len(outputs) == 13:
                              hm_logits = outputs[-1]
                              if hm_logits.ndim == 1: idx = torch.argmax(hm_logits).item()
                              elif hm_logits.ndim == 2 and hm_logits.shape[0] == 1 : idx = torch.argmax(hm_logits.squeeze(0)).item()
                              else: logger.warning(f"Unexpected hm_logits shape in chat: {hm_logits.shape}"); idx = HEAD_MOVEMENT_TO_IDX["idle"]
                              predicted_chat_hm_label = IDX_TO_HEAD_MOVEMENT.get(idx, "idle")
                         else: logger.warning(f"Forward mismatch during chat HM prediction ({len(outputs)}).")
                    self.model.train()
                except Exception as e:
                     logger.error(f"Error predicting head movement after chat: {e}")
            else: logger.warning("Skipping chat HM prediction due to invalid current_state.")


            # --- Update Avatar Movement ---
            if self.avatar and hasattr(self.avatar, 'update_predicted_movement'):
                self.avatar.update_predicted_movement(predicted_chat_hm_label)

            logger.debug(f"Chat response: '{self.current_response}' | Emotions: {self.last_response_emotions.cpu().numpy().round(2)} | Head Move: {predicted_chat_hm_label}")
            # Return the clean, unescaped response for display
            return self.current_response
        except Exception as e: logger.error(f"Error handling user chat: {e}", exc_info=True); self.current_response = "Sorry, I had trouble processing that."; return self.current_response
    # --- END MODIFIED handle_user_chat ---


    def reflect(self) -> ReflectReturnType:
        """Gathers and returns the current internal state and metrics of the agent."""
        stats_dict: ReflectReturnType = { "avg_reward_last20": 0.0, "total_steps": self.total_steps, "episode": self.episode_count, "current_emotions_internal": [0.0]*Config.Agent.EMOTION_DIM, "current_emotions_response": [0.0]*Config.Agent.EMOTION_DIM, "current_mood": [0.0]*Config.Agent.EMOTION_DIM, "last_monologue": self.last_internal_monologue, "I_S": 0.0, "rho_struct": 0.0, "att_score": 0.0, "self_consistency": 0.0, "rho_score": 0.0, "box_score": 0.0, "tau_t": 0.0, "R_acc": 0.0, "loss": self.last_reported_loss, "rho_struct_mem_short": 0.0, "rho_struct_mem_long": 0.0, "embedding_norm": 0.0, "base_state_norm": 0.0 }
        recent_rewards = self.episode_rewards[-20:];
        if recent_rewards: stats_dict["avg_reward_last20"] = sum(recent_rewards) / len(recent_rewards)
        try:
            current_combined_state = self.current_state
            if current_combined_state is not None and is_safe(current_combined_state) and current_combined_state.shape[0] == Config.Agent.STATE_DIM:
                self.model.eval();
                with torch.no_grad(): reflect_outputs = self.model.forward(current_combined_state, 0.0, self.model.state_history, use_target=False) # Use online for reflect
                if len(reflect_outputs) == 13:
                     # Unpack the 13 values correctly based on forward's return signature
                     (emotions_int, _, _, _value, I_S, rho_struct_val, att_score, self_consistency, rho_score, box_score, R_acc_mean, tau_t, _) = reflect_outputs;
                     stats_dict.update({ "current_emotions_internal": emotions_int.cpu().tolist() if is_safe(emotions_int) else [0.0]*Config.Agent.EMOTION_DIM,
                                        "I_S": I_S, "rho_struct": rho_struct_val, "att_score": att_score, # rho_struct from forward output
                                        "self_consistency": self_consistency, "rho_score": rho_score, "box_score": box_score,
                                        "tau_t": tau_t, "R_acc": R_acc_mean })
                else: logger.warning(f"Reflection: forward returned {len(reflect_outputs)} values, expected 13.")
                self.model.train();
            else: logger.warning(f"Reflection skipped: Invalid current_state.")
        except Exception as e: logger.error(f"Error during agent forward pass in reflect: {e}", exc_info=False);
        try:
             # Get memory norms separately
             stats_dict["rho_struct_mem_short"] = self.model.memory.get_short_term_norm();
             stats_dict["rho_struct_mem_long"] = self.model.memory.get_long_term_norm();
             # Combine memory norms for a final rho_struct value (potentially overwriting the one from forward if desired)
             stats_dict["rho_struct"] = stats_dict["rho_struct_mem_short"] * 0.3 + stats_dict["rho_struct_mem_long"] * 0.7

             stats_dict["current_emotions_response"] = self.last_response_emotions.cpu().tolist();
             stats_dict["current_mood"] = self.mood.cpu().tolist() if is_safe(self.mood) else [0.0]*Config.Agent.EMOTION_DIM
             if Config.Agent.USE_LANGUAGE_EMBEDDING and self.last_text_embedding is not None and is_safe(self.last_text_embedding): stats_dict["embedding_norm"] = torch.linalg.norm(self.last_text_embedding.float()).item()
             if self.current_base_state is not None and is_safe(self.current_base_state): stats_dict["base_state_norm"] = torch.linalg.norm(self.current_base_state.float()).item()
        except Exception as e: logger.error(f"Error getting memory norms/emo/mood/embed norm in reflect: {e}", exc_info=False)
        return stats_dict

    def test_completeness(self) -> Tuple[bool, str]:
        """Performs a Syntrometrie completeness test on the current state."""
        logger.info("Performing Completeness Test..."); current_combined_state = self.current_state
        if current_combined_state is None or not is_safe(current_combined_state) or current_combined_state.shape[0] != Config.Agent.STATE_DIM: return False, "Invalid current state for test"
        test_state = current_combined_state.clone().detach();
        # Modify state slightly to simulate a specific condition (e.g., high joy, low fear)
        if Config.Agent.EMOTION_DIM >= 2: test_state[0] = 0.9; test_state[1] = 0.1; # High Joy, Low Fear
        elif Config.Agent.EMOTION_DIM == 1: test_state[0] = 0.9 # High first emotion
        test_reward = 1.5; consistent = False; details = "Prerequisites failed."
        try:
            self.model.eval();
            with torch.no_grad(): test_outputs = self.model.forward(test_state, test_reward, self.model.state_history, use_target=False) # Use online network
            if len(test_outputs) == 13:
                 # Unpack based on forward signature
                 (emotions, _, _, _, _, _, att_score, _, rho_score, box_score, R_acc, _, _) = test_outputs
                 joy_check = False; joy_val = -1.0
                 if Config.Agent.EMOTION_DIM >= 1 and is_safe(emotions):
                     joy_val = emotions[0].item();
                     # Check if joy is high AND the dominant emotion
                     joy_check = joy_val > 0.5 and joy_val >= emotions.max().item() - 1e-6
                 att_check = att_score > Config.Agent.ATTENTION_THRESHOLD;
                 box_check = box_score > 0; # Simple check if box score is positive
                 consistent = joy_check and att_check and box_check # Define consistency criteria
                 details = (f"Joy={joy_val:.2f}(>{0.5}&max? {joy_check}), Att={att_score:.2f}(>{Config.Agent.ATTENTION_THRESHOLD}? {att_check}), Box={box_score:.2f}(>0? {box_check}), R_acc={R_acc:.2f}, RhoScr={rho_score:.2f}")
            else: details = f"Forward call returned incorrect number of items ({len(test_outputs)})."
            self.model.train();
        except Exception as e: logger.error(f"Error during completeness test execution: {e}"); details = f"Exception: {e}"
        logger.info(f"Completeness Test Result: {consistent}. Details: {details}"); return consistent, details

    def update_environment(self, event_freq: float, intensities: List[float]):
        """Updates environment parameters via the env object."""
        try: self.env.update_params(event_freq, intensities);
        except Exception as e: logger.error(f"Error updating environment parameters: {e}")

    # --- Save/Load Methods ---
    def save_agent(self):
        """Orchestrates saving the agent model, optimizer, and replay buffer."""
        logger.info("Orchestrator: Saving agent state...")
        # Save models and optimizer
        self.model.save_state(
            agent_path=AGENT_SAVE_PATH,
            gpt_path=GPT_SAVE_PATH, # Should point to a directory now
            optimizer_path=OPTIMIZER_SAVE_PATH,
            target_path_suffix=TARGET_NET_SAVE_SUFFIX
        )
        # Save replay buffer
        try:
            buffer_path = REPLAY_BUFFER_SAVE_PATH
            os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
            with open(buffer_path, 'wb') as f:
                pickle.dump(self.model.memory, f)
            logger.info(f"Replay buffer (Size: {len(self.model.memory)}) saved to {buffer_path}")
        except Exception as e:
            logger.error(f"Failed to save replay buffer: {e}", exc_info=True)

    def load_agent(self) -> bool:
        """Orchestrates loading the agent model, optimizer, and replay buffer. Returns True on success."""
        logger.info("Orchestrator: Loading agent state...")
        loaded_model = self.model.load_state(
            agent_path=AGENT_SAVE_PATH,
            gpt_path=GPT_SAVE_PATH, # Should point to a directory now
            optimizer_path=OPTIMIZER_SAVE_PATH,
            target_path_suffix=TARGET_NET_SAVE_SUFFIX
        )
        if loaded_model:
            # Load replay buffer
            try:
                buffer_path = REPLAY_BUFFER_SAVE_PATH
                if os.path.exists(buffer_path):
                    with open(buffer_path, 'rb') as f:
                        loaded_memory = pickle.load(f)
                        if isinstance(loaded_memory, MetaCognitiveMemory):
                            self.model.memory = loaded_memory
                            logger.info(f"Replay buffer loaded from {buffer_path}")
                            # Update PER beta based on loaded agent's step count
                            beta_frames = Config.RL.PER_BETA_FRAMES if Config.RL.PER_BETA_FRAMES > 0 else 1
                            self.model.beta = Config.RL.PER_BETA_START + (1.0 - Config.RL.PER_BETA_START) * min(1.0, self.model.step_count / beta_frames)
                            logger.info(f"Restored Replay Buffer (Size: {len(self.model.memory)}). Adjusted PER Beta to: {self.model.beta:.4f}")
                        else:
                            logger.error("Loaded object from replay buffer file is not a MetaCognitiveMemory instance. Starting fresh.")
                            self.model.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE)
                else:
                    logger.warning(f"Replay buffer file not found: {buffer_path}. Starting with empty buffer.")
                    self.model.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE) # Reinitialize if not found
            except ModuleNotFoundError as mnfe:
                 logger.error(f"Could not load replay buffer due to ModuleNotFoundError: {mnfe}. Likely a dependency mismatch or pickled custom class issue. Starting with empty buffer.")
                 self.model.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE) # Reinitialize
            except EOFError:
                logger.error(f"Error loading replay buffer: EOFError (file might be corrupted or empty). Starting with empty buffer.")
                self.model.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE) # Reinitialize
            except Exception as e:
                logger.error(f"Failed to load or reinitialize replay buffer: {e}", exc_info=True)
                self.model.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE) # Reinitialize as fallback

            # Sync Orchestrator state with loaded model state
            self.last_response_emotions = self.model.prev_emotions.clone().detach()
            self.total_steps = self.model.step_count # Sync total steps
            # Optionally reset mood or load from a separate file if needed
            # self.mood = torch.ones_like(self.model.prev_emotions) * 0.5 # Example reset
            logger.info("Orchestrator state synced with loaded model.")
            return True
        else:
            logger.error("Agent model loading failed. Replay buffer not loaded.")
            return False
    # --- END Save/Load ---

    def cleanup(self):
        """Cleans up resources like the thread pool executor."""
        logger.info("--- Orchestrator Cleanup Initiated ---")
        logger.info("Shutting down learn executor...")
        try:
            # Wait a very short time for any potentially running future
            if self.learn_future and not self.learn_future.done():
                 logger.debug("Waiting briefly for learn future before shutdown...")
                 concurrent.futures.wait([self.learn_future], timeout=0.5)
            # Shutdown executor, wait for tasks, request cancellation
            self.learn_executor.shutdown(wait=True, cancel_futures=True)
            logger.info("Learn executor shut down.")
        except Exception as e:
             logger.error(f"Error shutting down learn executor: {e}", exc_info=True)

        if self.avatar and hasattr(self.avatar, 'cleanup'):
            try:
                logger.info("Cleaning up avatar...")
                self.avatar.cleanup()
                logger.info("Avatar cleanup complete.")
            except Exception as e:
                logger.error(f"Error during avatar cleanup: {e}", exc_info=True)

        # Release Sentence Transformer model if loaded
        if self.st_model:
            try:
                logger.info("Releasing Sentence Transformer model...")
                del self.st_model
                self.st_model = None
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache() # Try to free GPU memory
                    logger.debug("Cleared CUDA cache after releasing ST model.")
            except Exception as e:
                 logger.error(f"Error releasing Sentence Transformer model: {e}")
        logger.info("--- Orchestrator Cleanup Finished ---")

# --- END OF FILE orchestrator.py ---
