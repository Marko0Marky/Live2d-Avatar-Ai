# --- START OF FILE orchestrator.py (Continuing from train_step) ---
import torch
from typing import Dict, Tuple, Optional, List, Union, Deque, Any # Added Any
import concurrent.futures
import asyncio
import time
import math
from collections import deque
import sys
import numpy as np # Import numpy for std calculation

# --- Import SentenceTransformer ---
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence-transformers library not found. Language embedding disabled.")
    print("Install with: pip install sentence-transformers")
# ---

# Use MasterConfig
from config import MasterConfig as Config
from config import DEVICE, logger # Import logger here
from environment import EmotionalSpace
from agent import ConsciousAgent
from graphics import Live2DCharacter
from utils import is_safe, Experience, MetaCognitiveMemory

ReflectReturnType = Dict[str, Union[float, List[float], int, str, np.ndarray]] # Add ndarray for embedding
TrainStepReturnType = Tuple[ReflectReturnType, float, bool, str, float, str]


class EnhancedConsciousAgent:
    def __init__(self, train_interval: int = Config.RL.AGENT_TRAIN_INTERVAL, batch_size: int = Config.RL.AGENT_BATCH_SIZE, num_workers: int = 1):
        logger.info("Initializing EnhancedConsciousAgent Orchestrator (LangEmbed)...")
        self.env = EmotionalSpace()
        # Agent __init__ now uses updated STATE_DIM from config
        self.model = ConsciousAgent().to(DEVICE)
        self.avatar = Live2DCharacter()
        logger.debug("Live2DCharacter display instance created for orchestrator.")

        # --- Load Sentence Transformer Model ---
        self.st_model: Optional[SentenceTransformer] = None
        if Config.Agent.USE_LANGUAGE_EMBEDDING:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    model_name = Config.NLP.SENTENCE_TRANSFORMER_MODEL
                    logger.info(f"Loading sentence transformer model: {model_name}...")
                    # Force model to specified device
                    self.st_model = SentenceTransformer(model_name, device=str(DEVICE))
                    # Quick check of embedding dimension
                    logger.debug("Performing sentence transformer dimension check...")
                    test_emb = self.st_model.encode(["test"], convert_to_tensor=True)[0]
                    actual_dim = test_emb.shape[0]
                    expected_dim = Config.Agent.LANGUAGE_EMBEDDING_DIM
                    if actual_dim != expected_dim:
                         logger.critical(f"FATAL: Sentence transformer model '{model_name}' output dim ({actual_dim}) != Config.Agent.LANGUAGE_EMBEDDING_DIM ({expected_dim}). Update config!")
                         sys.exit(1)
                    logger.info(f"Sentence transformer model loaded successfully to {self.st_model.device}. Output dim: {actual_dim}.")
                except Exception as e:
                    logger.error(f"Failed to load sentence transformer model '{Config.NLP.SENTENCE_TRANSFORMER_MODEL}': {e}. Disabling language embedding.", exc_info=True)
                    self.st_model = None
                    Config.Agent.USE_LANGUAGE_EMBEDDING = False # Disable if load fails
                    # Recalculate STATE_DIM in config if embedding disabled AFTER init
                    Config.Agent.__post_init__(Config.Agent) # Trigger recalculation
                    logger.warning(f"Language embedding disabled. Agent STATE_DIM recalculated to: {Config.Agent.STATE_DIM}")
            else:
                logger.warning("SentenceTransformers library not found. Disabling language embedding.")
                Config.Agent.USE_LANGUAGE_EMBEDDING = False
                Config.Agent.__post_init__(Config.Agent) # Trigger recalculation
                logger.warning(f"Language embedding disabled. Agent STATE_DIM recalculated to: {Config.Agent.STATE_DIM}")

        else:
            logger.info("Language embedding in state is disabled via config.")
        # --- End Load Model ---

        self.train_interval = max(1, train_interval)
        self.batch_size = batch_size

        self.episode_rewards: List[float] = []
        self.current_episode_reward_sum: float = 0.0
        self.total_steps: int = 0; self.episode_steps: int = 0; self.episode_count: int = 0
        self.last_reward: float = 0.0; self.last_reported_loss: float = 0.0
        self.learn_step_running: bool = False

        self.last_response_emotions: torch.Tensor = self.model.prev_emotions.clone().detach()
        self.current_response: str = "Initializing..."
        # Initialize mood to match EMOTION_DIM
        self.mood: torch.Tensor = torch.tensor([0.4, 0.1, 0.5, 0.1, 0.6, 0.2][:Config.Agent.EMOTION_DIM], device=DEVICE, dtype=torch.float32)
        if self.mood.shape[0] != Config.Agent.EMOTION_DIM: self.mood = torch.ones(Config.Agent.EMOTION_DIM, device=DEVICE) * 0.3
        self.last_internal_monologue: str = ""
        self.conversation_history: Deque[Tuple[str, str]] = deque(maxlen=Config.NLP.CONVERSATION_HISTORY_LENGTH * 2)

        # --- Initialize last text embedding ---
        emb_dim = Config.Agent.LANGUAGE_EMBEDDING_DIM if Config.Agent.USE_LANGUAGE_EMBEDDING else 0
        self.last_text_embedding: torch.Tensor = torch.zeros(emb_dim, device=DEVICE, dtype=torch.float32)
        logger.debug(f"Initialized last_text_embedding with shape: {self.last_text_embedding.shape}")


        self.num_workers = max(1, num_workers)
        self.learn_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix='LearnWorker')
        self.learn_future: Optional[concurrent.futures.Future] = None
        self.hud_widget = None # Placeholder for HUD widget if needed

        try:
             # Reset environment to get the *base* state (e.g., size 12)
             self.current_base_state: torch.Tensor = self.env.reset()
             if not is_safe(self.current_base_state) or self.current_base_state.shape[0] != Config.Agent.BASE_STATE_DIM:
                 raise RuntimeError(f"Initial base state is invalid. Shape: {self.current_base_state.shape}, Expected: ({Config.Agent.BASE_STATE_DIM},)")
             logger.debug(f"Initial base state received from env: {self.current_base_state.shape}")
             # Initial combined state includes zero embedding if enabled
             self.current_state = self._get_combined_state(self.current_base_state, "init")
             logger.debug(f"Initial combined state created: {self.current_state.shape}")
             # Final check of combined state dimension
             if self.current_state.shape[0] != Config.Agent.STATE_DIM:
                 raise RuntimeError(f"Initial combined state dimension error! Expected {Config.Agent.STATE_DIM}, got {self.current_state.shape}. Check config logic.")


        except Exception as e: logger.critical(f"Orchestrator Init Error: {e}", exc_info=True); raise RuntimeError("Init state failed.") from e

        self.current_event: Optional[str] = self.env.current_event_type
        logger.info(f"Orchestrator initialized (LangEmbed={'Enabled' if Config.Agent.USE_LANGUAGE_EMBEDDING else 'Disabled'}, StateDim={Config.Agent.STATE_DIM}).")


    # --- Helper to combine base state and embedding ---
    def _get_combined_state(self, base_state: torch.Tensor, context: str = "step") -> torch.Tensor:
        """Combines the base environment state with the last text embedding."""
        if base_state is None or not isinstance(base_state, torch.Tensor):
             logger.error(f"[{context}] _get_combined_state received invalid base_state: {type(base_state)}. Returning zeros.")
             return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)

        base_state_dev = base_state.to(DEVICE)

        if base_state_dev.shape[0] != Config.Agent.BASE_STATE_DIM:
            logger.error(f"[{context}] Base state dim mismatch! Expected {Config.Agent.BASE_STATE_DIM}, got {base_state_dev.shape}. Returning zeros.")
            # Attempt to resize base_state or handle error appropriately
            padded_base = torch.zeros(Config.Agent.BASE_STATE_DIM, device=DEVICE, dtype=base_state_dev.dtype)
            copy_len = min(base_state_dev.shape[0], Config.Agent.BASE_STATE_DIM)
            padded_base[:copy_len] = base_state_dev[:copy_len]
            base_state_dev = padded_base # Use padded base state

        if Config.Agent.USE_LANGUAGE_EMBEDDING:
            # Ensure embedding is on the correct device and has the correct shape
            if self.last_text_embedding is None:
                # logger.warning(f"[{context}] last_text_embedding is None. Using zeros.") # Can be noisy
                current_embedding = torch.zeros(Config.Agent.LANGUAGE_EMBEDDING_DIM, device=DEVICE, dtype=torch.float32)
            else:
                current_embedding = self.last_text_embedding.to(DEVICE)

            if current_embedding.shape[0] != Config.Agent.LANGUAGE_EMBEDDING_DIM:
                 logger.error(f"[{context}] Text embedding dim mismatch! Expected {Config.Agent.LANGUAGE_EMBEDDING_DIM}, got {current_embedding.shape}. Using zeros.")
                 current_embedding = torch.zeros(Config.Agent.LANGUAGE_EMBEDDING_DIM, device=DEVICE, dtype=torch.float32)

            # Concatenate base state and embedding
            try:
                combined = torch.cat((base_state_dev, current_embedding), dim=0)
            except Exception as e:
                logger.error(f"[{context}] Error concatenating base state ({base_state_dev.shape}) and embedding ({current_embedding.shape}): {e}. Returning zeros.")
                return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)

            if combined.shape[0] != Config.Agent.STATE_DIM:
                logger.error(f"[{context}] Combined state dimension error! Expected {Config.Agent.STATE_DIM}, got {combined.shape}. Returning zeros.")
                return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)
            return combined
        else:
            # If not using embedding, ensure base state has the configured STATE_DIM (which should be BASE_STATE_DIM)
             if base_state_dev.shape[0] != Config.Agent.STATE_DIM:
                 logger.error(f"[{context}] Lang embedding disabled, but base state dim {base_state_dev.shape[0]} != Config state dim {Config.Agent.STATE_DIM}. Critical config error!")
                 # Return zeros as a fallback, but this indicates a deeper issue
                 return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=base_state_dev.dtype)
             return base_state_dev # Return base state directly

    def set_hud_widget(self, hud_widget):
        """Stores a reference to the HUD widget for updates (optional)."""
        self.hud_widget = hud_widget
        logger.debug("HUD widget reference stored in orchestrator.")

    def _run_learn_task(self) -> Optional[float]:
        """Wrapper for the agent's learn method to run in a separate thread."""
        loss = None;
        try:
            # Check memory size again just before learning
            if len(self.model.memory) >= self.batch_size:
                loss = self.model.learn(self.batch_size)
            else:
                 loss = 0.0 # Not enough memory, loss is 0
        except Exception as e:
             # Log error from the learning thread
             logger.error(f"Exception in learn task thread: {e}", exc_info=True);
             loss = -1.0 # Indicate error with negative loss
        return loss

    def _check_learn_future(self):
        """Checks if the asynchronous learning task has finished and updates loss."""
        if self.learn_future and self.learn_future.done():
            try:
                loss_result = self.learn_future.result() # Get result from future
                if loss_result is not None and loss_result >= 0:
                     self.last_reported_loss = loss_result
                     # logger.debug(f"Async learn task finished. Loss: {loss_result:.4f}")
                elif loss_result is not None: # Negative loss indicates error
                     logger.warning(f"Async learn task completed with error indicator (Loss: {loss_result}).");
                     self.last_reported_loss = -1.0
                else: # Should not return None, but handle defensively
                     logger.warning("Async learn task returned None.");
                     self.last_reported_loss = -1.0
            except Exception as e:
                 # Handle exceptions raised *during* the learn task itself
                 logger.error(f"Exception retrieving learn task result: {e}");
                 self.last_reported_loss = -1.0
            finally:
                 self.learn_future = None; # Reset future
                 self.learn_step_running = False; # Mark as not running

    # --- train_step continues ---
    def train_step(self) -> TrainStepReturnType:
        """ Performs internal step, generates monologue/embedding, stores experience (gated), triggers learning, updates mood."""
        self._check_learn_future()

        # --- State Validation ---
        if self.current_state is None or not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM:
            logger.error(f"Orchestrator train_step: Invalid current combined state ({getattr(self.current_state, 'shape', 'None')}). Attempting reset.");
            try:
                self.current_base_state = self.env.reset();
                if not is_safe(self.current_base_state) or self.current_base_state.shape[0] != Config.Agent.BASE_STATE_DIM:
                     raise RuntimeError("Reset failed to provide valid base state.")
                self.last_text_embedding.zero_()
                self.current_state = self._get_combined_state(self.current_base_state, "reset_recovery")
                if not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM:
                    raise RuntimeError("Failed to create valid combined state after reset.")
                logger.info("State reset successfully after invalid state detected.")
            except Exception as reset_err:
                 logger.critical(f"CRITICAL RESET FAILED after invalid state: {reset_err}. Stopping.");
                 return ({'error': True, 'message':"FATAL STATE RESET", "last_monologue":"", "embedding_norm": 0.0, "base_state_norm": 0.0},
                        -1.0, True, "FATAL STATE", self.last_reported_loss, "")

        self.total_steps += 1; self.episode_steps += 1
        state_before_step_combined = self.current_state.clone().detach()

        # --- Environment Step ---
        try: next_base_state, reward, env_done, context_internal, event_type = self.env.step()
        except Exception as e:
             logger.error(f"Error during env.step: {e}", exc_info=True);
             next_base_state = self.current_base_state.clone().detach() if self.current_base_state is not None else torch.zeros(Config.Agent.BASE_STATE_DIM, device=DEVICE)
             reward = -0.1; env_done = False; context_internal = "Error: Env Step Failed"; event_type = self.current_event
        if not isinstance(next_base_state, torch.Tensor) or next_base_state.shape[0] != Config.Agent.BASE_STATE_DIM or not is_safe(next_base_state):
            logger.error(f"Env returned invalid next_base_state (shape {getattr(next_base_state, 'shape', 'None')}). Using previous base state.");
            next_base_state = self.current_base_state.clone().detach() if self.current_base_state is not None else torch.zeros(Config.Agent.BASE_STATE_DIM, device=DEVICE)
            reward = -0.1; env_done = False; context_internal = "Error: Invalid Env State"; event_type = self.current_event

        self.last_reward = reward; self.current_event = event_type; self.current_episode_reward_sum += reward

        # --- Agent Step ---
        att_score_metric = 0.0
        response_internal = "(Agent step error)"
        emotions_internal = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
        belief_for_memory = None
        try:
            emotions_internal, response_internal, belief_for_memory, att_score_metric = self.model.step(
                state_before_step_combined, reward, self.model.state_history, context_internal
            )
        except Exception as e:
             logger.error(f"Error during agent.step: {e}", exc_info=True);
             emotions_internal = self.model.prev_emotions if is_safe(self.model.prev_emotions) else torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE);
             belief_for_memory = torch.zeros(Config.Agent.HIDDEN_DIM, device=DEVICE);
             att_score_metric = 0.0

        # --- Generate Internal Monologue & Embedding ---
        self.last_internal_monologue = ""
        current_monologue_embedding = torch.zeros_like(self.last_text_embedding)
        if Config.Agent.USE_LANGUAGE_EMBEDDING and self.st_model:
            try:
                monologue_context = f"Internal state: {context_internal}. Feeling:"
                temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7)
                top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9)
                self.last_internal_monologue = self.model.gpt.generate(monologue_context, emotions_internal, temperature=temp, top_p=top_p, max_len=16)

                if self.last_internal_monologue and self.last_internal_monologue != "...":
                    with torch.no_grad():
                         emb = self.st_model.encode([self.last_internal_monologue], convert_to_tensor=True, device=DEVICE)
                         if emb.ndim > 1: emb = emb.squeeze(0)
                         if emb.shape[0] == Config.Agent.LANGUAGE_EMBEDDING_DIM:
                            current_monologue_embedding = emb.float()
                         else: logger.warning(f"Monologue embedding dim mismatch! Got {emb.shape}, expected ({Config.Agent.LANGUAGE_EMBEDDING_DIM},).")

                self.last_text_embedding = current_monologue_embedding.clone().detach()

            except Exception as e:
                logger.error(f"Error generating/embedding internal monologue: {e}")
                self.last_text_embedding.zero_()
        else:
             self.last_text_embedding.zero_()

        # --- Store Experience (Continuing from previous state) ---
        next_combined_state = self._get_combined_state(next_base_state, "experience_next")

        # Estimate initial TD error for PER
        initial_td_error = MetaCognitiveMemory.INITIAL_TD_ERROR # Default priority
        if is_safe(state_before_step_combined) and is_safe(next_combined_state):
             try:
                 self.model.eval() # Ensure eval mode for priority estimation
                 with torch.no_grad():
                     outputs_s = self.model.forward(state_before_step_combined.unsqueeze(0), torch.tensor([[reward]], device=DEVICE), None) # Batch size 1
                     outputs_sp = self.model.forward(next_combined_state.unsqueeze(0), torch.tensor([[0.0]], device=DEVICE), None) # Batch size 1

                     if len(outputs_s) == 12 and len(outputs_sp) == 12:
                         current_value = outputs_s[3].squeeze() # Get scalar value
                         next_value = outputs_sp[3].squeeze()    # Get scalar value
                         if is_safe(current_value) and is_safe(next_value):
                             target_val = reward + Config.RL.GAMMA * next_value * (0.0 if env_done else 1.0)
                             td_error_raw = target_val - current_value
                             initial_td_error = abs(td_error_raw.item())
                         else: logger.warning("Unsafe V(s) or V(s') in TD error estimation.")
                     else: logger.warning("Forward call length mismatch during TD error estimation.")
                 self.model.train() # Switch back to train mode if needed
             except Exception as e: logger.warning(f"Could not estimate initial TD error: {e}")
        else: logger.warning("Skipping TD error estimation due to unsafe state(s).")


        # Memory Gating Logic
        should_store_long_term = True
        if Config.RL.MEMORY_GATING_ENABLED:
             if att_score_metric < Config.RL.MEMORY_GATE_ATTENTION_THRESHOLD:
                 should_store_long_term = False
                 # logger.debug(f"Memory Gate: Skipped LTM store (Att: {att_score_metric:.3f} < {Config.RL.MEMORY_GATE_ATTENTION_THRESHOLD:.3f})")

        # Add experience to memory if gating allows
        if should_store_long_term:
            try:
                belief_to_store = belief_for_memory if isinstance(belief_for_memory, torch.Tensor) and is_safe(belief_for_memory) else None
                base_priority = initial_td_error + 1e-5 # Add epsilon for non-zero priority
                attention_factor = 1.0 + Config.RL.PRIORITY_ATTENTION_WEIGHT * max(0.0, min(1.0, att_score_metric))
                final_priority = base_priority * attention_factor

                # Store the COMBINED states in the experience tuple
                # Ensure tensors are detached before storing
                exp = Experience(state=state_before_step_combined.detach(),
                                 belief=belief_to_store.detach() if belief_to_store is not None else None,
                                 reward=reward,
                                 next_state=next_combined_state.detach(),
                                 done=env_done,
                                 td_error=final_priority) # Store final priority here
                self.model.memory.add(exp)
            except AttributeError as ae: logger.error(f"AttributeError adding experience (likely config issue): {ae}", exc_info=True)
            except Exception as e: logger.error(f"Failed adding experience to memory: {e}")


        # Trigger Asynchronous Learning (if conditions met)
        if not self.learn_step_running and (self.total_steps % self.train_interval == 0) and (len(self.model.memory) >= self.batch_size):
            try:
                 logger.debug(f"Submitting learn task at step {self.total_steps}. Memory size: {len(self.model.memory)}")
                 self.learn_future = self.learn_executor.submit(self._run_learn_task);
                 self.learn_step_running = True
            except Exception as e:
                 logger.error(f"Failed submit learn task: {e}");
                 self.learn_step_running = False

        # --- State Update for Next Iteration ---
        # Update the base state and create the new combined state
        self.current_base_state = next_base_state.detach().clone()
        # Use the embedding generated *this step* (stored in self.last_text_embedding)
        self.current_state = self._get_combined_state(self.current_base_state, "state_update")

        # Update Mood (Slowly changing average of internal emotions)
        try:
            internal_emotion_this_step = emotions_internal.detach()
            if is_safe(internal_emotion_this_step) and is_safe(self.mood):
                decay = Config.RL.MOOD_UPDATE_DECAY
                self.mood = decay * self.mood + (1.0 - decay) * internal_emotion_this_step
                self.mood = torch.clamp(self.mood, 0.0, 1.0)
            elif not is_safe(self.mood): logger.warning("Mood unsafe. Resetting."); self.mood.fill_(0.3)
            # else: internal emotion was unsafe, don't update mood
        except Exception as e: logger.error(f"Error updating mood: {e}")

        # Episode Handling
        done = env_done
        if done:
            logger.info(f"--- Episode {self.episode_count + 1} ended (Steps: {self.episode_steps}, Reward: {self.current_episode_reward_sum:.2f}) ---")
            self.episode_rewards.append(self.current_episode_reward_sum); self.episode_count += 1
            try:
                # Reset environment and states
                self.current_base_state = self.env.reset();
                if not is_safe(self.current_base_state) or self.current_base_state.shape[0] != Config.Agent.BASE_STATE_DIM:
                     raise RuntimeError("Reset failed to provide valid base state.")
                self.last_text_embedding.zero_(); # Reset embedding
                self.current_state = self._get_combined_state(self.current_base_state, "episode_reset") # Reset combined state
                if not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM:
                    raise RuntimeError("Failed to create valid combined state after episode reset.")

                # Reset agent/orchestrator episode-specific vars
                self.model.prev_emotions.zero_(); self.last_response_emotions.zero_(); self.mood.fill_(0.3);
                self.conversation_history.clear(); self.episode_steps = 0; self.current_episode_reward_sum = 0.0;
                self.current_event = self.env.current_event_type;

            except Exception as e:
                 logger.critical(f"CRITICAL: Failed episode reset: {e}. Stopping simulation.");
                 done = True; # Ensure done is true
                 # Return error structure
                 return ({'error': True, 'message':"FATAL EPISODE RESET", "last_monologue":"", "embedding_norm": 0.0, "base_state_norm": 0.0},
                        reward, done, "FATAL RESET", self.last_reported_loss, "")

        # Reflect current state for reporting
        metrics_dict = self.reflect()
        # Return comprehensive results
        return metrics_dict, reward, done, self.current_response, self.last_reported_loss, self.last_internal_monologue

    # --- MODIFIED: Embed user text, update last_text_embedding ---
    def handle_user_chat(self, user_text: str) -> str:
        """Processes user chat, embeds text, blends emotions, generates response, updates avatar."""
        logger.info(f"Handling user chat: '{user_text[:50]}...'")
        if not isinstance(user_text, str) or not user_text.strip(): return "..."

        # --- Embed User Text for NEXT state ---
        user_text_embedding = torch.zeros_like(self.last_text_embedding) # Default zero
        if Config.Agent.USE_LANGUAGE_EMBEDDING and self.st_model:
            try:
                with torch.no_grad():
                     emb = self.st_model.encode([user_text], convert_to_tensor=True, device=DEVICE)
                     if emb.ndim > 1: emb = emb.squeeze(0)
                     if emb.shape[0] == Config.Agent.LANGUAGE_EMBEDDING_DIM:
                         user_text_embedding = emb.float()
                     else: logger.warning(f"User text embedding dim mismatch! Got {emb.shape}.")
            except Exception as e:
                logger.error(f"Error embedding user text: {e}")
        # --- Update last_text_embedding. This will be used in the *next* combined state. ---
        self.last_text_embedding = user_text_embedding.clone().detach()
        # ---

        # Add user turn to history *after* embedding it for the next state
        self.conversation_history.append(("User", user_text))

        try:
            # 1. Get emotional impact of user text
            impact_vector = self.env.get_emotional_impact_from_text(user_text)

            # 2. Get agent's current emotional state (blend response + mood)
            current_response_emotions = self.last_response_emotions.clone().detach()
            mood_influence = 0.15 # How much the slow-changing mood affects the current reaction
            biased_current_emotions = torch.clamp( current_response_emotions * (1.0 - mood_influence) + self.mood * mood_influence, 0.0, 1.0 )

            # 3. Blend agent's state with user text impact
            current_max_emo = biased_current_emotions.max().item()
            blend_factor_user = max(0.4, min(0.7, 0.7 - (current_max_emo * 0.3))) # Less impact if already emotional
            blend_factor_current = 1.0 - blend_factor_user
            blended_emotions = torch.clamp( biased_current_emotions * blend_factor_current + impact_vector * blend_factor_user, 0.0, 1.0 )

            if not is_safe(blended_emotions):
                 logger.warning("Blended chat emotions unsafe. Using pre-blend state.");
                 blended_emotions = biased_current_emotions

            # 4. Update response emotions and agent's internal previous emotions
            self.last_response_emotions = blended_emotions.clone().detach()
            # Gently nudge agent's internal state towards the response state
            self.model.prev_emotions = torch.clamp(self.model.prev_emotions * 0.5 + self.last_response_emotions * 0.5, 0.0, 1.0)

            # 5. Update Avatar display
            if self.avatar: self.avatar.update_emotions(self.last_response_emotions)

            # 6. Prepare Context with History for GPT
            context_for_gpt = ""
            history_turns = list(self.conversation_history) # Convert deque to list for slicing if needed
            # Iterate in reverse to get recent turns first
            for speaker, text in reversed(history_turns):
                 turn = f"{speaker}: {text}\n"
                 # Simple length limit for context
                 if len(context_for_gpt) + len(turn) > 512: break # Approx limit
                 context_for_gpt = turn + context_for_gpt
            # logger.debug(f"Context for GPT:\n{context_for_gpt.strip()}")

            # 7. Generate Response using GPT
            temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7)
            top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9)
            response = self.model.gpt.generate(context=context_for_gpt, emotions=self.last_response_emotions, temperature=temp, top_p=top_p)
            self.current_response = response

            # 8. Add AI response to history
            self.conversation_history.append(("AI", self.current_response))

            logger.debug(f"Generated response: '{response}' with emotions: {self.last_response_emotions.cpu().numpy().round(2)}")
            return self.current_response

        except Exception as e:
            logger.error(f"Error handling user chat: {e}", exc_info=True)
            self.current_response = "Sorry, I had trouble thinking about that."
            return self.current_response

    # --- MODIFIED: Include embedding norm in reflection ---
    def reflect(self) -> ReflectReturnType:
        """Gathers current statistics and internal states of the agent."""
        stats_dict: ReflectReturnType = {
            "avg_reward_last20": 0.0, "total_steps": self.total_steps, "episode": self.episode_count,
            "current_emotions_internal": [0.0]*Config.Agent.EMOTION_DIM,
            "current_emotions_response": [0.0]*Config.Agent.EMOTION_DIM,
            "current_mood": [0.0]*Config.Agent.EMOTION_DIM,
            "last_monologue": self.last_internal_monologue,
            "I_S": 0.0, "rho_struct": 0.0, "att_score": 0.0, "self_consistency": 0.0,
            "rho_score": 0.0, "box_score": 0.0, "tau_t": 0.0, "R_acc": 0.0,
            "loss": self.last_reported_loss,
            "rho_struct_mem_short": 0.0, "rho_struct_mem_long": 0.0,
            "embedding_norm": 0.0, # NEW
            "base_state_norm": 0.0 # NEW
        }
        # Calculate average reward
        recent_rewards = self.episode_rewards[-20:];
        if recent_rewards: stats_dict["avg_reward_last20"] = sum(recent_rewards) / len(recent_rewards)

        # Get metrics from forward pass on current state
        try:
            # Use the *current combined state* for reflection forward pass
            current_combined_state = self.current_state
            if current_combined_state is not None and is_safe(current_combined_state) and current_combined_state.shape[0] == Config.Agent.STATE_DIM:
                self.model.eval() # Set model to evaluation mode
                with torch.no_grad():
                     # Use current reward=0 for reflection pass, provide history
                     reflect_outputs = self.model.forward(current_combined_state, 0.0, self.model.state_history)
                     if len(reflect_outputs) == 12:
                         (emotions_int, _, _, _value, I_S, rho_struct_fwd, att_score, self_consistency, rho_score, box_score, R_acc_mean, tau_t) = reflect_outputs;
                         # Update stats_dict with forward pass results
                         stats_dict.update({
                             "current_emotions_internal": emotions_int.cpu().tolist() if is_safe(emotions_int) else [0.0]*Config.Agent.EMOTION_DIM,
                             "I_S": I_S, "att_score": att_score, "self_consistency": self_consistency,
                             "rho_score": rho_score, "box_score": box_score, "tau_t": tau_t, "R_acc": R_acc_mean
                         })
                         # Note: rho_struct from forward pass might differ from memory-based one
                     else: logger.warning(f"Reflection: forward returned {len(reflect_outputs)} values, expected 12.")
                self.model.train() # Set back to training mode
            else: logger.warning(f"Reflection skipped: Invalid current_state.")
        except Exception as e: logger.error(f"Error during reflect agent forward pass: {e}", exc_info=False);

        # Get memory norms, response emotions, mood, and embedding norm
        try:
             stats_dict["rho_struct_mem_short"] = self.model.memory.get_short_term_norm();
             stats_dict["rho_struct_mem_long"] = self.model.memory.get_long_term_norm();
             # Calculate overall rho_struct based on memory norms
             stats_dict["rho_struct"] = stats_dict["rho_struct_mem_short"] * 0.3 + stats_dict["rho_struct_mem_long"] * 0.7
             stats_dict["current_emotions_response"] = self.last_response_emotions.cpu().tolist()
             stats_dict["current_mood"] = self.mood.cpu().tolist() if is_safe(self.mood) else [0.0]*Config.Agent.EMOTION_DIM
             # Calculate norm of the last text embedding
             if Config.Agent.USE_LANGUAGE_EMBEDDING and self.last_text_embedding is not None and is_safe(self.last_text_embedding):
                 stats_dict["embedding_norm"] = torch.linalg.norm(self.last_text_embedding.float()).item()
             # Calculate norm of the base state
             if self.current_base_state is not None and is_safe(self.current_base_state):
                 stats_dict["base_state_norm"] = torch.linalg.norm(self.current_base_state.float()).item()

        except Exception as e: logger.error(f"Error getting memory norms/emotions/mood/embedding norm in reflect: {e}", exc_info=False)
        return stats_dict

    # --- MODIFIED: Use combined state for testing ---
    def test_completeness(self) -> Tuple[bool, str]:
        """Tests agent's response consistency based on a modified combined state."""
        logger.info("Performing Completeness Test...")
        # Use the *current combined state* as the base for testing
        current_combined_state = self.current_state
        if current_combined_state is None or not is_safe(current_combined_state) or current_combined_state.shape[0] != Config.Agent.STATE_DIM:
             return False, "Invalid current state for completeness test"

        # Create a modified test state based on the current combined state
        test_state = current_combined_state.clone().detach();
        # Modify the *emotion part* (first EMOTION_DIM elements) of the test state
        if Config.Agent.EMOTION_DIM >= 2:
            test_state[0] = 0.9; # High Joy
            test_state[1] = 0.1; # Low Fear
            # Optionally modify other parts like meta-features or embedding if needed for specific tests
        else:
            logger.warning("Completeness test less effective: EMOTION_DIM < 2.")
            if Config.Agent.EMOTION_DIM == 1: test_state[0] = 0.9 # High Emo 1

        test_reward = 1.5; consistent = False; details = "Prerequisites failed."
        try:
            self.model.eval() # Set to eval mode
            with torch.no_grad():
                 # Pass the modified combined state to forward
                 test_outputs = self.model.forward(test_state, test_reward, self.model.state_history)
                 if len(test_outputs) == 12:
                     (emotions, _, _, _, _, _, att_score, _, rho_score, box_score, R_acc, _) = test_outputs
                     # Check if the expected emotion (Joy) is dominant in the output
                     joy_check = False
                     if Config.Agent.EMOTION_DIM >= 1 and is_safe(emotions):
                         joy_val = emotions[0].item()
                         joy_check = joy_val > 0.5 and joy_val == emotions.max().item()
                     else: joy_val = -1.0

                     # Check consistency metrics against thresholds
                     att_check = att_score > Config.Agent.ATTENTION_THRESHOLD;
                     box_check = box_score > 0 # Simple check if box score is positive
                     # Combine checks for overall consistency
                     consistent = joy_check and att_check and box_check
                     details = (f"Joy={joy_val:.2f}(>{0.5} & max? {joy_check}), Att={att_score:.2f}(>{Config.Agent.ATTENTION_THRESHOLD}? {att_check}), "
                                f"Box={box_score:.2f}(>0? {box_check}), R_acc={R_acc:.2f}, RhoScr={rho_score:.2f}")
                 else: details = f"Forward call returned {len(test_outputs)} items, expected 12."
            self.model.train() # Set back to train mode
        except Exception as e: logger.error(f"Error during completeness test: {e}"); details = f"Exception: {e}"
        logger.info(f"Completeness Test Result: {consistent}. Details: {details}")
        return consistent, details

    def update_environment(self, event_freq: float, intensities: List[float]):
        """Updates environment parameters dynamically."""
        try: self.env.update_params(event_freq, intensities)
        except Exception as e: logger.error(f"Error updating env params: {e}")

    def cleanup(self):
        """Shuts down background processes and cleans up resources."""
        logger.info("--- Orchestrator Cleanup Initiated ---")
        # 1. Shutdown ThreadPoolExecutor for learning tasks
        logger.info("Shutting down learn executor...")
        try:
            # Signal shutdown and wait for running tasks (but don't cancel aggressively)
            self.learn_executor.shutdown(wait=True, cancel_futures=False)
            logger.info("Learn executor shut down successfully.")
        except Exception as e:
            logger.error(f"Error shutting down learn executor: {e}", exc_info=True)

        # 2. Cleanup Avatar (OpenGL resources)
        if self.avatar and hasattr(self.avatar, 'cleanup'):
            try:
                logger.info("Cleaning up avatar widget...")
                self.avatar.cleanup()
                logger.info("Avatar widget cleanup complete.")
            except Exception as e:
                logger.error(f"Error during avatar cleanup: {e}", exc_info=True)

        # 3. Release Sentence Transformer model if loaded
        if self.st_model:
            try:
                logger.info("Releasing Sentence Transformer model...")
                # SentenceTransformer doesn't have an explicit release, rely on GC
                del self.st_model
                self.st_model = None
                if DEVICE.type == 'cuda':
                     torch.cuda.empty_cache() # Try to free CUDA memory
                     logger.debug("Cleared CUDA cache after ST model release.")
            except Exception as e:
                logger.error(f"Error releasing Sentence Transformer model: {e}")

        logger.info("--- Orchestrator Cleanup Finished ---")


# --- END OF FILE orchestrator.py ---
