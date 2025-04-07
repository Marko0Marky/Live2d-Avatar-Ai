# --- START OF FILE orchestrator.py ---
import torch
from typing import Dict, Tuple, Optional, List, Union, Deque
import concurrent.futures
import asyncio
import time
import math
from collections import deque
import sys

# --- NEW: Import SentenceTransformer ---
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
# ---

# Use MasterConfig
from config import MasterConfig as Config
from config import DEVICE, logger # Import logger here
from environment import EmotionalSpace
from agent import ConsciousAgent
from graphics import Live2DCharacter
from utils import is_safe, Experience, MetaCognitiveMemory

ReflectReturnType = Dict[str, Union[float, List[float], int, str]]
TrainStepReturnType = Tuple[ReflectReturnType, float, bool, str, float, str]


class EnhancedConsciousAgent:
    def __init__(self, train_interval: int = Config.RL.AGENT_TRAIN_INTERVAL, batch_size: int = Config.RL.AGENT_BATCH_SIZE, num_workers: int = 1):
        logger.info("Initializing EnhancedConsciousAgent Orchestrator (Async Learn + Chat + Mood + ConvHistory + MemGate + LangEmbed)...")
        self.env = EmotionalSpace()
        self.model = ConsciousAgent().to(DEVICE) # Agent __init__ now uses updated STATE_DIM
        self.avatar = Live2DCharacter()
        logger.debug("Live2DCharacter display instance created for orchestrator.")

        # --- Load Sentence Transformer Model ---
        self.st_model: Optional[SentenceTransformer] = None
        if Config.Agent.USE_LANGUAGE_EMBEDDING:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    model_name = Config.NLP.SENTENCE_TRANSFORMER_MODEL
                    logger.info(f"Loading sentence transformer model: {model_name}...")
                    self.st_model = SentenceTransformer(model_name, device=DEVICE)
                    # Quick check of embedding dimension
                    test_emb = self.st_model.encode(["test"])[0]
                    if test_emb.shape[0] != Config.Agent.LANGUAGE_EMBEDDING_DIM:
                         logger.critical(f"FATAL: Sentence transformer model '{model_name}' output dim ({test_emb.shape[0]}) != Config.Agent.LANGUAGE_EMBEDDING_DIM ({Config.Agent.LANGUAGE_EMBEDDING_DIM}). Update config!")
                         sys.exit(1)
                    logger.info(f"Sentence transformer model loaded successfully ({test_emb.shape[0]} dims).")
                except Exception as e:
                    logger.error(f"Failed to load sentence transformer model '{Config.NLP.SENTENCE_TRANSFORMER_MODEL}': {e}. Disabling language embedding.", exc_info=True)
                    self.st_model = None
                    Config.Agent.USE_LANGUAGE_EMBEDDING = False # Disable if load fails
            else:
                logger.warning("SentenceTransformers library not found. Disabling language embedding.")
                Config.Agent.USE_LANGUAGE_EMBEDDING = False
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
        self.mood: torch.Tensor = torch.tensor([0.4, 0.1, 0.5, 0.1, 0.6, 0.2], device=DEVICE, dtype=torch.float32)
        if self.mood.shape[0] != Config.Agent.EMOTION_DIM: self.mood = torch.ones(Config.Agent.EMOTION_DIM, device=DEVICE) * 0.3
        self.last_internal_monologue: str = ""
        self.conversation_history: Deque[Tuple[str, str]] = deque(maxlen=Config.NLP.CONVERSATION_HISTORY_LENGTH * 2)

        # --- Initialize last text embedding ---
        emb_dim = Config.Agent.LANGUAGE_EMBEDDING_DIM if Config.Agent.USE_LANGUAGE_EMBEDDING else 0
        self.last_text_embedding: torch.Tensor = torch.zeros(emb_dim, device=DEVICE, dtype=torch.float32)

        self.num_workers = max(1, num_workers)
        self.learn_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix='LearnWorker')
        self.learn_future: Optional[concurrent.futures.Future] = None

        try:
             # Reset environment to get the *base* state (size 12)
             self.current_base_state: torch.Tensor = self.env.reset()
             if not is_safe(self.current_base_state) or self.current_base_state.shape[0] != Config.Agent.BASE_STATE_DIM:
                 raise RuntimeError("Initial base state is invalid.")
             # Initial combined state includes zero embedding if enabled
             self.current_state = self._get_combined_state(self.current_base_state)

        except Exception as e: logger.critical(f"Orchestrator Init Error: {e}", exc_info=True); raise RuntimeError("Init state failed.") from e

        self.current_event: Optional[str] = self.env.current_event_type
        logger.info(f"Orchestrator initialized (LangEmbed={'Enabled' if Config.Agent.USE_LANGUAGE_EMBEDDING else 'Disabled'}).")


    # --- NEW: Helper to combine base state and embedding ---
    def _get_combined_state(self, base_state: torch.Tensor) -> torch.Tensor:
        """Combines the base environment state with the last text embedding."""
        if Config.Agent.USE_LANGUAGE_EMBEDDING and self.last_text_embedding is not None:
            if base_state.shape[0] != Config.Agent.BASE_STATE_DIM:
                logger.error(f"Base state dim mismatch! Expected {Config.Agent.BASE_STATE_DIM}, got {base_state.shape}. Returning base state only.")
                # Attempt to resize base_state or handle error appropriately
                # For now, just return base state, agent will likely fail later
                return base_state.to(DEVICE)
            if self.last_text_embedding.shape[0] != Config.Agent.LANGUAGE_EMBEDDING_DIM:
                 logger.error(f"Text embedding dim mismatch! Expected {Config.Agent.LANGUAGE_EMBEDDING_DIM}, got {self.last_text_embedding.shape}. Using zeros.")
                 # Create correct size zero embedding if needed
                 zero_emb = torch.zeros(Config.Agent.LANGUAGE_EMBEDDING_DIM, device=DEVICE, dtype=torch.float32)
                 return torch.cat((base_state.to(DEVICE), zero_emb), dim=0)

            # Concatenate base state and embedding
            combined = torch.cat((base_state.to(DEVICE), self.last_text_embedding.to(DEVICE)), dim=0)
            if combined.shape[0] != Config.Agent.STATE_DIM:
                logger.error(f"Combined state dimension error! Expected {Config.Agent.STATE_DIM}, got {combined.shape}. Returning base state.")
                return base_state.to(DEVICE) # Fallback
            return combined
        else:
            # If not using embedding, ensure state has the configured (original) STATE_DIM
             if base_state.shape[0] != Config.Agent.STATE_DIM:
                 logger.warning(f"Language embedding disabled, but base state dim {base_state.shape[0]} != Config state dim {Config.Agent.STATE_DIM}. Padding/Truncating base state.")
                 padded_state = torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=base_state.dtype)
                 copy_len = min(base_state.shape[0], Config.Agent.STATE_DIM)
                 padded_state[:copy_len] = base_state[:copy_len]
                 return padded_state
             return base_state.to(DEVICE) # Return base state directly


    # ... (set_hud_widget, _run_learn_task, _check_learn_future remain the same) ...
    def set_hud_widget(self, hud_widget): pass

    def _run_learn_task(self) -> Optional[float]:
        loss = None; 
        try: loss = self.model.learn(self.batch_size) if len(self.model.memory) >= self.batch_size else 0.0
        except Exception as e: logger.error(f"Exception in learn task thread: {e}", exc_info=True); loss = -1.0
        return loss

    def _check_learn_future(self):
        if self.learn_future and self.learn_future.done():
            try:
                loss_result = self.learn_future.result()
                if loss_result is not None and loss_result >= 0: self.last_reported_loss = loss_result
                elif loss_result is not None: logger.warning(f"Async learn task error (Loss: {loss_result})."); self.last_reported_loss = -1.0
                else: logger.warning("Async learn task returned None."); self.last_reported_loss = -1.0
            except Exception as e: logger.error(f"Exception retrieving learn task result: {e}"); self.last_reported_loss = -1.0
            finally: self.learn_future = None; self.learn_step_running = False;


    # --- MODIFIED: Use combined state ---
    def train_step(self) -> TrainStepReturnType:
        """ Performs internal step, generates monologue/embedding, stores experience (gated), triggers learning, updates mood."""
        self._check_learn_future()

        # We use self.current_state (potentially combined) for the agent step
        # but need self.current_base_state for the environment step result check
        if self.current_state is None or not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM:
            logger.error(f"Orchestrator train_step: Invalid combined state ({self.current_state.shape if hasattr(self.current_state, 'shape') else 'None'}). Resetting.");
            try:
                self.current_base_state = self.env.reset(); assert is_safe(self.current_base_state) and self.current_base_state.shape[0] == Config.Agent.BASE_STATE_DIM
                self.last_text_embedding.zero_() # Reset embedding
                self.current_state = self._get_combined_state(self.current_base_state) # Create new combined state
                assert is_safe(self.current_state) and self.current_state.shape[0] == Config.Agent.STATE_DIM
            except Exception as reset_err: logger.critical(f"Reset failed: {reset_err}"); return ({'error': True, 'message':"FATAL RESET", "last_monologue":""}, -1.0, True, "FATAL STATE", self.last_reported_loss, "")

        self.total_steps += 1; self.episode_steps += 1
        # --- Pass the potentially combined state to the agent ---
        state_before_step_combined = self.current_state.clone().detach()
        base_state_before_step = self.current_base_state.clone().detach() # Keep track of base state before env step

        # Env Step - only provides the base state update
        try: next_base_state, reward, env_done, context_internal, event_type = self.env.step()
        except Exception as e: logger.error(f"Error env step: {e}"); next_base_state = base_state_before_step; reward = -0.1; env_done = False; context_internal = "Error env step."; event_type = self.current_event
        if not isinstance(next_base_state, torch.Tensor) or next_base_state.shape[0] != Config.Agent.BASE_STATE_DIM or not is_safe(next_base_state):
            logger.error("Env invalid next_base_state."); next_base_state = base_state_before_step; reward = -0.1; env_done = False; context_internal = "Env State Error"; event_type = self.current_event

        self.last_reward = reward; self.current_event = event_type; self.current_episode_reward_sum += reward

        # Agent Step - uses the *combined* state from the *previous* timestep
        att_score_metric = 0.0
        response_internal = "(Agent step error)"
        emotions_internal = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
        try:
            # Agent processes the combined state
            emotions_internal, response_internal, belief_for_memory, att_score_metric = self.model.step(state_before_step_combined, reward, self.model.state_history, context_internal)
        except Exception as e:
             logger.error(f"Error agent step: {e}");
             emotions_internal = self.model.prev_emotions if is_safe(self.model.prev_emotions) else torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE);
             belief_for_memory = torch.zeros(Config.Agent.HIDDEN_DIM, device=DEVICE); att_score_metric = 0.0

        # Generate Internal Monologue & Embedding
        self.last_internal_monologue = ""
        current_monologue_embedding = torch.zeros_like(self.last_text_embedding) # Default zero embedding
        try:
            monologue_context = f"Internal state: {context_internal}. Feeling:"
            temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7)
            top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9)
            self.last_internal_monologue = self.model.gpt.generate(monologue_context, emotions_internal, temperature=temp, top_p=top_p, max_len=12)
            if self.last_internal_monologue and self.last_internal_monologue != "...":
                logger.debug(f"Internal Monologue: '{self.last_internal_monologue}' (based on internal emo: {emotions_internal.cpu().numpy().round(2)})")
                # --- Embed the monologue ---
                if Config.Agent.USE_LANGUAGE_EMBEDDING and self.st_model:
                    with torch.no_grad():
                         emb = self.st_model.encode([self.last_internal_monologue], convert_to_tensor=True, device=DEVICE)
                         if emb.ndim > 1: emb = emb.squeeze(0) # Ensure 1D tensor
                         current_monologue_embedding = emb.float() # Store this step's embedding

            # --- Update last_text_embedding for the *next* state ---
            self.last_text_embedding = current_monologue_embedding.clone().detach()

        except Exception as e:
            logger.error(f"Error generating/embedding internal monologue: {e}")
            self.last_text_embedding.zero_() # Reset on error


        # --- Store Experience (using COMBINED states) ---
        initial_td_error = MetaCognitiveMemory.INITIAL_TD_ERROR
        with torch.no_grad():
             try:
                 # Use combined states for value estimation
                 next_combined_state = self._get_combined_state(next_base_state) # Create next combined state
                 outputs_s = self.model.forward(state_before_step_combined, reward, self.model.state_history)
                 outputs_sp = self.model.forward(next_combined_state, 0.0, None)
                 if len(outputs_s) == 12 and len(outputs_sp) == 12:
                     current_value = outputs_s[3]; next_value = outputs_sp[3]
                     if is_safe(current_value) and is_safe(next_value):
                         target_val = reward + Config.RL.GAMMA * next_value * (0.0 if env_done else 1.0)
                         initial_td_error = abs((target_val - current_value).item())
                 else: logger.warning("Forward mismatch during TD error estimation.")
             except Exception as e: logger.warning(f"Could not estimate initial TD error: {e}")

        should_store_long_term = True
        if Config.RL.MEMORY_GATING_ENABLED:
             if att_score_metric < Config.RL.MEMORY_GATE_ATTENTION_THRESHOLD:
                 should_store_long_term = False

        if should_store_long_term:
            try:
                belief_to_store = belief_for_memory if isinstance(belief_for_memory, torch.Tensor) and is_safe(belief_for_memory) else None
                base_priority = abs(initial_td_error) + 1e-5
                attention_factor = 1.0 + Config.RL.PRIORITY_ATTENTION_WEIGHT * max(0.0, min(1.0, att_score_metric))
                final_priority = base_priority * attention_factor
                # Store the COMBINED states in the experience tuple
                exp = Experience(state_before_step_combined.detach(), # Store combined state
                                 belief_to_store, reward,
                                 self._get_combined_state(next_base_state).detach(), # Store next combined state
                                 env_done, final_priority)
                self.model.memory.add(exp)
            except AttributeError as ae: logger.error(f"AttributeError adding experience (likely config issue): {ae}", exc_info=True)
            except Exception as e: logger.error(f"Failed adding experience: {e}")

        # Trigger Async Learning
        if not self.learn_step_running and (self.total_steps % self.train_interval == 0) and (len(self.model.memory) >= self.batch_size):
            try: self.learn_future = self.learn_executor.submit(self._run_learn_task); self.learn_step_running = True
            except Exception as e: logger.error(f"Failed submit learn task: {e}"); self.learn_step_running = False

        # --- State Update ---
        # Update the base state and create the new combined state for the next iteration
        self.current_base_state = next_base_state.detach().clone()
        self.current_state = self._get_combined_state(self.current_base_state) # self.last_text_embedding was updated above

        # Update Mood
        try:
            internal_emotion_this_step = emotions_internal.detach()
            if is_safe(internal_emotion_this_step) and is_safe(self.mood):
                decay = Config.RL.MOOD_UPDATE_DECAY
                self.mood = decay * self.mood + (1.0 - decay) * internal_emotion_this_step
                self.mood = torch.clamp(self.mood, 0.0, 1.0)
            elif not is_safe(self.mood): logger.warning("Mood unsafe. Resetting."); self.mood.fill_(0.3)
        except Exception as e: logger.error(f"Error updating mood: {e}")

        # Episode Handling
        done = env_done
        if done:
            logger.info(f"Episode {self.episode_count + 1} finished...")
            self.episode_rewards.append(self.current_episode_reward_sum); self.episode_count += 1
            try:
                self.current_base_state = self.env.reset(); assert is_safe(self.current_base_state) and self.current_base_state.shape[0] == Config.Agent.BASE_STATE_DIM
                self.last_text_embedding.zero_(); self.current_state = self._get_combined_state(self.current_base_state) # Reset combined state
            except Exception as e: logger.critical(f"CRITICAL: Failed reset: {e}."); done = True; self.current_state = None; response = "FATAL RESET"; return ({'error': True, 'message':"FATAL RESET", "last_monologue":""}, reward, done, response, self.last_reported_loss, "")
            self.model.prev_emotions.zero_(); self.last_response_emotions.zero_(); self.mood.fill_(0.3); self.conversation_history.clear(); self.episode_steps = 0; self.current_episode_reward_sum = 0.0; self.current_event = self.env.current_event_type;

        metrics_dict = self.reflect()
        return metrics_dict, reward, done, self.current_response, self.last_reported_loss, self.last_internal_monologue

    # --- MODIFIED: Embed user text, update last_text_embedding ---
    def handle_user_chat(self, user_text: str) -> str:
        """Processes user chat, embeds text, blends emotions, generates response, updates avatar."""
        logger.info(f"Handling user chat: '{user_text[:50]}...'")
        if not isinstance(user_text, str) or not user_text.strip(): return "..."

        # --- Embed User Text ---
        user_text_embedding = torch.zeros_like(self.last_text_embedding) # Default zero
        if Config.Agent.USE_LANGUAGE_EMBEDDING and self.st_model:
            try:
                with torch.no_grad():
                     emb = self.st_model.encode([user_text], convert_to_tensor=True, device=DEVICE)
                     if emb.ndim > 1: emb = emb.squeeze(0)
                     user_text_embedding = emb.float()
            except Exception as e:
                logger.error(f"Error embedding user text: {e}")
        # --- Update last_text_embedding for the *next* state ---
        self.last_text_embedding = user_text_embedding.clone().detach()
        # --- End Embedding ---


        # Add user turn to history *after* embedding it for the next state
        self.conversation_history.append(("User", user_text))

        try:
            impact_vector = self.env.get_emotional_impact_from_text(user_text)
            current_response_emotions = self.last_response_emotions.clone().detach()

            mood_influence = 0.15
            biased_current_emotions = torch.clamp( current_response_emotions * (1.0 - mood_influence) + self.mood * mood_influence, 0.0, 1.0 )

            current_max_emo = biased_current_emotions.max().item()
            blend_factor_user = max(0.4, min(0.7, 0.7 - (current_max_emo * 0.3)))
            blend_factor_current = 1.0 - blend_factor_user

            blended_emotions = torch.clamp( biased_current_emotions * blend_factor_current + impact_vector * blend_factor_user, 0.0, 1.0 )
            if not is_safe(blended_emotions): logger.warning("Blended chat emotions unsafe."); blended_emotions = biased_current_emotions

            self.last_response_emotions = blended_emotions.clone().detach()
            self.model.prev_emotions = torch.clamp(self.model.prev_emotions * 0.5 + self.last_response_emotions * 0.5, 0.0, 1.0)

            if self.avatar: self.avatar.update_emotions(self.last_response_emotions)

            # --- Prepare Context with History ---
            context_for_gpt = ""
            for speaker, text in reversed(self.conversation_history):
                 turn = f"{speaker}: {text}\n"
                 if len(context_for_gpt) + len(turn) > Config.NLP.MAX_RESPONSE_LEN * 4: break # Simple length limit
                 context_for_gpt = turn + context_for_gpt
            logger.debug(f"Context for GPT:\n{context_for_gpt}")
            # --- End Context Prep ---

            temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7)
            top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9)
            response = self.model.gpt.generate(context=context_for_gpt, emotions=self.last_response_emotions, temperature=temp, top_p=top_p)
            self.current_response = response

            self.conversation_history.append(("AI", self.current_response))

            logger.debug(f"Generated response: '{response}' with emotions: {self.last_response_emotions.cpu().numpy().round(2)}")
            return self.current_response

        except Exception as e:
            logger.error(f"Error handling user chat: {e}", exc_info=True)
            self.current_response = "Sorry, I had trouble thinking about that."
            return self.current_response

    # ... (reflect, test_completeness, update_environment, cleanup remain the same) ...
    def reflect(self) -> ReflectReturnType:
        # ... (reflect logic remains the same) ...
        stats_dict: ReflectReturnType = { "avg_reward_last20": 0.0, "total_steps": self.total_steps, "episode": self.episode_count, "current_emotions_internal": [0.0]*Config.Agent.EMOTION_DIM, "current_emotions_response": [0.0]*Config.Agent.EMOTION_DIM, "current_mood": [0.0]*Config.Agent.EMOTION_DIM, "last_monologue": self.last_internal_monologue, "I_S": 0.0, "rho_struct": 0.0, "att_score": 0.0, "self_consistency": 0.0, "rho_score": 0.0, "box_score": 0.0, "tau_t": 0.0, "R_acc": 0.0, "loss": self.last_reported_loss, "rho_struct_mem_short": 0.0, "rho_struct_mem_long": 0.0 }
        recent_rewards = self.episode_rewards[-20:];
        if recent_rewards: stats_dict["avg_reward_last20"] = sum(recent_rewards) / len(recent_rewards)
        try:
            # Use the *current combined state* for reflection forward pass
            current_combined_state = self.current_state
            if current_combined_state is not None and is_safe(current_combined_state) and current_combined_state.shape[0] == Config.Agent.STATE_DIM:
                self.model.eval()
                with torch.no_grad():
                     reflect_outputs = self.model.forward(current_combined_state, self.last_reward, self.model.state_history)
                     if len(reflect_outputs) == 12:
                         (emotions_int, _, _, _value, I_S, _, att_score, self_consistency, rho_score, box_score, R_acc_mean, tau_t) = reflect_outputs;
                         stats_dict.update({ "current_emotions_internal": emotions_int.cpu().tolist() if is_safe(emotions_int) else [0.0]*Config.Agent.EMOTION_DIM, "I_S": I_S, "att_score": att_score, "self_consistency": self_consistency, "rho_score": rho_score, "box_score": box_score, "tau_t": tau_t, "R_acc": R_acc_mean })
                     else: logger.warning(f"Reflection: forward returned {len(reflect_outputs)} values, expected 12.")
            else: logger.warning(f"Reflection skipped: Invalid current_state.")
        except Exception as e: logger.error(f"Error reflect agent state: {e}", exc_info=False);
        try:
             stats_dict["rho_struct_mem_short"] = self.model.memory.get_short_term_norm();
             stats_dict["rho_struct_mem_long"] = self.model.memory.get_long_term_norm();
             stats_dict["rho_struct"] = stats_dict["rho_struct_mem_short"] * 0.3 + stats_dict["rho_struct_mem_long"] * 0.7
             stats_dict["current_emotions_response"] = self.last_response_emotions.cpu().tolist()
             stats_dict["current_mood"] = self.mood.cpu().tolist() if is_safe(self.mood) else [0.0]*Config.Agent.EMOTION_DIM
        except Exception as e: logger.error(f"Error get memory norms/response emotions/mood reflect: {e}", exc_info=False)
        return stats_dict

    def test_completeness(self) -> Tuple[bool, str]:
        logger.info("Performing Completeness Test...")
        # Use the *current combined state* for testing
        current_combined_state = self.current_state
        if current_combined_state is None or not is_safe(current_combined_state) or current_combined_state.shape[0] != Config.Agent.STATE_DIM: return False, "Invalid state for test"

        # Create a test state based on the combined state
        test_state = current_combined_state.clone().detach();
        # Modify the *emotion part* of the test state
        if Config.Agent.EMOTION_DIM >= 2:
            test_state[0] = 0.9; # High Joy
            test_state[1] = 0.1; # Low Fear

        test_reward = 1.5; consistent = False; details = "Prerequisites failed."
        try:
            self.model.eval()
            with torch.no_grad():
                 # Pass the modified combined state to forward
                 test_outputs = self.model.forward(test_state, test_reward, self.model.state_history)
                 if len(test_outputs) == 12:
                     (emotions, _, _, _, _, _, att_score, _, rho_score, box_score, R_acc, _) = test_outputs
                     joy_check = emotions[0].item() > 0.5 if Config.Agent.EMOTION_DIM >= 1 and is_safe(emotions) else False
                     att_check = att_score > Config.Agent.ATTENTION_THRESHOLD;
                     box_check = box_score > 0
                     consistent = joy_check and att_check and box_check
                     joy_val = emotions[0].item() if Config.Agent.EMOTION_DIM >=1 and is_safe(emotions) else -1.0
                     details = (f"Joy={joy_val:.2f}(>{0.5}? {joy_check}), Att={att_score:.2f}(>{Config.Agent.ATTENTION_THRESHOLD}? {att_check}), "
                                f"Box={box_score:.2f}(>0? {box_check}), R_acc={R_acc:.2f}, RhoScr={rho_score:.2f}")
                 else: details = f"Forward mismatch."
        except Exception as e: logger.error(f"Error completeness test: {e}"); details = f"Exception: {e}"
        logger.info(f"Completeness Test Result: {consistent}. Details: {details}")
        return consistent, details

    def update_environment(self, event_freq: float, intensities: List[float]):
        try: self.env.update_params(event_freq, intensities)
        except Exception as e: logger.error(f"Error updating env params: {e}")

    def cleanup(self):
        logger.info("Shutting down learn executor...")
        try:
            self.learn_executor.shutdown(wait=True, cancel_futures=False)
            logger.info("Learn executor shut down successfully.")
        except Exception as e:
            logger.error(f"Error shutting down learn executor: {e}", exc_info=True)


# --- END OF FILE orchestrator.py ---
