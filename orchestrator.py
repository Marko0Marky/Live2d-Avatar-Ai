# --- START OF FILE orchestrator.py ---
import torch
from typing import Dict, Tuple, Optional
import concurrent.futures
import asyncio
import time

from config import Config, DEVICE, logger
from environment import EmotionalSpace
from agent import ConsciousAgent
from graphics import Live2DCharacter
from utils import is_safe, Experience, MetaCognitiveMemory

class EnhancedConsciousAgent: # Orchestrator Class
    """Orchestrates interaction, triggers batched learning asynchronously, and handles chat."""
    def __init__(self, train_interval=Config.AGENT_TRAIN_INTERVAL, batch_size=Config.AGENT_BATCH_SIZE, num_workers=1): # Use Config values
        logger.info("Initializing EnhancedConsciousAgent Orchestrator (Async Batched Learning + Chat)...")
        self.env = EmotionalSpace()
        self.model = ConsciousAgent(state_dim=Config.STATE_DIM,
                                    hidden_dim=Config.HIDDEN_DIM,
                                    vocab_size=Config.VOCAB_SIZE).to(DEVICE)
        self.avatar = Live2DCharacter() # GUI takes ownership
        logger.debug("Live2DCharacter display instance created for orchestrator.")

        # Learning parameters
        self.train_interval = max(1, train_interval)
        self.batch_size = batch_size

        # State Tracking
        self.episode_rewards = []
        self.current_episode_reward_sum = 0.0
        self.total_steps = 0; self.episode_steps = 0; self.episode_count = 0
        self.last_reward = 0.0; self.last_reported_loss = 0.0
        self.learn_step_running = False

        # --- NEW: Separate emotion state for chat/avatar ---
        self.last_response_emotions = self.model.prev_emotions.clone().detach()
        self.current_response = "Initializing..." # Response shown in GUI

        # --- Thread Pool for Asynchronous Learning ---
        self.num_workers = max(1, num_workers)
        self.learn_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix='LearnWorker')
        self.learn_future: Optional[concurrent.futures.Future] = None

        try: # Initial state reset
             self.current_state = self.env.reset(); assert is_safe(self.current_state) and self.current_state.shape == (Config.STATE_DIM,)
        except Exception as e: logger.critical(f"Orchestrator Init Error: {e}", exc_info=True); raise RuntimeError("Init state failed.") from e

        self.current_event = self.env.current_event_type
        logger.info(f"Orchestrator initialized (Async Learn, {self.num_workers} worker(s), Chat Enabled).")

    def set_hud_widget(self, hud_widget):
        # Pass HUD widget ref to avatar if needed, though HUD updates itself now
        pass # No direct need currently

    def _run_learn_task(self) -> Optional[float]:
        # ... (_run_learn_task remains the same) ...
        """The function executed by the thread pool to run agent learning."""
        loss = None # Use None to indicate not run/error vs 0.0 loss
        try:
            if len(self.model.memory) >= self.batch_size:
                 loss = self.model.learn(self.batch_size) # This returns float loss
            else:
                 loss = 0.0 # Indicate skipped but no error
        except Exception as e:
            logger.error(f"Exception in learn task thread: {e}", exc_info=True)
            loss = -1.0 # Indicate error state with negative loss
        finally:
             pass # Flag reset happens in _check_learn_future
        return loss # Return loss or None/negative for error

    def _check_learn_future(self):
        # ... (_check_learn_future remains the same) ...
        """Checks if the asynchronous learn task is done and updates the loss."""
        if self.learn_future and self.learn_future.done():
            try:
                loss_result = self.learn_future.result() # Get result
                if loss_result is not None and loss_result >= 0: self.last_reported_loss = loss_result
                elif loss_result is not None: logger.warning(f"Async learn task error (Loss: {loss_result})."); self.last_reported_loss = -1.0
                else: logger.warning("Async learn task returned None."); self.last_reported_loss = -1.0
            except Exception as e: logger.error(f"Exception retrieving learn task result: {e}"); self.last_reported_loss = -1.0
            finally: self.learn_future = None; self.learn_step_running = False; # Clear future and reset flag

    def train_step(self) -> Tuple[Dict, float, bool, str, float]:
        """ Performs internal simulation step, stores experience, triggers async learning."""
        self._check_learn_future() # Check previous learning task

        if self.current_state is None or not is_safe(self.current_state) or self.current_state.shape != (Config.STATE_DIM,):
            logger.error(f"Orchestrator train_step: Invalid state. Resetting.");
            try: self.current_state = self.env.reset(); assert is_safe(self.current_state) and self.current_state.shape==(Config.STATE_DIM,)
            except: logger.critical("Reset failed."); return ({'error': True, 'message':"FATAL STATE"}, -1.0, True, "FATAL STATE", self.last_reported_loss)

        self.total_steps += 1; self.episode_steps += 1
        state_before_step = self.current_state.clone().detach()

        # Env Step (Internal events)
        try: next_state, reward, env_done, context_internal, event_type = self.env.step()
        except Exception as e: logger.error(f"Error env step: {e}"); next_state = state_before_step; reward = -0.1; env_done = False; context_internal = "Error env step."; event_type = self.current_event
        if not isinstance(next_state, torch.Tensor) or next_state.shape != (Config.STATE_DIM,) or not is_safe(next_state):
            logger.error("Env invalid next_state."); next_state = state_before_step; reward = -0.1; env_done = False; context_internal = "Env State Error"; event_type = self.current_event

        self.last_reward = reward; self.current_event = event_type; self.current_episode_reward_sum += reward

        # Agent Step (Internal update based on internal context)
        try:
            # Use agent's internal step method which includes learning logic trigger
            emotions_internal, response_internal, belief_for_memory, att_score_metric = self.model.step(state_before_step, reward, self.model.state_history, context_internal)
            # Note: model.step updates model.prev_emotions internally
        except Exception as e:
             logger.error(f"Error agent step: {e}");
             emotions_internal = self.model.prev_emotions if is_safe(self.model.prev_emotions) else torch.zeros(Config.EMOTION_DIM, device=DEVICE);
             response_internal = "(Agent step error)"; belief_for_memory = torch.zeros(Config.HIDDEN_DIM, device=DEVICE); att_score_metric = 0.0

        # --- Store Experience (using internal step results) ---
        initial_td_error = MetaCognitiveMemory.INITIAL_TD_ERROR
        with torch.no_grad():
             try:
                 # Use agent's forward pass to estimate values for priority
                 outputs_s = self.model.forward(state_before_step, reward, self.model.state_history)
                 outputs_sp = self.model.forward(next_state, 0.0, None)
                 if len(outputs_s) == 12 and len(outputs_sp) == 12:
                     current_value = outputs_s[3] # Value is 4th element
                     next_value = outputs_sp[3]
                     if is_safe(current_value) and is_safe(next_value):
                         target_val = reward + Config.GAMMA * next_value * (0.0 if env_done else 1.0)
                         initial_td_error = abs((target_val - current_value).item())
                 else: logger.warning("Forward mismatch during TD error estimation.")
             except Exception as e: logger.warning(f"Could not estimate initial TD error: {e}")
        try:
            belief_to_store = belief_for_memory if isinstance(belief_for_memory, torch.Tensor) and is_safe(belief_for_memory) else None
            exp = Experience(state_before_step, belief_to_store, reward, next_state.detach(), env_done, initial_td_error)
            self.model.memory.add(exp)
        except Exception as e: logger.error(f"Failed adding experience: {e}")

        # --- Trigger Async Learning ---
        if not self.learn_step_running and (self.total_steps % self.train_interval == 0) and (len(self.model.memory) >= self.batch_size):
            try: self.learn_future = self.learn_executor.submit(self._run_learn_task); self.learn_step_running = True
            except Exception as e: logger.error(f"Failed submit learn task: {e}"); self.learn_step_running = False

        # --- State Update ---
        # DO NOT update avatar here based on internal step. Chat handles avatar.
        self.current_state = next_state.detach().clone() # Update internal state

        # Episode Handling
        done = env_done
        if done:
            logger.info(f"Episode {self.episode_count + 1} finished. Steps: {self.episode_steps}. Reward: {self.current_episode_reward_sum:.2f}. Last Learn Loss: {self.last_reported_loss:.4f}")
            self.episode_rewards.append(self.current_episode_reward_sum); self.episode_count += 1
            try: self.current_state = self.env.reset(); assert is_safe(self.current_state) and self.current_state.shape == (Config.STATE_DIM,)
            except Exception as e: logger.critical(f"CRITICAL: Failed reset: {e}."); done = True; self.current_state = None; response = "FATAL RESET"; return ({'error': True, 'message':"FATAL RESET"}, reward, done, response, self.last_reported_loss)
            self.model.prev_emotions.zero_() # Reset internal agent emotion state
            self.last_response_emotions.zero_() # Reset response emotion state
            self.episode_steps = 0; self.current_episode_reward_sum = 0.0; self.current_event = self.env.current_event_type;

        # Return Results (Metrics from internal state, response/emotion from chat state)
        metrics_dict = self.reflect() # Gets latest metrics + chat emotions
        # Return tuple: metrics_dict, reward, done, LATEST response, last_loss
        return metrics_dict, reward, done, self.current_response, self.last_reported_loss

    # --- NEW: Chat handling method ---
    def handle_user_chat(self, user_text: str) -> str:
        """Processes user chat, blends emotions, generates response, updates avatar."""
        logger.info(f"Handling user chat: '{user_text[:50]}...'")
        if not isinstance(user_text, str) or not user_text.strip():
            return "..." # Handle empty input

        try:
            # 1. Get emotional impact vector from text
            impact_vector = self.env.get_emotional_impact_from_text(user_text)

            # 2. Get current chat/response emotions
            current_response_emotions = self.last_response_emotions.clone().detach()

            # 3. Blend emotions (give user text decent weight)
            blend_factor_user = 0.65
            blend_factor_current = 1.0 - blend_factor_user
            blended_emotions = torch.clamp(
                current_response_emotions * blend_factor_current + impact_vector * blend_factor_user,
                0.0, 1.0
            )
            if not is_safe(blended_emotions):
                logger.warning("Blended emotions unsafe, using previous."); blended_emotions = current_response_emotions

            # 4. Update the persistent response emotion state
            self.last_response_emotions = blended_emotions.clone().detach()

            # 5. Slightly nudge the agent's internal previous emotion state
            self.model.prev_emotions = self.model.prev_emotions * 0.5 + self.last_response_emotions * 0.5
            self.model.prev_emotions = torch.clamp(self.model.prev_emotions, 0.0, 1.0)

            # 6. Update avatar immediately
            if self.avatar:
                self.avatar.update_emotions(self.last_response_emotions)

            # 7. Generate response using GPT with blended emotions
            # Pass user_text as context
            response = self.model.gpt.generate(context=user_text, emotions=self.last_response_emotions)
            self.current_response = response # Update the response to be shown by GUI

            logger.debug(f"Generated response: '{response}' with emotions: {self.last_response_emotions.cpu().numpy()}")
            return self.current_response

        except Exception as e:
            logger.error(f"Error handling user chat: {e}", exc_info=True)
            self.current_response = "Sorry, I had trouble thinking about that."
            return self.current_response

    def reflect(self) -> Dict:
        # ... (reflect method adjusted to include both emotion sets) ...
        stats_dict = {
            "avg_reward_last20": 0.0, "total_steps": self.total_steps, "episode": self.episode_count,
            "current_emotions_internal": [0.0]*Config.EMOTION_DIM, # Agent's internal state
            "current_emotions_response": [0.0]*Config.EMOTION_DIM, # State used for avatar/chat
            "I_S": 0.0, "rho_struct": 0.0, "att_score": 0.0, "self_consistency": 0.0,
            "rho_score": 0.0, "box_score": 0.0, "tau_t": 0.0, "R_acc": 0.0,
            "loss": self.last_reported_loss, "rho_struct_mem_short": 0.0, "rho_struct_mem_long": 0.0
            }
        recent_rewards = self.episode_rewards[-20:];
        if recent_rewards: stats_dict["avg_reward_last20"] = sum(recent_rewards) / len(recent_rewards)
        try:
            if self.current_state is not None and is_safe(self.current_state) and self.current_state.shape == (Config.STATE_DIM,):
                self.model.eval()
                with torch.no_grad():
                     reflect_outputs = self.model.forward(self.current_state, self.last_reward, self.model.state_history)
                if len(reflect_outputs) == 12:
                    (emotions_int, _, _, _value, I_S, _, att_score, self_consistency, rho_score, box_score, R_acc_mean, tau_t) = reflect_outputs;
                    stats_dict.update({
                         "current_emotions_internal": emotions_int.cpu().tolist() if is_safe(emotions_int) else [0.0]*Config.EMOTION_DIM,
                         "I_S": I_S, "att_score": att_score, "self_consistency": self_consistency,
                         "rho_score": rho_score, "box_score": box_score, "tau_t": tau_t, "R_acc": R_acc_mean
                         })
                else: logger.warning(f"Reflection: forward mismatch.")
            else: logger.warning(f"Reflection skipped: Invalid state.")
        except Exception as e: logger.error(f"Error reflect: {e}", exc_info=False);
        try:
             stats_dict["rho_struct_mem_short"] = self.model.memory.get_short_term_norm();
             stats_dict["rho_struct_mem_long"] = self.model.memory.get_long_term_norm();
             stats_dict["rho_struct"] = stats_dict["rho_struct_mem_short"] * 0.3 + stats_dict["rho_struct_mem_long"] * 0.7
             # Add the current response/avatar emotion state
             stats_dict["current_emotions_response"] = self.last_response_emotions.cpu().tolist()
        except Exception as e: logger.error(f"Error memory norms/response emotions reflect: {e}", exc_info=False)
        return stats_dict


    def test_completeness(self) -> Tuple[bool, str]:
        # ... (test_completeness remains the same) ...
        logger.info("Performing Completeness Test...")
        if self.current_state is None or not is_safe(self.current_state) or self.current_state.shape != (Config.STATE_DIM,): return False, "Invalid state"
        test_state = self.current_state.clone().detach();
        if Config.EMOTION_DIM >= 2: test_state[0] = 0.9; test_state[1] = 0.1;
        test_reward = 1.5; consistent = False; details = "Prerequisites failed."
        try:
            self.model.eval()
            with torch.no_grad(): test_outputs = self.model.forward(test_state, test_reward, self.model.state_history)
            if len(test_outputs) == 12:
                 (emotions, _, _, _, _, _, att_score, _, _, box_score, R_acc, _) = test_outputs;
                 joy_check = emotions[0].item() > 0.5 if Config.EMOTION_DIM >= 1 and is_safe(emotions) else False
                 att_check = att_score > Config.ATTENTION_THRESHOLD; box_check = box_score > 0
                 consistent = joy_check and att_check and box_check
                 joy_val = emotions[0].item() if Config.EMOTION_DIM >=1 and is_safe(emotions) else -1.0
                 details = (f"Joy={joy_val:.2f}(>{0.5}? {joy_check}), Att={att_score:.2f}(>{Config.ATTENTION_THRESHOLD}? {att_check}), Box={box_score:.2f}(>0? {box_check}), R_acc={R_acc:.2f}")
            else: details = f"Forward mismatch."
        except Exception as e: logger.error(f"Error completeness test: {e}"); details = f"Exception: {e}"
        logger.info(f"Completeness Test Result: {consistent}. Details: {details}")
        return consistent, details

    def update_environment(self, event_freq, intensities):
        # ... (remains the same) ...
        try: self.env.update_params(event_freq, intensities)
        except Exception as e: logger.error(f"Error updating env params: {e}")

    def cleanup(self):
        # ... (remains the same) ...
        """Shuts down the thread pool executor."""
        logger.info("Shutting down learn executor...")
        try:
            self.learn_executor.shutdown(wait=True, cancel_futures=False)
            logger.info("Learn executor shut down successfully.")
        except Exception as e:
            logger.error(f"Error shutting down learn executor: {e}", exc_info=True)

# --- END OF FILE orchestrator.py ---