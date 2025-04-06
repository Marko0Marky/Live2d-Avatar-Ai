# --- START OF FILE orchestrator.py ---
import torch
from typing import Dict, Tuple, Optional, List, Union # Added Union
import concurrent.futures
import asyncio
import time

# Use MasterConfig
from config import MasterConfig as Config
from config import DEVICE, logger
from environment import EmotionalSpace
from agent import ConsciousAgent
from graphics import Live2DCharacter
from utils import is_safe, Experience, MetaCognitiveMemory # Import Experience, MCM

# --- Define return type alias for reflect *before* it's used ---
# Use Union for list element type flexibility if needed later
ReflectReturnType = Dict[str, Union[float, List[float], int]]
# --- Define return type alias for train_step *before* it's used ---
TrainStepReturnType = Tuple[ReflectReturnType, float, bool, str, float] # Use ReflectReturnType here


class EnhancedConsciousAgent: # Orchestrator Class
    """Orchestrates interaction, triggers batched learning asynchronously, and handles chat."""
    def __init__(self, train_interval: int = Config.RL.AGENT_TRAIN_INTERVAL, batch_size: int = Config.RL.AGENT_BATCH_SIZE, num_workers: int = 1): # Added type hints
        logger.info("Initializing EnhancedConsciousAgent Orchestrator (Async Batched Learning + Chat)...")
        self.env = EmotionalSpace()
        self.model = ConsciousAgent().to(DEVICE) # Uses updated config internally
        self.avatar = Live2DCharacter()
        logger.debug("Live2DCharacter display instance created for orchestrator.")

        self.train_interval = max(1, train_interval)
        self.batch_size = batch_size

        self.episode_rewards: List[float] = [] # Now List is defined
        self.current_episode_reward_sum: float = 0.0
        self.total_steps: int = 0; self.episode_steps: int = 0; self.episode_count: int = 0
        self.last_reward: float = 0.0; self.last_reported_loss: float = 0.0
        self.learn_step_running: bool = False

        self.last_response_emotions: torch.Tensor = self.model.prev_emotions.clone().detach() # Type hint tensor
        self.current_response: str = "Initializing..." # Type hint str

        self.num_workers = max(1, num_workers)
        self.learn_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix='LearnWorker')
        self.learn_future: Optional[concurrent.futures.Future] = None

        try:
             self.current_state: torch.Tensor = self.env.reset() # Type hint tensor
             assert is_safe(self.current_state) and self.current_state.shape == (Config.Agent.STATE_DIM,)
        except Exception as e: logger.critical(f"Orchestrator Init Error: {e}", exc_info=True); raise RuntimeError("Init state failed.") from e

        self.current_event: Optional[str] = self.env.current_event_type # Type hint optional str
        logger.info(f"Orchestrator initialized (Async Learn, {self.num_workers} worker(s), Chat Enabled).")

    def set_hud_widget(self, hud_widget): pass # Keep signature if needed elsewhere

    def _run_learn_task(self) -> Optional[float]:
        loss = None
        try:
            if len(self.model.memory) >= self.batch_size:
                 loss = self.model.learn(self.batch_size)
            else:
                 loss = 0.0
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

    # Use the predefined type alias
    def train_step(self) -> TrainStepReturnType: # Now TrainStepReturnType is defined
        """ Performs internal simulation step, stores experience, triggers async learning."""
        self._check_learn_future()

        if self.current_state is None or not is_safe(self.current_state) or self.current_state.shape != (Config.Agent.STATE_DIM,):
            logger.error(f"Orchestrator train_step: Invalid state. Resetting.");
            try: self.current_state = self.env.reset(); assert is_safe(self.current_state) and self.current_state.shape==(Config.Agent.STATE_DIM,)
            except: logger.critical("Reset failed."); return ({'error': True, 'message':"FATAL STATE"}, -1.0, True, "FATAL STATE", self.last_reported_loss)

        self.total_steps += 1; self.episode_steps += 1
        state_before_step = self.current_state.clone().detach()

        try: next_state, reward, env_done, context_internal, event_type = self.env.step()
        except Exception as e: logger.error(f"Error env step: {e}"); next_state = state_before_step; reward = -0.1; env_done = False; context_internal = "Error env step."; event_type = self.current_event
        if not isinstance(next_state, torch.Tensor) or next_state.shape != (Config.Agent.STATE_DIM,) or not is_safe(next_state):
            logger.error("Env invalid next_state."); next_state = state_before_step; reward = -0.1; env_done = False; context_internal = "Env State Error"; event_type = self.current_event

        self.last_reward = reward; self.current_event = event_type; self.current_episode_reward_sum += reward

        try:
            emotions_internal, response_internal, belief_for_memory, att_score_metric = self.model.step(state_before_step, reward, self.model.state_history, context_internal)
        except Exception as e:
             logger.error(f"Error agent step: {e}");
             emotions_internal = self.model.prev_emotions if is_safe(self.model.prev_emotions) else torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE);
             response_internal = "(Agent step error)"; belief_for_memory = torch.zeros(Config.Agent.HIDDEN_DIM, device=DEVICE); att_score_metric = 0.0

        initial_td_error = MetaCognitiveMemory.INITIAL_TD_ERROR
        with torch.no_grad():
             try:
                 outputs_s = self.model.forward(state_before_step, reward, self.model.state_history)
                 outputs_sp = self.model.forward(next_state, 0.0, None)
                 if len(outputs_s) == 12 and len(outputs_sp) == 12:
                     current_value = outputs_s[3]; next_value = outputs_sp[3]
                     if is_safe(current_value) and is_safe(next_value):
                         target_val = reward + Config.RL.GAMMA * next_value * (0.0 if env_done else 1.0)
                         initial_td_error = abs((target_val - current_value).item())
                 else: logger.warning("Forward mismatch during TD error estimation.")
             except Exception as e: logger.warning(f"Could not estimate initial TD error: {e}")
        try:
            belief_to_store = belief_for_memory if isinstance(belief_for_memory, torch.Tensor) and is_safe(belief_for_memory) else None
            exp = Experience(state_before_step, belief_to_store, reward, next_state.detach(), env_done, initial_td_error)
            self.model.memory.add(exp)
        except Exception as e: logger.error(f"Failed adding experience: {e}")

        if not self.learn_step_running and (self.total_steps % self.train_interval == 0) and (len(self.model.memory) >= self.batch_size):
            try: self.learn_future = self.learn_executor.submit(self._run_learn_task); self.learn_step_running = True
            except Exception as e: logger.error(f"Failed submit learn task: {e}"); self.learn_step_running = False

        self.current_state = next_state.detach().clone()

        done = env_done
        if done:
            logger.info(f"Episode {self.episode_count + 1} finished. Steps: {self.episode_steps}. Reward: {self.current_episode_reward_sum:.2f}. Last Learn Loss: {self.last_reported_loss:.4f}")
            self.episode_rewards.append(self.current_episode_reward_sum); self.episode_count += 1
            try: self.current_state = self.env.reset(); assert is_safe(self.current_state) and self.current_state.shape == (Config.Agent.STATE_DIM,)
            except Exception as e: logger.critical(f"CRITICAL: Failed reset: {e}."); done = True; self.current_state = None; response = "FATAL RESET"; return ({'error': True, 'message':"FATAL RESET"}, reward, done, response, self.last_reported_loss)
            self.model.prev_emotions.zero_()
            self.last_response_emotions.zero_()
            self.episode_steps = 0; self.current_episode_reward_sum = 0.0; self.current_event = self.env.current_event_type;

        metrics_dict = self.reflect()
        return metrics_dict, reward, done, self.current_response, self.last_reported_loss

    def handle_user_chat(self, user_text: str) -> str:
        """Processes user chat, dynamically blends emotions, generates response, updates avatar."""
        logger.info(f"Handling user chat: '{user_text[:50]}...'")
        if not isinstance(user_text, str) or not user_text.strip(): return "..."

        try:
            impact_vector = self.env.get_emotional_impact_from_text(user_text)
            current_response_emotions = self.last_response_emotions.clone().detach()

            current_max_emo = current_response_emotions.max().item()
            blend_factor_user = 0.7 - (current_max_emo * 0.3)
            blend_factor_user = max(0.4, min(0.7, blend_factor_user))
            blend_factor_current = 1.0 - blend_factor_user
            logger.debug(f"Emotion blend factor (user): {blend_factor_user:.2f}")

            blended_emotions = torch.clamp(
                current_response_emotions * blend_factor_current + impact_vector * blend_factor_user,
                0.0, 1.0
            )
            if not is_safe(blended_emotions):
                logger.warning("Blended emotions unsafe, using previous."); blended_emotions = current_response_emotions

            self.last_response_emotions = blended_emotions.clone().detach()
            self.model.prev_emotions = self.model.prev_emotions * 0.5 + self.last_response_emotions * 0.5
            self.model.prev_emotions = torch.clamp(self.model.prev_emotions, 0.0, 1.0)

            if self.avatar: self.avatar.update_emotions(self.last_response_emotions)

            response = self.model.gpt.generate(context=user_text, emotions=self.last_response_emotions)
            self.current_response = response

            logger.debug(f"Generated response: '{response}' with emotions: {self.last_response_emotions.cpu().numpy()}")
            return self.current_response

        except Exception as e:
            logger.error(f"Error handling user chat: {e}", exc_info=True)
            self.current_response = "Sorry, I had trouble thinking about that."
            return self.current_response

    def reflect(self) -> ReflectReturnType: # Now ReflectReturnType is defined
        """Collects various metrics and states for logging/display."""
        stats_dict: ReflectReturnType = {
            "avg_reward_last20": 0.0, "total_steps": self.total_steps, "episode": self.episode_count,
            "current_emotions_internal": [0.0]*Config.Agent.EMOTION_DIM,
            "current_emotions_response": [0.0]*Config.Agent.EMOTION_DIM,
            "I_S": 0.0, "rho_struct": 0.0, "att_score": 0.0, "self_consistency": 0.0,
            "rho_score": 0.0, "box_score": 0.0, "tau_t": 0.0, "R_acc": 0.0,
            "loss": self.last_reported_loss, "rho_struct_mem_short": 0.0, "rho_struct_mem_long": 0.0
            }
        recent_rewards = self.episode_rewards[-20:];
        if recent_rewards: stats_dict["avg_reward_last20"] = sum(recent_rewards) / len(recent_rewards)
        try:
            if self.current_state is not None and is_safe(self.current_state) and self.current_state.shape == (Config.Agent.STATE_DIM,):
                self.model.eval()
                with torch.no_grad():
                     reflect_outputs = self.model.forward(self.current_state, self.last_reward, self.model.state_history)
                     if len(reflect_outputs) == 12:
                         (emotions_int, _, _, _value, I_S, _, att_score, self_consistency, rho_score, box_score, R_acc_mean, tau_t) = reflect_outputs;
                         stats_dict.update({
                              "current_emotions_internal": emotions_int.cpu().tolist() if is_safe(emotions_int) else [0.0]*Config.Agent.EMOTION_DIM,
                              "I_S": I_S, "att_score": att_score, "self_consistency": self_consistency,
                              "rho_score": rho_score, "box_score": box_score, "tau_t": tau_t, "R_acc": R_acc_mean
                              })
                     else: logger.warning(f"Reflection: forward returned {len(reflect_outputs)} values, expected 12.")
            else: logger.warning(f"Reflection skipped: Invalid state.")
        except Exception as e: logger.error(f"Error reflect: {e}", exc_info=False);
        try:
             stats_dict["rho_struct_mem_short"] = self.model.memory.get_short_term_norm();
             stats_dict["rho_struct_mem_long"] = self.model.memory.get_long_term_norm();
             stats_dict["rho_struct"] = stats_dict["rho_struct_mem_short"] * 0.3 + stats_dict["rho_struct_mem_long"] * 0.7
             stats_dict["current_emotions_response"] = self.last_response_emotions.cpu().tolist()
        except Exception as e: logger.error(f"Error memory norms/response emotions reflect: {e}", exc_info=False)
        return stats_dict


    def test_completeness(self) -> Tuple[bool, str]:
        logger.info("Performing Completeness Test...")
        if self.current_state is None or not is_safe(self.current_state) or self.current_state.shape != (Config.Agent.STATE_DIM,): return False, "Invalid state"
        test_state = self.current_state.clone().detach();
        if Config.Agent.EMOTION_DIM >= 2: test_state[0] = 0.9; test_state[1] = 0.1;
        test_reward = 1.5; consistent = False; details = "Prerequisites failed."
        try:
            self.model.eval()
            with torch.no_grad():
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

    def update_environment(self, event_freq: float, intensities: List[float]): # Now List is defined
        try: self.env.update_params(event_freq, intensities)
        except Exception as e: logger.error(f"Error updating env params: {e}")

    def cleanup(self):
        """Shuts down the thread pool executor."""
        logger.info("Shutting down learn executor...")
        try:
            # Consider adding a timeout to shutdown
            self.learn_executor.shutdown(wait=True, cancel_futures=False) # wait=True is generally safer
            logger.info("Learn executor shut down successfully.")
        except Exception as e:
            logger.error(f"Error shutting down learn executor: {e}", exc_info=True)

# --- END OF FILE orchestrator.py ---
