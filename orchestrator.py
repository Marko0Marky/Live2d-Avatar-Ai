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
import pickle
import html
import os

# Assume SentenceTransformers are not used in this version
SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import MasterConfig as Config, DEVICE, logger
from config import AGENT_SAVE_PATH, GPT_SAVE_PATH, OPTIMIZER_SAVE_PATH, TARGET_NET_SAVE_SUFFIX, REPLAY_BUFFER_SAVE_PATH
from environment import EmotionalSpace # Environment handles 12D state and qualia feedback
from agent import ConsciousAgent, AgentForwardReturnType # Import refactored agent and its return type
from graphics import Live2DCharacter
from utils import is_safe, Experience, MetaCognitiveMemory # Uses simplified Experience

# Type Aliases
ReflectReturnType = Dict[str, Union[float, List[float], int, str, np.ndarray]]
# train_step returns metrics, reward, done, response, loss, "" (empty monologue), "idle" (legacy HM)
TrainStepReturnType = Tuple[ReflectReturnType, float, bool, str, float, str, str]

class EnhancedConsciousAgent:
    def __init__(self, train_interval: int = Config.RL.AGENT_TRAIN_INTERVAL, batch_size: int = Config.RL.AGENT_BATCH_SIZE, num_workers: int = 1):
        logger.info("Initializing Orchestrator (RRDT/PyG Agent v5)...")
        self.env = EmotionalSpace()
        self.model = ConsciousAgent().to(DEVICE)
        self.avatar = Live2DCharacter()
        logger.debug("Live2DCharacter instance created.")

        self.train_interval = max(1, train_interval); self.batch_size = batch_size
        self.episode_rewards: List[float] = []; self.current_episode_reward_sum: float = 0.0
        self.total_steps: int = 0; self.episode_steps: int = 0; self.episode_count: int = 0
        self.last_reward: float = 0.0; self.last_reported_loss: float = 0.0; self.learn_step_running: bool = False
        self.last_response_emotions: torch.Tensor = self.model.prev_emotions.clone().detach()
        self.current_response: str = "Initializing..." # Holds the last *chat* response
        self.mood: torch.Tensor = torch.rand(Config.Agent.EMOTION_DIM, device=DEVICE) * 0.5 # Init mood
        self.last_internal_monologue: str = "" # Keep variable but don't generate new content
        self.conversation_history: Deque[Tuple[str, str]] = deque(maxlen=Config.NLP.CONVERSATION_HISTORY_LENGTH * 2)
        self.num_workers = max(1, num_workers); self.learn_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix='LearnWorker'); self.learn_future: Optional[concurrent.futures.Future] = None
        self.hud_widget = None; self._cleaned_up = False

        try:
            self.current_state: torch.Tensor = self.env.reset() # Env provides 12D state
            if not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM: raise RuntimeError(f"Initial state invalid. Shape: {self.current_state.shape}")
            # Initialize agent's history
            for _ in range(Config.Agent.HISTORY_SIZE): self.model.state_history_deque.append(self.current_state.clone().detach())
        except Exception as e: logger.critical(f"Orchestrator Init State Error: {e}"); raise RuntimeError("Init state failed.") from e
        self.current_event: Optional[str] = self.env.current_event_type
        logger.info(f"Orchestrator initialized for 12D state (StateDim={Config.Agent.STATE_DIM}).")

    def set_hud_widget(self, hud_widget): self.hud_widget = hud_widget; logger.debug("HUD widget ref stored.")

    def _run_learn_task(self) -> Optional[float]:
        """Background task: samples batch, calls agent.learn."""
        avg_loss = None
        try:
            if len(self.model.memory) >= self.batch_size:
                 sample = self.model.memory.sample(self.batch_size, self.model.beta)
                 if sample:
                      batch_data, indices, weights = sample
                      avg_loss = self.model.learn(batch_data, indices, weights)
                      if avg_loss is None: avg_loss = -1.0
                 else: avg_loss = 0.0
            else: avg_loss = 0.0
        except Exception as e: logger.error(f"Exception in learn task thread: {e}", exc_info=True); avg_loss = -1.0
        return avg_loss

    def _check_learn_future(self):
        """Checks status of background learning task."""
        if self.learn_future and self.learn_future.done():
            try:
                loss_result = self.learn_future.result()
                if loss_result is not None and loss_result >= -1e-7: self.last_reported_loss = loss_result
                else: logger.warning(f"Async learn task error/no loss ({loss_result})."); self.last_reported_loss = -1.0
            except Exception as e: logger.error(f"Error getting learn result: {e}"); self.last_reported_loss = -1.0
            finally: self.learn_future = None; self.learn_step_running = False

    def train_step(self) -> TrainStepReturnType:
        """Performs one step: Agent Forward -> Env Feedback -> Env Step -> Trigger Learn."""
        self._check_learn_future()

        # --- State Validation ---
        if self.current_state is None or not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM:
            logger.error(f"Orch invalid state: {getattr(self.current_state, 'shape', 'None')}. Resetting.");
            try: self.current_state = self.env.reset(); assert is_safe(self.current_state) and self.current_state.shape[0] == Config.Agent.STATE_DIM; logger.info("State reset successful.")
            except Exception as reset_err: logger.critical(f"RESET FAILED: {reset_err}"); default_metrics = {'error': True, 'message': 'RESET FAIL'}; return (default_metrics, -1.0, True, "FATAL", -1.0, "", "idle")

        self.total_steps += 1; self.episode_steps += 1
        state_before_env_step = self.current_state.clone().detach()

        # --- 1. Agent Forward Pass (on current state for internal calcs & feedback) ---
        att_score_metric = 0.0; qualia_feedback = None
        emotions_internal = self.model.prev_emotions # Use previous as default if forward fails
        try:
             # Run forward pass to get internal states, metrics, and qualia for feedback
             forward_output = self.model.forward_single(state_before_env_step, self.last_reward, self.model.state_history)
             emotions_internal = forward_output.emotions[0]
             att_score_metric = forward_output.att_score[0].item() # Needed for gating/potential decisions
             agent_full_state = forward_output.full_state[0]
             qualia_feedback = agent_full_state[self.model.emotion_dim:] # Extract R7-12 feedback

             # --- *** DO NOT GENERATE CHAT RESPONSE HERE *** ---

             # Update agent's internal history deque *after* using it in forward_single
             self.model.state_history_deque.append(state_before_env_step)

        except Exception as e: logger.error(f"Error during agent forward pass: {e}", exc_info=True); qualia_feedback = torch.zeros(Config.Agent.QUALIA_DIM, device=DEVICE)

        # Store the latest calculated internal emotions for potential display/mood update
        self.last_response_emotions = emotions_internal.clone().detach() # Renaming this might be clearer (e.g., last_internal_emotions)

        # Ensure internal monologue is not generated or used
        self.last_internal_monologue = ""

        # --- 2. Environment Feedback & Step ---
        self.env.update_qualia_feedback(qualia_feedback) # Give feedback computed above
        try: next_state, reward, env_done, context_internal, event_type = self.env.step()
        except Exception as e: logger.error(f"Env step error: {e}"); next_state = state_before_env_step; reward = -0.1; env_done = False; context_internal = "Err"; event_type = self.current_event
        if not is_safe(next_state) or next_state.shape[0] != Config.Agent.STATE_DIM: logger.error(f"Env returned invalid next_state."); next_state = state_before_env_step; reward = -0.1; env_done = False; context_internal = "Env State Err";
        self.last_reward = reward; self.current_event = event_type; self.current_episode_reward_sum += reward

        # --- 3. Trigger Learning Task ---
        # Experience (s, r, s', d) is implicitly added by agent.learn via sampling
        if not self.learn_step_running and (self.total_steps % self.train_interval == 0) and (len(self.model.memory) >= self.batch_size):
            try: self.learn_future = self.learn_executor.submit(self._run_learn_task); self.learn_step_running = True
            except Exception as e: logger.error(f"Failed submit learning task: {e}"); self.learn_step_running = False

        # --- 4. Update Orchestrator's Current State ---
        self.current_state = next_state.detach().clone()

        # --- 5. Update Mood ---
        try:
            if is_safe(emotions_internal) and is_safe(self.mood):
                decay = Config.RL.MOOD_UPDATE_DECAY
                self.mood = decay * self.mood + (1.0 - decay) * emotions_internal
                self.mood = torch.clamp(self.mood, 0.0, 1.0)
            elif not is_safe(self.mood): logger.warning("Mood unsafe. Resetting."); self.mood.fill_(0.3)
        except AttributeError as ae: logger.error(f"Mood update failed: Missing config? {ae}")
        except Exception as e: logger.error(f"Error updating mood: {e}")

        # --- 6. Episode Reset Handling ---
        done = env_done
        if done:
            logger.info(f"--- Episode {self.episode_count + 1} Ended (Steps: {self.episode_steps}, Reward: {self.current_episode_reward_sum:.2f}) ---")
            self.episode_rewards.append(self.current_episode_reward_sum); self.episode_count += 1
            try:
                self.current_state = self.env.reset()
                if not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM: raise RuntimeError("Reset invalid state.")
                self.model.prev_emotions.zero_(); self.last_response_emotions.zero_(); self.mood.fill_(0.3); self.conversation_history.clear(); self.episode_steps = 0; self.current_episode_reward_sum = 0.0; self.current_event = self.env.current_event_type;
                self.model.prev_belief_embedding = None
                for _ in range(Config.Agent.HISTORY_SIZE): self.model.state_history_deque.append(self.current_state.clone().detach())
            except Exception as e: logger.critical(f"CRITICAL: Failed episode reset: {e}."); done = True; default_metrics = {'error': True, 'message': 'RESET FAIL'}; return (default_metrics, reward, done, "FATAL", -1.0, "", "idle")

        # --- Return Results ---
        metrics_dict = self.reflect()
        # Return the last CHAT response stored in self.current_response
        return metrics_dict, reward, done, self.current_response, self.last_reported_loss, "", "idle"

    def handle_user_chat(self, user_text: str) -> str:
        """Handles user chat input, generates response via agent's GPT."""
        logger.info(f"Handling user chat: '{user_text[:50]}...'")
        if not isinstance(user_text, str) or not user_text.strip(): return "..."
        safe_user_text = html.escape(user_text.replace('\n', ' ').strip())
        self.conversation_history.append(("User", safe_user_text))
        try:
            # --- Emotion Blending for Avatar ---
            impact_vector = self.env.get_emotional_impact_from_text(safe_user_text)
            # Use the *actual* last response emotions for blending basis
            current_display_emotions = self.last_response_emotions.clone().detach()
            mood_influence = 0.15
            biased_display_emotions = torch.clamp( current_display_emotions * (1.0 - mood_influence) + self.mood * mood_influence, 0.0, 1.0 )
            current_max_emo = biased_display_emotions.max().item(); blend_factor_user = max(0.4, min(0.7, 0.7 - (current_max_emo * 0.3))); blend_factor_current = 1.0 - blend_factor_user
            blended_emotions = torch.clamp( biased_display_emotions * blend_factor_current + impact_vector * blend_factor_user, 0.0, 1.0 )
            if not is_safe(blended_emotions): blended_emotions = biased_display_emotions
            # Update self.last_response_emotions as this IS the emotion state tied to the CHAT response
            self.last_response_emotions = blended_emotions.clone().detach()
            if self.avatar: self.avatar.update_emotions(self.last_response_emotions)

            # --- Context Building ---
            context_for_gpt = ""; history_turns = list(self.conversation_history); temp_hist_context = ""
            max_hist_len = 512
            for speaker, text in reversed(history_turns):
                turn = f"{speaker}: {text}\n"
                if len(temp_hist_context) + len(turn) > max_hist_len: break
                temp_hist_context = turn + temp_hist_context
            context_for_gpt = temp_hist_context.strip()
            if not context_for_gpt.endswith("\nAI:"):
                 if context_for_gpt.endswith("AI:"): context_for_gpt = context_for_gpt[:-3].strip()
                 context_for_gpt += "\nAI:"

            # --- Generate Response using Agent's GPT ---
            current_metrics = self.reflect()
            att_score_for_chat = current_metrics.get("att_score", Config.Agent.ATTENTION_THRESHOLD + 0.1)
            # Agent's generate_response now handles prepending internal state context
            raw_response = self.model.generate_response(context_for_gpt, att_score_for_chat)

            response_unescaped = html.unescape(raw_response)
            final_response_clean = response_unescaped.replace('\n', ' ').strip() or "..."
            # *** This is where the response for chat is set ***
            self.current_response = final_response_clean
            self.conversation_history.append(("AI", self.current_response))

            # --- Update Avatar Movement (Legacy/Placeholder) ---
            if self.avatar and hasattr(self.avatar, 'update_predicted_movement'): self.avatar.update_predicted_movement("idle")

            logger.debug(f"Chat response: '{self.current_response}' | Display Emo: {self.last_response_emotions.cpu().numpy().round(2)}")
            return self.current_response
        except Exception as e: logger.error(f"Error handling chat: {e}", exc_info=True); self.current_response = "[Chat Error]"; return self.current_response # Return error message

    def reflect(self) -> ReflectReturnType:
        """ Gathers agent metrics by running a forward pass on the current state. """
        stats_dict: ReflectReturnType = { "avg_reward_last20": 0.0, "total_steps": self.total_steps, "episode": self.episode_count, "current_emotions_internal": [0.0]*Config.Agent.EMOTION_DIM, "current_emotions_response": [0.0]*Config.Agent.EMOTION_DIM, "current_mood": [0.0]*Config.Agent.EMOTION_DIM, "last_monologue": "", "loss": self.last_reported_loss, "I_S": 0.0, "rho_score": 0.0, "tau_t": 0.0, "stability": 0.0, "zeta": 0.0, "att_score": 0.0, "self_consistency": 0.0, "state_norm": 0.0, "belief_norm_long": 0.0, "lyapunov_proxy": 0.0 }
        recent_rewards = self.episode_rewards[-20:];
        if recent_rewards: stats_dict["avg_reward_last20"] = sum(recent_rewards) / len(recent_rewards)
        try:
            current_state_for_reflect = self.current_state
            if current_state_for_reflect is not None and is_safe(current_state_for_reflect) and current_state_for_reflect.shape[0] == Config.Agent.STATE_DIM:
                reflect_outputs = self.model.forward_single(current_state_for_reflect, 0.0, self.model.state_history)
                if reflect_outputs.emotions is not None: stats_dict["current_emotions_internal"]=reflect_outputs.emotions[0].cpu().tolist()
                if reflect_outputs.integration_I is not None: stats_dict["I_S"] = reflect_outputs.integration_I[0].item()
                if reflect_outputs.rho_score is not None: stats_dict["rho_score"] = reflect_outputs.rho_score[0].item()
                if reflect_outputs.zeta is not None: stats_dict["zeta"] = reflect_outputs.zeta[0].item()
                if reflect_outputs.stability is not None: stats_dict["stability"] = reflect_outputs.stability[0].item()
                if reflect_outputs.att_score is not None: stats_dict["att_score"] = reflect_outputs.att_score[0].item()
                if reflect_outputs.self_consistency is not None: stats_dict["self_consistency"]=reflect_outputs.self_consistency[0].item()
                if reflect_outputs.tau_t is not None: stats_dict["tau_t"] = reflect_outputs.tau_t[0].item()
                if reflect_outputs.lyapunov_max_proxy is not None: stats_dict["lyapunov_proxy"] = reflect_outputs.lyapunov_max_proxy[0].item()
            else: logger.warning(f"Reflection skipped: Invalid current_state.")
        except Exception as e: logger.error(f"Error during agent reflection forward pass: {e}", exc_info=False);
        try:
             stats_dict["belief_norm_long"] = self.model.memory.get_long_term_norm();
             stats_dict["current_emotions_response"] = self.last_response_emotions.cpu().tolist(); # Display last *actual* response emotions
             stats_dict["current_mood"] = self.mood.cpu().tolist() if is_safe(self.mood) else [0.0]*Config.Agent.EMOTION_DIM
             if self.current_state is not None and is_safe(self.current_state): stats_dict["state_norm"] = torch.linalg.norm(self.current_state.float()).item()
        except Exception as e: logger.error(f"Error getting other reflect stats: {e}")
        return stats_dict

    def test_completeness(self) -> Tuple[bool, str]:
        """ Performs RIH completeness test. """
        logger.info("Performing Completeness Test (RIH version)...");
        current_sim_state = self.current_state
        if not is_safe(current_sim_state) or current_sim_state.shape[0] != Config.Agent.STATE_DIM: return False, "Invalid state"
        try:
            test_outputs = self.model.forward_single(current_sim_state, 0.0, self.model.state_history)
            integration_I = test_outputs.integration_I[0].item() if test_outputs.integration_I is not None else -1.0
            rho_score = test_outputs.rho_score[0].item() if test_outputs.rho_score is not None else -1.0
            tau_t = test_outputs.tau_t[0].item() if test_outputs.tau_t is not None else float('inf')
            stability = test_outputs.stability[0].item() if test_outputs.stability is not None else -1.0
            integration_check = integration_I >= tau_t; reflexivity_check = rho_score >= Config.RL.RHO_SIMILARITY_THRESHOLD
            is_complete = integration_check and reflexivity_check
            details = (f"I_S={integration_I:.3f} >= Tau(t)={tau_t:.3f}? ({integration_check}), Rho={rho_score:.3f} >= {Config.RL.RHO_SIMILARITY_THRESHOLD}? ({reflexivity_check}), Stab={stability:.3f}")
        except Exception as e: logger.error(f"Completeness test error: {e}"); is_complete, details = False, f"Exception: {e}"
        logger.info(f"Completeness Test Result: {is_complete}. Details: {details}");
        return is_complete, details

    def update_environment(self, event_freq: float, intensities: List[float]):
        try: self.env.update_params(event_freq, intensities);
        except Exception as e: logger.error(f"Error updating env params: {e}")

    def save_agent(self):
        logger.info("Orchestrator: Saving agent state...");
        self.model.save_state() # Agent method handles paths
        try:
            buffer_path = REPLAY_BUFFER_SAVE_PATH; os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
            with open(buffer_path, 'wb') as f: pickle.dump(self.model.memory, f)
            logger.info(f"Replay buffer saved to {buffer_path}")
        except Exception as e: logger.error(f"Failed to save replay buffer: {e}", exc_info=True)

    def load_agent(self) -> bool:
        logger.info("Orchestrator: Loading agent state...");
        loaded_model = self.model.load_state() # Agent method handles paths
        if loaded_model:
            try: # Load replay buffer
                buffer_path = REPLAY_BUFFER_SAVE_PATH
                if os.path.exists(buffer_path):
                    with open(buffer_path, 'rb') as f: loaded_memory = pickle.load(f)
                    if isinstance(loaded_memory, MetaCognitiveMemory): self.model.memory = loaded_memory; self.model.beta = Config.RL.PER_BETA_START + (1.0 - Config.RL.PER_BETA_START) * min(1.0, self.model.step_count / max(1, Config.RL.PER_BETA_FRAMES)); logger.info(f"Replay buffer loaded. Size: {len(self.model.memory)}, Beta: {self.model.beta:.4f}")
                    else: logger.error("Loaded object not MetaCognitiveMemory."); self.model.memory = MetaCognitiveMemory()
                else: logger.warning(f"Replay buffer not found: {buffer_path}."); self.model.memory = MetaCognitiveMemory()
            except Exception as e: logger.error(f"Failed to load replay buffer: {e}"); self.model.memory = MetaCognitiveMemory()
            self.last_response_emotions = self.model.prev_emotions.clone().detach(); self.total_steps = self.model.step_count; self.mood.fill_(0.4)
            logger.info("Orchestrator state synced with loaded model.")
            return True
        else: logger.error("Agent model loading failed."); return False

    @property
    def cleaned_up(self) -> bool: return self._cleaned_up
    def cleanup(self):
        if self._cleaned_up: return
        logger.info("--- Orchestrator Cleanup Initiated ---")
        logger.info("Shutting down learn executor...");
        try:
            if self.learn_future and not self.learn_future.done(): concurrent.futures.wait([self.learn_future], timeout=0.5)
            self.learn_executor.shutdown(wait=True, cancel_futures=True); logger.info("Learn executor shut down.")
        except Exception as e: logger.error(f"Error shutting down learn executor: {e}")
        if self.avatar and hasattr(self.avatar, 'cleanup'):
            try: logger.info("Cleaning up avatar..."); self.avatar.cleanup(); logger.info("Avatar cleanup complete.")
            except Exception as e: logger.error(f"Error during avatar cleanup: {e}")
        self._cleaned_up = True; logger.info("--- Orchestrator Cleanup Finished ---")

# --- END OF FILE orchestrator.py ---
