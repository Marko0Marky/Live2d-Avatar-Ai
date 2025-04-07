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

try: from sentence_transformers import SentenceTransformer; SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError: SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import MasterConfig as Config
from config import DEVICE, logger
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
        self.model = ConsciousAgent().to(DEVICE)
        self.avatar = Live2DCharacter()
        logger.debug("Live2DCharacter display instance created.")
        self.st_model: Optional[SentenceTransformer] = None
        if Config.Agent.USE_LANGUAGE_EMBEDDING:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Error L99: Try needs except/finally; Error L100: Indentation; Error L103, L104, L105: Expected expression
                try: # Corrected indentation
                    model_name = Config.NLP.SENTENCE_TRANSFORMER_MODEL
                    logger.info(f"Loading sentence transformer: {model_name}...")
                    self.st_model = SentenceTransformer(model_name, device=str(DEVICE))
                    with torch.no_grad(): # Added no_grad for check
                        test_emb = self.st_model.encode(["test"], convert_to_tensor=True)[0]
                    actual_dim = test_emb.shape[0]
                    expected_dim = Config.Agent.LANGUAGE_EMBEDDING_DIM
                    if actual_dim != expected_dim:
                         logger.critical(f"FATAL: ST model dim ({actual_dim}) != Config dim ({expected_dim}).")
                         sys.exit(1)
                    logger.info(f"Sentence transformer loaded to {self.st_model.device}. Output dim: {actual_dim}.")
                except Exception as e: # Added except block
                    logger.error(f"Failed load ST model '{Config.NLP.SENTENCE_TRANSFORMER_MODEL}': {e}. Disabling.", exc_info=True)
                    self.st_model = None
                    Config.Agent.USE_LANGUAGE_EMBEDDING = False
                    Config.Agent.__post_init__(Config.Agent)
                    logger.warning(f"Lang embedding disabled. Agent STATE_DIM recalced: {Config.Agent.STATE_DIM}")
            else:
                 logger.warning("SentenceTransformers not found. Disabling language embedding.")
                 Config.Agent.USE_LANGUAGE_EMBEDDING = False
                 Config.Agent.__post_init__(Config.Agent)
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

    # ... (_get_combined_state, set_hud_widget, _run_learn_task, _check_learn_future unchanged) ...
    def _get_combined_state(self, base_state: torch.Tensor, context: str = "step") -> torch.Tensor:
        if base_state is None or not isinstance(base_state, torch.Tensor): logger.error(f"[{context}] Invalid base_state: {type(base_state)}. Zeros."); return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)
        base_state_dev = base_state.to(DEVICE)
        if base_state_dev.shape[0] != Config.Agent.BASE_STATE_DIM: logger.error(f"[{context}] Base state dim mismatch! Exp {Config.Agent.BASE_STATE_DIM}, got {base_state_dev.shape}. Padding/Truncating."); padded_base = torch.zeros(Config.Agent.BASE_STATE_DIM, device=DEVICE, dtype=base_state_dev.dtype); copy_len = min(base_state_dev.shape[0], Config.Agent.BASE_STATE_DIM); padded_base[:copy_len] = base_state_dev[:copy_len]; base_state_dev = padded_base
        if Config.Agent.USE_LANGUAGE_EMBEDDING:
            if self.last_text_embedding is None: current_embedding = torch.zeros(Config.Agent.LANGUAGE_EMBEDDING_DIM, device=DEVICE, dtype=torch.float32)
            else: current_embedding = self.last_text_embedding.to(DEVICE)
            if current_embedding.shape[0] != Config.Agent.LANGUAGE_EMBEDDING_DIM: logger.error(f"[{context}] Text embed dim mismatch! Exp {Config.Agent.LANGUAGE_EMBEDDING_DIM}, got {current_embedding.shape}. Zeros."); current_embedding = torch.zeros(Config.Agent.LANGUAGE_EMBEDDING_DIM, device=DEVICE, dtype=torch.float32)
            try: combined = torch.cat((base_state_dev, current_embedding), dim=0)
            except Exception as e: logger.error(f"[{context}] Error cat base ({base_state_dev.shape}) embed ({current_embedding.shape}): {e}. Zeros."); return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)
            if combined.shape[0] != Config.Agent.STATE_DIM: logger.error(f"[{context}] Combined dim error! Exp {Config.Agent.STATE_DIM}, got {combined.shape}. Zeros."); return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)
            return combined
        else:
             if base_state_dev.shape[0] != Config.Agent.STATE_DIM: logger.error(f"[{context}] Lang embed disabled, base dim {base_state_dev.shape[0]} != config state dim {Config.Agent.STATE_DIM}. Critical config error!"); return torch.zeros(Config.Agent.STATE_DIM, device=DEVICE, dtype=base_state_dev.dtype)
             return base_state_dev

    def set_hud_widget(self, hud_widget): self.hud_widget = hud_widget; logger.debug("HUD widget ref stored.")
    
    def _run_learn_task(self) -> Optional[float]:
        loss = None # Initialize loss to None
        try:
            if len(self.model.memory) >= self.batch_size:
                 loss = self.model.learn(self.batch_size)
            else:
                 loss = 0.0
        except Exception as e:
             logger.error(f"Exception in learn task thread: {e}", exc_info=True)
             loss = -1.0 # Assign error value
        # Removed semicolon after loss assignment
        return loss
    
    def _check_learn_future(self):
        if self.learn_future and self.learn_future.done():
            try:
                loss_result = self.learn_future.result() # Removed semicolon
                if loss_result is not None and loss_result >= 0:
                    self.last_reported_loss = loss_result
                elif loss_result is not None:
                    logger.warning(f"Async learn task error indicator (Loss: {loss_result}).")
                    self.last_reported_loss = -1.0
                else:
                    logger.warning("Async learn task returned None.")
                    self.last_reported_loss = -1.0
            except concurrent.futures.CancelledError:
                logger.warning("Learn task was cancelled.")
                self.last_reported_loss = -1.0
            except Exception as e:
                logger.error(f"Exception get learn task result: {e}", exc_info=True)
                self.last_reported_loss = -1.0
            finally:
                self.learn_future = None
                self.learn_step_running = False # Removed semicolon

    def train_step(self) -> TrainStepReturnType:
        self._check_learn_future()
        if self.current_state is None or not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM:
            logger.error(f"Orchestrator invalid combined state ({getattr(self.current_state, 'shape', 'None')}). Reset.");
            try:
                self.current_base_state = self.env.reset();
                if not is_safe(self.current_base_state) or self.current_base_state.shape[0] != Config.Agent.BASE_STATE_DIM: raise RuntimeError("Reset failed: invalid base state.")
                self.last_text_embedding.zero_(); self.current_state = self._get_combined_state(self.current_base_state, "reset_recovery")
                if not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM: raise RuntimeError("Reset failed: invalid combined state.")
                logger.info("State reset successful after invalid state detected.")
            except Exception as reset_err:
                 logger.critical(f"CRITICAL RESET FAILED: {reset_err}. Stopping.");
                 default_metrics = {'error': True, 'message':"FATAL STATE RESET", "last_monologue":"", "embedding_norm": 0.0, "base_state_norm": 0.0}
                 return (default_metrics, -1.0, True, "FATAL STATE", self.last_reported_loss, "", "idle")
        self.total_steps += 1; self.episode_steps += 1
        state_before_step_combined = self.current_state.clone().detach()
        base_state_before_step = self.current_base_state.clone().detach() if self.current_base_state is not None else torch.zeros(Config.Agent.BASE_STATE_DIM, device=DEVICE)
        try: next_base_state, reward, env_done, context_internal, event_type = self.env.step()
        except Exception as e: logger.error(f"Error env step: {e}", exc_info=True); next_base_state = base_state_before_step; reward = -0.1; env_done = False; context_internal = "Error env step."; event_type = self.current_event
        if not isinstance(next_base_state, torch.Tensor) or next_base_state.shape[0] != Config.Agent.BASE_STATE_DIM or not is_safe(next_base_state): logger.error(f"Env invalid next_base_state (shape {getattr(next_base_state, 'shape', 'None')}). Use prev."); next_base_state = base_state_before_step; reward = -0.1; env_done = False; context_internal = "Env State Error"; event_type = self.current_event
        self.last_reward = reward; self.current_event = event_type; self.current_episode_reward_sum += reward
        att_score_metric = 0.0; response_internal = "(Agent step error)"; emotions_internal = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE); predicted_hm_label = "idle"
        belief_for_memory = None
        try: emotions_internal, response_internal, belief_for_memory, att_score_metric, predicted_hm_label = self.model.step(state_before_step_combined, reward, self.model.state_history, context_internal)
        except Exception as e: logger.error(f"Error agent step: {e}", exc_info=True); emotions_internal = self.model.prev_emotions if is_safe(self.model.prev_emotions) else torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE); belief_for_memory = torch.zeros(Config.Agent.HIDDEN_DIM, device=DEVICE); att_score_metric = 0.0; predicted_hm_label = "idle"
        self.last_internal_monologue = ""; current_monologue_embedding = torch.zeros_like(self.last_text_embedding)
        if Config.Agent.USE_LANGUAGE_EMBEDDING and self.st_model:
             try:
                 monologue_context = f"Internal state: {context_internal}. Feeling:"; temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7); top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9)
                 self.last_internal_monologue = self.model.gpt.generate(monologue_context, emotions_internal, temperature=temp, top_p=top_p, max_len=16)
                 if self.last_internal_monologue and self.last_internal_monologue != "...":
                     with torch.no_grad(): emb = self.st_model.encode([self.last_internal_monologue], convert_to_tensor=True, device=DEVICE);
                     if emb.ndim > 1: emb = emb.squeeze(0)
                     if emb.shape[0] == Config.Agent.LANGUAGE_EMBEDDING_DIM: current_monologue_embedding = emb.float()
                     else: logger.warning(f"Monologue embedding dim mismatch! Got {emb.shape}")
                 self.last_text_embedding = current_monologue_embedding.clone().detach()
             except Exception as e: logger.error(f"Error generating/embedding internal monologue: {e}"); self.last_text_embedding.zero_()
        else: self.last_text_embedding.zero_()
        next_combined_state = self._get_combined_state(next_base_state, "experience_next")
        initial_td_error = MetaCognitiveMemory.INITIAL_TD_ERROR
        if is_safe(state_before_step_combined) and is_safe(next_combined_state):
            try:
                self.model.eval()
                with torch.no_grad():
                     outputs_s = self.model.forward(state_before_step_combined.unsqueeze(0), torch.tensor([[reward]], device=DEVICE), None)
                     outputs_sp = self.model.forward(next_combined_state.unsqueeze(0), torch.tensor([[0.0]], device=DEVICE), None)
                     if len(outputs_s) == 13 and len(outputs_sp) == 13:
                         current_value = outputs_s[3].squeeze(); next_value = outputs_sp[3].squeeze()
                         if is_safe(current_value) and is_safe(next_value): target_val = reward + Config.RL.GAMMA * next_value * (0.0 if env_done else 1.0); initial_td_error = abs((target_val - current_value).item())
                         else: logger.warning("Unsafe V(s)/V(s') in TD estimate.")
                     else: logger.warning(f"Forward mismatch TD estimate (Exp 13, Got {len(outputs_s)}, {len(outputs_sp)}).")
                self.model.train()
            except Exception as e: logger.warning(f"Could not estimate initial TD error: {e}")
        else: logger.warning("Skip TD estimate: unsafe state(s).")
        should_store_long_term = not Config.RL.MEMORY_GATING_ENABLED or att_score_metric >= Config.RL.MEMORY_GATE_ATTENTION_THRESHOLD
        if should_store_long_term:
            try:
                belief_to_store = belief_for_memory if isinstance(belief_for_memory, torch.Tensor) and is_safe(belief_for_memory) else None
                base_priority = initial_td_error + 1e-5; attention_factor = 1.0 + Config.RL.PRIORITY_ATTENTION_WEIGHT * max(0.0, min(1.0, att_score_metric)); final_priority = base_priority * attention_factor
                target_hm_idx = HEAD_MOVEMENT_TO_IDX.get(predicted_hm_label, HEAD_MOVEMENT_TO_IDX["idle"])
                exp = Experience(state=state_before_step_combined.detach(), belief=belief_to_store.detach() if belief_to_store is not None else None, reward=reward, next_state=next_combined_state.detach(), done=env_done, td_error=final_priority, head_movement_idx=target_hm_idx)
                self.model.memory.add(exp)
            except Exception as e: logger.error(f"Failed adding experience: {e}", exc_info=True)
        if not self.learn_step_running and (self.total_steps % self.train_interval == 0) and (len(self.model.memory) >= self.batch_size):
            try: self.learn_future = self.learn_executor.submit(self._run_learn_task); self.learn_step_running = True
            except Exception as e: logger.error(f"Failed submit learn task: {e}"); self.learn_step_running = False
        self.current_base_state = next_base_state.detach().clone()
        self.current_state = self._get_combined_state(self.current_base_state, "state_update")
        try:
            internal_emotion_this_step = emotions_internal.detach()
            if is_safe(internal_emotion_this_step) and is_safe(self.mood): decay = Config.RL.MOOD_UPDATE_DECAY; self.mood = decay * self.mood + (1.0 - decay) * internal_emotion_this_step; self.mood = torch.clamp(self.mood, 0.0, 1.0)
            elif not is_safe(self.mood): logger.warning("Mood unsafe. Reset."); self.mood.fill_(0.3)
        except Exception as e: logger.error(f"Error updating mood: {e}")
        done = env_done
        if done:
            logger.info(f"--- Episode {self.episode_count + 1} ended (Steps: {self.episode_steps}, Reward: {self.current_episode_reward_sum:.2f}) ---")
            self.episode_rewards.append(self.current_episode_reward_sum); self.episode_count += 1
            try:
                self.current_base_state = self.env.reset();
                if not is_safe(self.current_base_state) or self.current_base_state.shape[0] != Config.Agent.BASE_STATE_DIM: raise RuntimeError("Reset invalid base state.")
                self.last_text_embedding.zero_(); self.current_state = self._get_combined_state(self.current_base_state, "episode_reset")
                if not is_safe(self.current_state) or self.current_state.shape[0] != Config.Agent.STATE_DIM: raise RuntimeError("Reset invalid combined state.")
                self.model.prev_emotions.zero_(); self.last_response_emotions.zero_(); self.mood.fill_(0.3); self.conversation_history.clear(); self.episode_steps = 0; self.current_episode_reward_sum = 0.0; self.current_event = self.env.current_event_type;
            except Exception as e:
                 logger.critical(f"CRITICAL: Failed episode reset: {e}. Stopping."); done = True;
                 default_metrics = {'error': True, 'message':"FATAL EPISODE RESET", "last_monologue":"", "embedding_norm": 0.0, "base_state_norm": 0.0}
                 return (default_metrics, reward, done, "FATAL RESET", self.last_reported_loss, "", "idle")
        metrics_dict = self.reflect()
        return metrics_dict, reward, done, self.current_response, self.last_reported_loss, self.last_internal_monologue, predicted_hm_label

    def handle_user_chat(self, user_text: str) -> str:
        logger.info(f"Handling user chat: '{user_text[:50]}...'")
        if not isinstance(user_text, str) or not user_text.strip(): return "..."
        user_text_embedding = torch.zeros_like(self.last_text_embedding)
        if Config.Agent.USE_LANGUAGE_EMBEDDING and self.st_model:
            try:
                with torch.no_grad(): emb = self.st_model.encode([user_text], convert_to_tensor=True, device=DEVICE);
                if emb.ndim > 1: emb = emb.squeeze(0)
                if emb.shape[0] == Config.Agent.LANGUAGE_EMBEDDING_DIM: user_text_embedding = emb.float()
                else: logger.warning(f"User text embed dim mismatch! Got {emb.shape}")
            except Exception as e: logger.error(f"Error embedding user text: {e}")
        self.last_text_embedding = user_text_embedding.clone().detach()
        self.conversation_history.append(("User", user_text))
        try:
            impact_vector = self.env.get_emotional_impact_from_text(user_text)
            current_response_emotions = self.last_response_emotions.clone().detach()
            mood_influence = 0.15
            biased_current_emotions = torch.clamp( current_response_emotions * (1.0 - mood_influence) + self.mood * mood_influence, 0.0, 1.0 )
            current_max_emo = biased_current_emotions.max().item(); blend_factor_user = max(0.4, min(0.7, 0.7 - (current_max_emo * 0.3))); blend_factor_current = 1.0 - blend_factor_user
            blended_emotions = torch.clamp( biased_current_emotions * blend_factor_current + impact_vector * blend_factor_user, 0.0, 1.0 )
            if not is_safe(blended_emotions): logger.warning("Blended chat emotions unsafe."); blended_emotions = biased_current_emotions
            self.last_response_emotions = blended_emotions.clone().detach()
            self.model.prev_emotions = torch.clamp(self.model.prev_emotions * 0.5 + self.last_response_emotions * 0.5, 0.0, 1.0)
            if self.avatar: self.avatar.update_emotions(self.last_response_emotions)
            context_for_gpt = ""; history_turns = list(self.conversation_history)
            for speaker, text in reversed(history_turns):
                turn = f"{speaker}: {text}\n"
                 # Error L263: break outside loop fixed
                if len(context_for_gpt) + len(turn) > 512:
                    break # Correct indentation
                context_for_gpt = turn + context_for_gpt # Correct indentation
            temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7); top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9)
            response = self.model.gpt.generate(context=context_for_gpt, emotions=self.last_response_emotions, temperature=temp, top_p=top_p)
            self.current_response = response
            self.conversation_history.append(("AI", self.current_response))
            predicted_chat_hm_label = "idle"
            if self.current_state is not None and is_safe(self.current_state) and self.current_state.shape[0] == Config.Agent.STATE_DIM:
                # Error L345: Try needs except/finally; Error L347: Expected expression
                try: # Added try
                    self.model.eval()
                    with torch.no_grad():
                         outputs = self.model.forward(self.current_state, 0.0, self.model.state_history)
                         if len(outputs) == 13:
                              hm_logits = outputs[-1]
                              if hm_logits.ndim == 1: idx = torch.argmax(hm_logits).item()
                              elif hm_logits.ndim == 2 and hm_logits.shape[0] == 1 : idx = torch.argmax(hm_logits.squeeze(0)).item()
                              else: logger.warning(f"Unexpected hm_logits shape in chat: {hm_logits.shape}"); idx = HEAD_MOVEMENT_TO_IDX["idle"]
                              predicted_chat_hm_label = IDX_TO_HEAD_MOVEMENT.get(idx, "idle")
                         else: logger.warning(f"Forward mismatch during chat HM prediction ({len(outputs)}).")
                    self.model.train()
                except Exception as e: # Added except
                     logger.error(f"Error predicting head movement after chat: {e}")
            else: logger.warning("Skipping chat HM prediction due to invalid current_state.")
            if self.avatar and hasattr(self.avatar, 'update_predicted_movement'): self.avatar.update_predicted_movement(predicted_chat_hm_label)
            logger.debug(f"Chat response: '{response}' | Emotions: {self.last_response_emotions.cpu().numpy().round(2)} | Head Move: {predicted_chat_hm_label}")
            return self.current_response
        except Exception as e: logger.error(f"Error handling user chat: {e}", exc_info=True); self.current_response = "Sorry, I had trouble processing that."; return self.current_response

    def reflect(self) -> ReflectReturnType:
        stats_dict: ReflectReturnType = { "avg_reward_last20": 0.0, "total_steps": self.total_steps, "episode": self.episode_count, "current_emotions_internal": [0.0]*Config.Agent.EMOTION_DIM, "current_emotions_response": [0.0]*Config.Agent.EMOTION_DIM, "current_mood": [0.0]*Config.Agent.EMOTION_DIM, "last_monologue": self.last_internal_monologue, "I_S": 0.0, "rho_struct": 0.0, "att_score": 0.0, "self_consistency": 0.0, "rho_score": 0.0, "box_score": 0.0, "tau_t": 0.0, "R_acc": 0.0, "loss": self.last_reported_loss, "rho_struct_mem_short": 0.0, "rho_struct_mem_long": 0.0, "embedding_norm": 0.0, "base_state_norm": 0.0 }
        recent_rewards = self.episode_rewards[-20:];
        if recent_rewards: stats_dict["avg_reward_last20"] = sum(recent_rewards) / len(recent_rewards)
        try:
            current_combined_state = self.current_state
            if current_combined_state is not None and is_safe(current_combined_state) and current_combined_state.shape[0] == Config.Agent.STATE_DIM:
                self.model.eval();
                with torch.no_grad(): reflect_outputs = self.model.forward(current_combined_state, 0.0, self.model.state_history)
                if len(reflect_outputs) == 13:
                     (emotions_int, _, _, _value, I_S, _, att_score, self_consistency, rho_score, box_score, R_acc_mean, tau_t, _) = reflect_outputs;
                     stats_dict.update({ "current_emotions_internal": emotions_int.cpu().tolist() if is_safe(emotions_int) else [0.0]*Config.Agent.EMOTION_DIM, "I_S": I_S, "att_score": att_score, "self_consistency": self_consistency, "rho_score": rho_score, "box_score": box_score, "tau_t": tau_t, "R_acc": R_acc_mean })
                else: logger.warning(f"Reflection: forward returned {len(reflect_outputs)} values, expected 13.")
                self.model.train();
            else: logger.warning(f"Reflection skipped: Invalid current_state.")
        except Exception as e: logger.error(f"Error reflect agent forward pass: {e}", exc_info=False);
        try:
             stats_dict["rho_struct_mem_short"] = self.model.memory.get_short_term_norm(); stats_dict["rho_struct_mem_long"] = self.model.memory.get_long_term_norm(); stats_dict["rho_struct"] = stats_dict["rho_struct_mem_short"] * 0.3 + stats_dict["rho_struct_mem_long"] * 0.7
             stats_dict["current_emotions_response"] = self.last_response_emotions.cpu().tolist(); stats_dict["current_mood"] = self.mood.cpu().tolist() if is_safe(self.mood) else [0.0]*Config.Agent.EMOTION_DIM
             if Config.Agent.USE_LANGUAGE_EMBEDDING and self.last_text_embedding is not None and is_safe(self.last_text_embedding): stats_dict["embedding_norm"] = torch.linalg.norm(self.last_text_embedding.float()).item()
             if self.current_base_state is not None and is_safe(self.current_base_state): stats_dict["base_state_norm"] = torch.linalg.norm(self.current_base_state.float()).item()
        except Exception as e: logger.error(f"Error get memory/norms/emo/mood/embed norm reflect: {e}", exc_info=False)
        return stats_dict

    def test_completeness(self) -> Tuple[bool, str]:
        logger.info("Performing Completeness Test..."); current_combined_state = self.current_state
        if current_combined_state is None or not is_safe(current_combined_state) or current_combined_state.shape[0] != Config.Agent.STATE_DIM: return False, "Invalid state for test"
        test_state = current_combined_state.clone().detach();
        if Config.Agent.EMOTION_DIM >= 2: test_state[0] = 0.9; test_state[1] = 0.1;
        elif Config.Agent.EMOTION_DIM == 1: test_state[0] = 0.9
        test_reward = 1.5; consistent = False; details = "Prerequisites failed."
        try:
            self.model.eval();
            with torch.no_grad(): test_outputs = self.model.forward(test_state, test_reward, self.model.state_history)
            if len(test_outputs) == 13:
                 (emotions, _, _, _, _, _, att_score, _, rho_score, box_score, R_acc, _, _) = test_outputs
                 joy_check = False; joy_val = -1.0
                 if Config.Agent.EMOTION_DIM >= 1 and is_safe(emotions): joy_val = emotions[0].item(); joy_check = joy_val > 0.5 and joy_val >= emotions.max().item() - 1e-6
                 att_check = att_score > Config.Agent.ATTENTION_THRESHOLD; box_check = box_score > 0; consistent = joy_check and att_check and box_check
                 details = (f"Joy={joy_val:.2f}(>{0.5}&max? {joy_check}), Att={att_score:.2f}(>{Config.Agent.ATTENTION_THRESHOLD}? {att_check}), Box={box_score:.2f}(>0? {box_check}), R_acc={R_acc:.2f}, RhoScr={rho_score:.2f}")
            else: details = f"Forward mismatch ({len(test_outputs)} items)."
            self.model.train();
        except Exception as e: logger.error(f"Error completeness test: {e}"); details = f"Exception: {e}"
        logger.info(f"Completeness Test Result: {consistent}. Details: {details}"); return consistent, details

    def update_environment(self, event_freq: float, intensities: List[float]): 
        try: self.env.update_params(event_freq, intensities); 
        except Exception as e: logger.error(f"Error updating env params: {e}")
    
    def cleanup(self):
        logger.info("--- Orchestrator Cleanup Initiated ---")
        logger.info("Shutting down learn executor...")
        try:
            self.learn_executor.shutdown(wait=True, cancel_futures=True) # Removed semicolon
            logger.info("Learn executor shut down.")
        except Exception as e:
             logger.error(f"Error shutting down learn executor: {e}", exc_info=True)

        if self.avatar and hasattr(self.avatar, 'cleanup'):
            try:
                logger.info("Cleaning up avatar...")
                self.avatar.cleanup() # Removed semicolon
                logger.info("Avatar cleanup complete.")
            except Exception as e: # Correctly indented except
                logger.error(f"Error avatar cleanup: {e}", exc_info=True)

        if self.st_model:
            try:
                logger.info("Releasing ST model...")
                del self.st_model
                self.st_model = None # Removed semicolon
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache() # Removed semicolon
                    logger.debug("Cleared CUDA cache.")
            except Exception as e: # Correctly indented except
                 logger.error(f"Error releasing ST model: {e}")
        # Final log message correctly indented under the main function
        logger.info("--- Orchestrator Cleanup Finished ---")

# --- END OF FILE orchestrator.py ---
