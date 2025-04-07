# --- START OF FILE agent.py ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Dict, Optional, List, Union, Deque, Any

# Use MasterConfig object and tokenizer variables
from config import MasterConfig as Config
# Use TRAIN_DATA loaded in config.py
from config import DEVICE, logger, TRAIN_DATA, tokenizer, tokenize, detokenize, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID
# --- Import Head Movement Labels ---
from config import HEAD_MOVEMENT_LABELS, NUM_HEAD_MOVEMENTS, IDX_TO_HEAD_MOVEMENT, HEAD_MOVEMENT_TO_IDX
# ---
from utils import MetronicLattice, MetaCognitiveMemory, is_safe, Experience
from ai_modules import EmotionalModule, SyntrixKorporator, StrukturKaskade, SimpleGPT


class ConsciousAgent(nn.Module):
    # Error L26: Removed semicolon and fixed indentation for the whole method
    def __init__(self, state_dim: int = Config.Agent.STATE_DIM, hidden_dim: int = Config.Agent.HIDDEN_DIM, vocab_size: int = Config.NLP.VOCAB_SIZE):
        super().__init__() # Corrected indentation
        self.state_dim = state_dim
        self.base_state_dim = Config.Agent.BASE_STATE_DIM
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        logger.info(f"Agent Init: Combined State Dim={state_dim}, Base State Dim={self.base_state_dim}, Hidden Dim={hidden_dim}, Vocab Size={vocab_size}")
        if self.base_state_dim > self.state_dim: raise ValueError("AgentConfig BASE_STATE_DIM cannot be larger than STATE_DIM")

        self.lattice = MetronicLattice(dim=self.state_dim, tau=Config.Agent.TAU) # Corrected indentation
        self.korporator = SyntrixKorporator(self.state_dim, self.hidden_dim, m=min(8, hidden_dim // 2) if hidden_dim >= 2 else 1)
        self.kaskade = StrukturKaskade(self.hidden_dim, self.hidden_dim, levels=Config.Agent.CASCADE_LEVELS)
        kaskade_out_dim = self.kaskade._output_dim
        self.emotional_module = EmotionalModule(input_dim=Config.Agent.EMOTION_DIM + 1)
        self.attention: Optional[nn.MultiheadAttention] = None
        if self.state_dim > 0:
            possible_heads=[h for h in [16, 12, 8, 6, 4, 2, 1] if self.state_dim % h == 0]; num_heads = possible_heads[0] if possible_heads else 1
            if not possible_heads: logger.warning(f"Agent Attention fallback 1 head dim {self.state_dim}.")
            # Error L56: Try needs except/finally; Error L57: Indentation; Error L59: Expected expression
            try: # Corrected indentation
                self.attention=nn.MultiheadAttention(embed_dim=self.state_dim, num_heads=num_heads, batch_first=True, dropout=0.15)
                logger.info(f"Agent Attention init {num_heads} heads dim {self.state_dim}.")
            except Exception as e: # Added except block
                logger.error(f"Failed init Attention: {e}. Disabled.")
                self.attention=None
        self.feedback = nn.Linear(kaskade_out_dim, self.state_dim)
        self.value_head = nn.Linear(kaskade_out_dim, 1)
        self.head_movement_head = nn.Linear(kaskade_out_dim, NUM_HEAD_MOVEMENTS)
        logger.info(f"Initialized Head Movement prediction head ({kaskade_out_dim} -> {NUM_HEAD_MOVEMENTS})")
        self.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE)
        self.gpt = SimpleGPT(vocab_size=Config.NLP.VOCAB_SIZE, embed_dim=64, hidden_dim=128, num_heads=4)
        agent_params = list(self.korporator.parameters()) + list(self.kaskade.parameters()) + \
                       list(self.feedback.parameters()) + list(self.emotional_module.parameters()) + \
                       list(self.value_head.parameters()) + list(self.head_movement_head.parameters())
        if self.attention: agent_params.extend(list(self.attention.parameters()))
        self.optimizer = optim.Adam(agent_params, lr=Config.RL.LR)
        self._base_lr = Config.RL.LR
        self.state_history_deque: Deque[torch.Tensor] = deque(maxlen=Config.Agent.HISTORY_SIZE)
        # Corrected list comprehension indentation
        for _ in range(Config.Agent.HISTORY_SIZE):
            self.state_history_deque.append(torch.zeros(self.state_dim, device=DEVICE))
        self.prev_emotions = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
        self.step_count = 0
        self.beta: float = Config.RL.PER_BETA_START
        beta_frames: int = Config.RL.PER_BETA_FRAMES
        self.beta_increment: float = (1.0 - self.beta) / beta_frames if beta_frames > 0 else 0.0
        if TRAIN_DATA and tokenizer is not None:
            try:
                logger.info("Starting initial GPT training...")
                self.gpt.train_model(TRAIN_DATA, epochs=Config.NLP.TRAIN_EPOCHS)
                logger.info("Initial GPT training complete.")
            except Exception as e:
                logger.error(f"Error initial GPT training: {e}", exc_info=True)
        elif tokenizer is None:
            logger.warning("Skipping initial GPT training: Tokenizer not initialized.")
        else:
            logger.info("Skipping initial GPT training: No valid training data.")
        logger.info(f"ConsciousAgent initialized with STATE_DIM={self.state_dim}.")

    # Error L96: Indentation; Error L98: Expected expression
    @property
    def state_history(self) -> torch.Tensor:
        if not self.state_history_deque:
            return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE, dtype=torch.float32)
        valid_elements = True
        for t in self.state_history_deque:
             if not isinstance(t, torch.Tensor) or t.shape != (self.state_dim,):
                  logger.error(f"Agent history invalid shape {t.shape}, expected ({self.state_dim},).")
                  valid_elements = False
                  break # Added break
        if not valid_elements:
             self.state_history_deque.clear()
             for _ in range(Config.Agent.HISTORY_SIZE): # Use loop instead of list comp
                 self.state_history_deque.append(torch.zeros(self.state_dim, device=DEVICE))
             return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE)
        try:
            return torch.stack(list(self.state_history_deque)).to(device=DEVICE, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error stacking agent history: {e}.")
            return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE)

    # ... (compute_accessibility, forward, _get_default_outputs, step, learn methods remain structurally the same as the last correct version) ...
    # Make sure no stray semicolons exist within these methods from copy-paste.
    # The Pylance errors didn't point inside these methods for the last round.
    def compute_accessibility(self, history_tensor: torch.Tensor) -> torch.Tensor:
        default_matrix = torch.zeros((Config.Agent.HISTORY_SIZE, Config.Agent.HISTORY_SIZE), device=DEVICE)
        if not isinstance(history_tensor, torch.Tensor) or history_tensor.shape != (Config.Agent.HISTORY_SIZE, self.state_dim):
            logger.warning(f"Accessibility input shape mismatch: Got {history_tensor.shape}, Expected ({Config.Agent.HISTORY_SIZE}, {self.state_dim})")
            return default_matrix
        if not is_safe(history_tensor):
            logger.warning("Accessibility skipped: Unsafe history.")
            return default_matrix
        try:
            history_float = history_tensor.float()
            norms = torch.linalg.norm(history_float, dim=1, keepdim=True) + 1e-8
            normalized_history = history_float / norms
            similarity_matrix = torch.matmul(normalized_history, normalized_history.t())
            similarity_matrix = (similarity_matrix + 1.0) / 2.0 # Scale to [0, 1]
            similarity_matrix = torch.clamp(similarity_matrix, 0.0, 1.0)
            accessibility_matrix = torch.where(
                similarity_matrix > Config.Agent.ACCESSIBILITY_THRESHOLD,
                similarity_matrix,
                torch.zeros_like(similarity_matrix)
            )
            if not is_safe(accessibility_matrix):
                logger.warning("Calculated accessibility matrix unsafe.")
                return default_matrix
            return accessibility_matrix
        except Exception as e:
            logger.error(f"Error accessibility calc: {e}", exc_info=True)
            return default_matrix
    ForwardReturnType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float, float, float, float, float, torch.Tensor]

    def forward(self, state_input: torch.Tensor, current_reward_input: Union[float, torch.Tensor], full_state_history_input: Optional[torch.Tensor]) -> ForwardReturnType:
        is_batch = state_input.ndim == 2; batch_size = state_input.shape[0] if is_batch else 1
        if not isinstance(state_input, torch.Tensor) or state_input.shape[-1] != self.state_dim: logger.error(f"Agent.forward: Invalid state type/shape. Expected (*, {self.state_dim}), got {state_input.shape}."); return self._get_default_outputs(batch_size)
        state = state_input.to(DEVICE);
        if not is_safe(state): logger.warning("Agent.forward: Unsafe state input. Using zeros."); state = torch.zeros_like(state_input)
        if is_batch:
            if isinstance(current_reward_input, torch.Tensor):
                if current_reward_input.shape == (batch_size, 1): current_reward = current_reward_input.to(DEVICE).float()
                elif current_reward_input.shape == (batch_size,): current_reward = current_reward_input.to(DEVICE).float().unsqueeze(1)
                else: logger.warning(f"Agent.forward batch: Reward shape mismatch {current_reward_input.shape}. Using zeros."); current_reward = torch.zeros(batch_size, 1, device=DEVICE, dtype=torch.float32)
            else:
                 try: rewards_float = [float(r) for r in current_reward_input]; current_reward = torch.tensor(rewards_float, device=DEVICE, dtype=torch.float32).unsqueeze(1); assert current_reward.shape[0] == batch_size # type: ignore
                 except: logger.warning(f"Agent.forward batch: Invalid non-tensor reward input type {type(current_reward_input)}. Using zeros."); current_reward = torch.zeros(batch_size, 1, device=DEVICE, dtype=torch.float32)
        else:
            try: current_reward_float = float(current_reward_input); current_reward = torch.tensor([[current_reward_float]], device=DEVICE, dtype=torch.float32)
            except: logger.warning(f"Agent.forward single: Invalid reward input type {type(current_reward_input)}. Using zero."); current_reward = torch.tensor([[0.0]], device=DEVICE, dtype=torch.float32)
        if not is_safe(current_reward): logger.warning("Agent.forward: Unsafe reward. Using zeros."); current_reward = torch.zeros_like(current_reward).float()
        discretized_state = self.lattice.discretize(state.clone().detach());
        if not is_safe(discretized_state): discretized_state = torch.zeros_like(state)
        history_to_use = None; attn_output_context = discretized_state.clone(); attn_weights = None; att_score = 0.0
        if not is_batch:
            history_to_use = self.state_history
            if isinstance(full_state_history_input, torch.Tensor) and full_state_history_input.shape == (Config.Agent.HISTORY_SIZE, self.state_dim) and is_safe(full_state_history_input): history_to_use = full_state_history_input.to(DEVICE)
            elif full_state_history_input is not None: logger.warning("Agent.forward single: Invalid external history.")
            if self.attention and history_to_use is not None and history_to_use.shape[0] == Config.Agent.HISTORY_SIZE:
                 state_seq = history_to_use.unsqueeze(0).float().detach()
                                  # (Inside the forward method, within the try block for attention)
                 try:
                     attn_output_b, attn_weights_b = self.attention(state_seq, state_seq, state_seq) # Removed semicolon
                     if is_safe(attn_output_b) and attn_output_b.shape == (1, Config.Agent.HISTORY_SIZE, self.state_dim):
                         attn_output_context = attn_output_b[0, -1, :].detach()
                     else:
                         logger.warning("Unsafe/invalid shape attn output.")
                         attn_output_context = discretized_state[-1] if discretized_state.ndim == 2 else discretized_state
                     if attn_output_context.shape != (self.state_dim,):
                         logger.warning(f"Attn context shape mismatch.")
                         attn_output_context = discretized_state[-1] if discretized_state.ndim == 2 else discretized_state
                     attn_weights = attn_weights_b.squeeze(0)
                     if is_safe(attn_weights):
                         non_diag_mask = ~torch.eye(Config.Agent.HISTORY_SIZE, dtype=torch.bool, device=DEVICE)
                         valid_weights = attn_weights[non_diag_mask] # Removed semicolon
                         if valid_weights.numel() > 0:
                             att_score = valid_weights.mean().item()
                         # Removed else: pass as it's redundant
                     else:
                         logger.warning("Unsafe attention weights.")
                 except Exception as e: # Moved except block to match the try
                    logger.error(f"Error self-attention: {e}")
                    attn_output_context = discretized_state[-1] if discretized_state.ndim == 2 else discretized_state
            elif not is_batch and self.attention and history_to_use is not None: logger.warning(f"Attn skipped: History size mismatch.")
        elif is_batch and self.attention: attn_output_context = discretized_state; pass
        if is_batch: emotion_state_part = discretized_state[:, :Config.Agent.EMOTION_DIM]; prev_emotions_for_module = self.prev_emotions.unsqueeze(0).repeat(batch_size, 1)
        else: emotion_state_part = discretized_state[:Config.Agent.EMOTION_DIM].unsqueeze(0); prev_emotions_for_module = self.prev_emotions.unsqueeze(0)
        if not is_safe(prev_emotions_for_module): prev_emotions_for_module = torch.zeros_like(prev_emotions_for_module)
        current_emotions_batch = self.emotional_module(emotion_state_part, current_reward, prev_emotions_for_module)
        if not is_batch: current_emotions = current_emotions_batch.squeeze(0); self.prev_emotions = current_emotions.detach().clone()
        else: current_emotions = current_emotions_batch
        if not is_safe(current_emotions): logger.warning("Unsafe emotions from module."); current_emotions = torch.zeros_like(current_emotions)
        psi_input = attn_output_context
        belief_raw = self.korporator(discretized_state, psi_input, level=2); belief = self.kaskade(belief_raw);
        if not is_safe(belief): belief = torch.zeros_like(belief_raw)
        value = self.value_head(belief);
        if not is_safe(value): value = torch.zeros_like(value)
        feedback_signal = self.feedback(belief);
        if not is_safe(feedback_signal): feedback_signal = torch.zeros_like(feedback_signal)
        head_movement_logits = self.head_movement_head(belief)
        if not is_safe(head_movement_logits): logger.warning("Unsafe head movement logits."); hm_shape = (batch_size, NUM_HEAD_MOVEMENTS) if is_batch else (NUM_HEAD_MOVEMENTS,); head_movement_logits = torch.zeros(hm_shape, device=DEVICE)
        I_S_norm = 0.0; R_acc_mean = 0.0; tau_t = Config.Agent.ACCESSIBILITY_THRESHOLD; rho_struct_val = 0.0; box_score = 0.0;
        self_consistency_batch = torch.zeros(batch_size, 1, device=DEVICE) if is_batch else torch.tensor(0.0, device=DEVICE)
        rho_score_batch = torch.zeros(batch_size, 1, device=DEVICE) if is_batch else torch.tensor(0.0, device=DEVICE)
        if belief.shape == belief_raw.shape and belief.numel() > 0:
             belief_flat = belief.flatten(start_dim=1).float() if is_batch else belief.flatten().float(); belief_raw_flat = belief_raw.flatten(start_dim=1).float() if is_batch else belief_raw.flatten().float()
             belief_norm = torch.linalg.norm(belief_flat, dim=-1, keepdim=True) + 1e-8; belief_raw_norm = torch.linalg.norm(belief_raw_flat, dim=-1, keepdim=True) + 1e-8
             cosine_sim = F.cosine_similarity(belief_flat / belief_norm, belief_raw_flat / belief_raw_norm, dim=-1).unsqueeze(-1)
             self_consistency_batch = cosine_sim.detach(); rho_score_batch = torch.clamp((self_consistency_batch + 1.0) / 2.0, 0.0, 1.0).detach()
        else: logger.warning(f"Forward: Consistency shape mismatch {belief.shape} vs {belief_raw.shape}.")
        if is_batch:
            self_consistency = self_consistency_batch.mean().item(); rho_score = rho_score_batch.mean().item()
            I_S_norm, rho_struct_val, box_score, R_acc_mean, tau_t = 0.0, 0.0, 0.0, 0.0, Config.Agent.ACCESSIBILITY_THRESHOLD;
        else:
            self_consistency = self_consistency_batch.item(); rho_score = rho_score_batch.item()
            if history_to_use is not None:
                R_accessibility = self.compute_accessibility(history_to_use.detach()); R_acc_mean = R_accessibility.mean().item() if R_accessibility.numel() > 0 else 0.0
                I_S_vector = self.lattice.S(history_to_use.detach(), 0, Config.Agent.HISTORY_SIZE - 1); I_S_norm = torch.linalg.norm(I_S_vector.float()).item() if I_S_vector.numel() > 0 else 0.0
                tau_t = Config.Agent.ACCESSIBILITY_THRESHOLD * (1 + att_score * 0.5)
                rho_struct_mem_short = self.memory.get_short_term_norm(); rho_struct_mem_long = self.memory.get_long_term_norm(); rho_struct_val = (rho_struct_mem_short * 0.3 + rho_struct_mem_long * 0.7)
                emotion_max_val = current_emotions.max().item() if current_emotions.numel() > 0 else 0.0; is_stable = emotion_max_val < Config.Agent.STABILITY_THRESHOLD; box_score = R_acc_mean if is_stable else 0.0;
        metrics_float = tuple(float(m) for m in [I_S_norm, rho_struct_val, att_score, self_consistency, rho_score, box_score, R_acc_mean, tau_t])
        return (current_emotions, belief, feedback_signal, value, *metrics_float, head_movement_logits)

    def _get_default_outputs(self, batch_size=1) -> ForwardReturnType:
         zero_emo = torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE) if batch_size > 1 else torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
         kaskade_out_dim = getattr(self.kaskade, '_output_dim', self.hidden_dim); zero_belief = torch.zeros(batch_size, kaskade_out_dim, device=DEVICE) if batch_size > 1 else torch.zeros(kaskade_out_dim, device=DEVICE)
         zero_feedback = torch.zeros(batch_size, self.state_dim, device=DEVICE) if batch_size > 1 else torch.zeros(self.state_dim, device=DEVICE)
         zero_value = torch.zeros(batch_size, 1, device=DEVICE) if batch_size > 1 else torch.zeros(1, device=DEVICE)
         zero_metrics = (0.0,) * 8
         zero_hm_logits = torch.zeros(batch_size, NUM_HEAD_MOVEMENTS, device=DEVICE) if batch_size > 1 else torch.zeros(NUM_HEAD_MOVEMENTS, device=DEVICE)
         return (zero_emo, zero_belief, zero_feedback, zero_value, *zero_metrics, zero_hm_logits)

    StepReturnType = Tuple[torch.Tensor, str, torch.Tensor, float, str]

    def step(self, state: torch.Tensor, reward: float, state_history: torch.Tensor, context: Optional[str]) -> StepReturnType:
        self.eval()
        predicted_hm_label = "idle"
        with torch.no_grad():
             if not isinstance(state, torch.Tensor) or not is_safe(state) or state.shape != (self.state_dim,): logger.warning(f"Agent step invalid state {state.shape}. Exp ({self.state_dim},). Zeros."); state = torch.zeros(self.state_dim, device=DEVICE)
             if state.device != DEVICE: state = state.to(DEVICE)
             forward_outputs = self.forward(state, reward, state_history)
             current_emotions = forward_outputs[0]; belief_for_memory = forward_outputs[1]; att_score_metric = forward_outputs[6]; head_movement_logits = forward_outputs[-1]
             try:
                 if head_movement_logits.ndim == 1 and head_movement_logits.shape[0] == NUM_HEAD_MOVEMENTS: predicted_hm_idx = torch.argmax(head_movement_logits).item(); predicted_hm_label = IDX_TO_HEAD_MOVEMENT.get(predicted_hm_idx, "idle")
                 elif head_movement_logits.ndim == 2 and head_movement_logits.shape[0] == 1 and head_movement_logits.shape[1] == NUM_HEAD_MOVEMENTS: predicted_hm_idx = torch.argmax(head_movement_logits.squeeze(0)).item(); predicted_hm_label = IDX_TO_HEAD_MOVEMENT.get(predicted_hm_idx, "idle")
                 else: logger.warning(f"Unexpected HM logits shape: {head_movement_logits.shape}. Default 'idle'."); predicted_hm_label = "idle"
             except Exception as e: logger.error(f"Error predicting head movement: {e}"); predicted_hm_label = "idle"
             try: response_context = context if context else "..."; temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7); top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9); response = self.gpt.generate(response_context, current_emotions, temperature=temp, top_p=top_p)
             except Exception as e: logger.error(f"Error GPT gen step: {e}"); response = "..."
             discretized_state = self.lattice.discretize(state); self.state_history_deque.append(discretized_state.clone())
        return current_emotions, response, belief_for_memory, att_score_metric, predicted_hm_label

    def learn(self, batch_size: int = Config.RL.AGENT_BATCH_SIZE) -> float:
        self.train()
        self.beta = min(1.0, self.beta + self.beta_increment)
        if len(self.memory) < batch_size: return 0.0
        sample_result = self.memory.sample(batch_size, beta=self.beta)
        if sample_result is None: logger.warning("Learn: Memory sample returned None."); return 0.0
        batch_data, indices, weights = sample_result
        states = batch_data['states']; rewards = batch_data['rewards']; next_states = batch_data['next_states']; dones = batch_data['dones']
        target_hm_indices = batch_data.get('target_hm_idx')
        if states.shape[0] != rewards.shape[0] or states.shape[0] != next_states.shape[0] or states.shape[1] != self.state_dim: logger.error(f"Learn: Batch dim/state mismatch. States: {states.shape}, Exp Dim: {self.state_dim}"); return -1.0
        current_batch_size = states.shape[0];
        if current_batch_size == 0: logger.warning("Learn: Batch size 0."); return 0.0
        try:
            zero_rewards_batch = torch.zeros_like(rewards)
            outputs = self.forward(states, zero_rewards_batch, None)
            current_value_pred = outputs[3]; rho_score_batch = outputs[8]; box_score_batch = outputs[9]; head_movement_logits = outputs[-1]
            if not is_safe(current_value_pred) or current_value_pred.shape != (current_batch_size, 1): raise ValueError(f"Invalid V(s) shape/safety {current_value_pred.shape}")
            if isinstance(rho_score_batch, float): rho_score_tensor = torch.full((current_batch_size, 1), rho_score_batch, device=DEVICE)
            elif isinstance(rho_score_batch, torch.Tensor): rho_score_tensor = rho_score_batch.detach().clone()
            else: raise ValueError("Invalid rho_score type")
            if rho_score_tensor.ndim == 1: rho_score_tensor = rho_score_tensor.unsqueeze(1)
            if rho_score_tensor.shape != (current_batch_size, 1): raise ValueError(f"Rho score tensor shape mismatch: {rho_score_tensor.shape}")
            if not is_safe(rho_score_tensor): raise ValueError("Unsafe rho_score_tensor")
            if isinstance(box_score_batch, float): box_score_tensor = torch.full((current_batch_size, 1), box_score_batch, device=DEVICE)
            elif isinstance(box_score_batch, torch.Tensor): box_score_tensor = box_score_batch.detach().clone()
            else: raise ValueError("Invalid box_score type")
            if box_score_tensor.ndim == 1: box_score_tensor = box_score_tensor.unsqueeze(1)
            if box_score_tensor.shape != (current_batch_size, 1): raise ValueError(f"Box score tensor shape mismatch: {box_score_tensor.shape}")
            if not is_safe(box_score_tensor): raise ValueError("Unsafe box_score_tensor")
            if not is_safe(head_movement_logits) or head_movement_logits.shape != (current_batch_size, NUM_HEAD_MOVEMENTS): raise ValueError(f"Invalid HM logits shape/safety. Got {head_movement_logits.shape}, Exp ({current_batch_size}, {NUM_HEAD_MOVEMENTS})")
        except Exception as e: logger.error(f"Error during V(s)/Metrics calculation in learn: {e}", exc_info=True); return -1.0
        next_value_pred = torch.zeros_like(rewards)
        with torch.no_grad():
            try:
                zero_rewards_next_batch = torch.zeros_like(rewards)
                next_outputs = self.forward(next_states, zero_rewards_next_batch, None)
                v_sp = next_outputs[3];
                if is_safe(v_sp) and v_sp.shape == (current_batch_size, 1): next_value_pred = v_sp
                else: logger.warning(f"Learn: Invalid next_value prediction shape/safety {v_sp.shape}.")
                dones_squeezed = dones.squeeze(-1) if dones.ndim > 1 else dones
                next_value_pred[dones_squeezed] = 0.0
            except Exception as e: logger.error(f"Error during V(s') calculation in learn: {e}", exc_info=True); next_value_pred = torch.zeros_like(rewards)
        intrinsic_reward_consistency = rho_score_tensor * Config.RL.INTRINSIC_REWARD_SCALE_CONSISTENCY
        intrinsic_reward_box = box_score_tensor * Config.RL.INTRINSIC_REWARD_SCALE_BOX
        td_error_abs = torch.abs(rewards + Config.RL.GAMMA * next_value_pred.detach() - current_value_pred.detach())
        intrinsic_reward_td_error = td_error_abs.clamp(max=1.0) * Config.RL.INTRINSIC_REWARD_SCALE_TD
        effective_rewards = rewards + (intrinsic_reward_consistency + intrinsic_reward_box + intrinsic_reward_td_error).detach()
        target_value = effective_rewards + Config.RL.GAMMA * next_value_pred
        td_error = target_value - current_value_pred
        value_loss = torch.tensor(0.0, device=DEVICE)
        if current_value_pred.requires_grad:
            value_loss_elementwise = F.smooth_l1_loss(current_value_pred, target_value.detach(), reduction='none', beta=1.0)
            consistency_penalty = (1.0 + (1.0 - rho_score_tensor.clamp(0,1)))
            modulated_loss = value_loss_elementwise * weights * consistency_penalty
            value_loss = modulated_loss.mean()
        else: logger.warning("Learn: current_value_pred does not require grad. Value loss is 0.")
        movement_loss = torch.tensor(0.0, device=DEVICE)
        hm_loss_weight = Config.RL.HEAD_MOVEMENT_LOSS_WEIGHT
        if hm_loss_weight > 0 and target_hm_indices is not None and head_movement_logits.requires_grad:
            try:
                target_hm_indices_dev = target_hm_indices.to(DEVICE).long()
                if target_hm_indices_dev.shape[0] == current_batch_size:
                    if target_hm_indices_dev.ndim > 1: target_hm_indices_dev = target_hm_indices_dev.squeeze(-1)
                    hm_loss_func = nn.CrossEntropyLoss(reduction='none')
                    hm_loss_elementwise = hm_loss_func(head_movement_logits, target_hm_indices_dev)
                    movement_loss = (hm_loss_elementwise * weights.squeeze(-1)).mean()
                    if not torch.isfinite(movement_loss): logger.warning(f"Movement loss is NaN/Inf! Elementwise: {hm_loss_elementwise}, Weights: {weights.squeeze(-1)}"); movement_loss = torch.tensor(0.0, device=DEVICE)
                else: logger.warning(f"Learn: Mismatch batch size ({current_batch_size}) vs target HM ({target_hm_indices.shape}). Skip HM loss.")
            except Exception as hm_err: logger.error(f"Learn: Error calculating head movement loss: {hm_err}", exc_info=True); movement_loss = torch.tensor(0.0, device=DEVICE)
        total_loss = value_loss + movement_loss * hm_loss_weight
        original_lrs = []
        dynamic_lr = self._base_lr
        if Config.RL.ADAPTIVE_LR_ENABLED and rho_score_tensor.numel() > 0:
            avg_rho_score = rho_score_tensor.mean().item()
            min_factor = Config.RL.LR_ADAPTIVE_MIN_FACTOR; max_factor = Config.RL.LR_ADAPTIVE_MAX_FACTOR
            dynamic_lr_factor = min_factor + (max_factor - min_factor) * max(0.0, min(1.0, avg_rho_score))
            dynamic_lr = self._base_lr * dynamic_lr_factor
            for i, param_group in enumerate(self.optimizer.param_groups): original_lrs.append(param_group['lr']); param_group['lr'] = dynamic_lr
        loss_val = total_loss.item()
        if is_safe(total_loss) and total_loss.requires_grad and loss_val > 1e-9:
             self.optimizer.zero_grad()
             total_loss.backward()
             agent_params_with_grad = [p for pg in self.optimizer.param_groups for p in pg['params'] if p.grad is not None]
             if agent_params_with_grad: torch.nn.utils.clip_grad_norm_(agent_params_with_grad, max_norm=Config.RL.GRADIENT_CLIP_AGENT)
             else: logger.warning("Learn: No gradients found after backward pass.")
             self.optimizer.step();
             self.memory.update_priorities(indices, td_error.detach())
        elif not total_loss.requires_grad and abs(loss_val) > 1e-7 : logger.debug(f"Learn: Total Loss ({loss_val:.4f}) requires no grad.")
        elif not is_safe(total_loss): logger.warning(f"Learn: Unsafe total loss ({loss_val:.4f}). Skip step."); self.optimizer.zero_grad()
        else: pass
        if Config.RL.ADAPTIVE_LR_ENABLED and original_lrs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i < len(original_lrs): param_group['lr'] = original_lrs[i]
        return loss_val

# --- END OF FILE agent.py ---
