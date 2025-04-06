# --- START OF FILE agent.py ---
# (Imports and __init__ as previously corrected)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Dict, Optional

from config import Config, DEVICE, logger, TRAIN_DATA
from utils import MetronicLattice, MetaCognitiveMemory, is_safe, Experience
from ai_modules import EmotionalModule, SyntrixKorporator, StrukturKaskade, SimpleGPT

class ConsciousAgent(nn.Module):
    # ... (__init__ including value_head, PER beta vars) ...
    def __init__(self, state_dim=Config.STATE_DIM, hidden_dim=Config.HIDDEN_DIM, vocab_size=Config.VOCAB_SIZE):
        super().__init__(); self.state_dim = state_dim; self.hidden_dim = hidden_dim; self.vocab_size = vocab_size
        self.lattice = MetronicLattice(dim=state_dim); self.korporator = SyntrixKorporator(self.state_dim, self.hidden_dim, m=min(6, hidden_dim // 2) if hidden_dim >= 2 else 1); self.kaskade = StrukturKaskade(hidden_dim, hidden_dim, levels=Config.CASCADE_LEVELS); kaskade_out_dim = self.kaskade._output_dim; self.emotional_module = EmotionalModule()
        self.attention = None;
        if state_dim > 0: possible_heads=[h for h in [8, 4, 2, 1] if state_dim % h == 0]; num_heads = possible_heads[0] if possible_heads else 1;
        if not possible_heads: logger.warning(f"Agent Attention using fallback 1 head.");
        try: self.attention=nn.MultiheadAttention(embed_dim=state_dim, num_heads=num_heads, batch_first=True, dropout=0.1); logger.info(f"Agent Attention init {num_heads} heads.")
        except Exception as e: logger.error(f"Failed init Attention: {e}. Disabled."); self.attention=None
        else: logger.error("Agent Attention disabled: state_dim <= 0.")
        self.feedback = nn.Linear(kaskade_out_dim, state_dim); self.value_head = nn.Linear(kaskade_out_dim, 1)
        self.memory = MetaCognitiveMemory()
        self.gpt = SimpleGPT(vocab_size=vocab_size, embed_dim=64, hidden_dim=128, num_heads=4)
        agent_params = list(self.korporator.parameters()) + list(self.kaskade.parameters()) + list(self.feedback.parameters()) + list(self.emotional_module.parameters()) + list(self.value_head.parameters())
        if self.attention: agent_params.extend(list(self.attention.parameters()))
        self.optimizer = optim.Adam(agent_params, lr=Config.LR)
        self.state_history_deque = deque(maxlen=Config.HISTORY_SIZE); [self.state_history_deque.append(torch.zeros(Config.STATE_DIM, device=DEVICE)) for _ in range(Config.HISTORY_SIZE)]; self.prev_emotions = torch.zeros(Config.EMOTION_DIM, device=DEVICE); self.step_count = 0
        self.beta = getattr(Config, 'PER_BETA_START', 0.4); beta_frames = getattr(Config, 'PER_BETA_FRAMES', 100000); self.beta_increment = (1.0 - self.beta) / beta_frames if beta_frames > 0 else 0
        if TRAIN_DATA: 
            try: logger.info("Starting initial GPT training..."); self.gpt.train_model(TRAIN_DATA, epochs=Config.TRAIN_EPOCHS); logger.info("Initial GPT training complete.") 
            except Exception as e: logger.error(f"Error initial GPT training: {e}", exc_info=True)
        else: logger.info("Skipping initial GPT training.")
        logger.info("ConsciousAgent initialized.")


    # ... (state_history, compute_accessibility, forward, _get_default_outputs, step remain the same as previous correction) ...
    @property
    def state_history(self):
        if not self.state_history_deque: return torch.zeros(Config.HISTORY_SIZE, Config.STATE_DIM, device=DEVICE, dtype=torch.float32)
        valid_elements = True;
        for t in self.state_history_deque:
             if not isinstance(t, torch.Tensor) or t.shape != (Config.STATE_DIM,): logger.error(f"Agent history invalid."); valid_elements = False; break
        if not valid_elements:
             self.state_history_deque.clear();
             for _ in range(Config.HISTORY_SIZE): self.state_history_deque.append(torch.zeros(Config.STATE_DIM, device=DEVICE))
             return torch.zeros(Config.HISTORY_SIZE, Config.STATE_DIM, device=DEVICE)
        try: return torch.stack(list(self.state_history_deque)).to(device=DEVICE, dtype=torch.float32)
        except Exception as e: logger.error(f"Error stacking agent history: {e}."); return torch.zeros(Config.HISTORY_SIZE, Config.STATE_DIM, device=DEVICE)

    def compute_accessibility(self, history_tensor):
        default_matrix = torch.zeros((Config.HISTORY_SIZE, Config.HISTORY_SIZE), device=DEVICE)
        if not isinstance(history_tensor, torch.Tensor) or history_tensor.shape != (Config.HISTORY_SIZE, Config.STATE_DIM): return default_matrix
        if not is_safe(history_tensor): logger.warning("Accessibility skipped: Unsafe history."); return default_matrix
        try:
            history_float = history_tensor.float(); norms = torch.linalg.norm(history_float, dim=1, keepdim=True) + 1e-8
            normalized_history = history_float / norms; similarity_matrix = torch.matmul(normalized_history, normalized_history.t())
            similarity_matrix = (similarity_matrix + 1.0) / 2.0; similarity_matrix = torch.clamp(similarity_matrix, 0.0, 1.0)
            accessibility_matrix = torch.where(similarity_matrix > Config.ACCESSIBILITY_THRESHOLD, similarity_matrix, torch.zeros_like(similarity_matrix))
            if not is_safe(accessibility_matrix): logger.warning("Calculated accessibility matrix unsafe."); return default_matrix
            return accessibility_matrix
        except Exception as e: logger.error(f"Error accessibility calc: {e}", exc_info=True); return default_matrix

    def forward(self, state_input, current_reward_input, full_state_history_input) -> Tuple:
        is_batch = state_input.ndim == 2; batch_size = state_input.shape[0] if is_batch else 1
        if not isinstance(state_input, torch.Tensor): logger.error("Agent.forward: Invalid state type."); return self._get_default_outputs(batch_size)
        state = state_input.to(DEVICE);
        if not is_safe(state): logger.warning("Agent.forward: Unsafe state input. Using zeros."); state = torch.zeros_like(state_input)
        if is_batch: current_reward = current_reward_input.to(DEVICE).float() if isinstance(current_reward_input, torch.Tensor) and current_reward_input.shape == (batch_size, 1) else torch.zeros(batch_size, 1, device=DEVICE, dtype=torch.float32)
        else: 
            try: current_reward = float(current_reward_input); 
            except: current_reward = 0.0
        if not is_safe(current_reward): current_reward = torch.zeros_like(current_reward).float() if is_batch else 0.0
        discretized_state = self.lattice.discretize(state.clone().detach());
        if not is_safe(discretized_state): discretized_state = torch.zeros_like(state)
        history_to_use = None; attn_output_context = discretized_state; att_score = 0.0
        if not is_batch:
            history_to_use = self.state_history
            if isinstance(full_state_history_input, torch.Tensor) and full_state_history_input.shape == (Config.HISTORY_SIZE, Config.STATE_DIM) and is_safe(full_state_history_input): history_to_use = full_state_history_input.to(DEVICE)
            elif full_state_history_input is not None: logger.warning("Agent.forward single: Invalid external history.")
            if self.attention and history_to_use is not None and history_to_use.shape[0] == Config.HISTORY_SIZE:
                 state_seq = history_to_use.unsqueeze(0).float().detach()
                 try:
                     attn_output_b, attn_weights_b = self.attention(state_seq, state_seq, state_seq);
                     if is_safe(attn_output_b) and attn_output_b.shape == (1, Config.HISTORY_SIZE, self.state_dim): attn_output_context = attn_output_b[0, -1, :].detach()
                     else: logger.warning("Unsafe/invalid shape attn output."); attn_output_context = discretized_state
                     if attn_output_context.shape[0] != self.state_dim: logger.warning(f"Attn context shape mismatch."); attn_output_context = discretized_state
                     attn_weights = attn_weights_b.squeeze(0)
                     if is_safe(attn_weights): non_diag_mask = ~torch.eye(Config.HISTORY_SIZE, dtype=torch.bool, device=DEVICE); valid_weights = attn_weights[non_diag_mask];
                     if valid_weights.numel() > 0: att_score = valid_weights.mean().item()
                     else: logger.warning("Unsafe attention weights.")
                 except Exception as e: logger.error(f"Error self-attention: {e}"); attn_output_context = discretized_state
            elif not is_batch and self.attention and history_to_use is not None: logger.warning(f"Attn skipped: History size mismatch.")
        if is_batch: emotion_state_part = discretized_state[:, :Config.EMOTION_DIM]; prev_emotions_for_batch = discretized_state[:, :Config.EMOTION_DIM].detach(); current_emotions = self.emotional_module(emotion_state_part, current_reward, prev_emotions_for_batch)
        else: emotion_state_part = discretized_state[:Config.EMOTION_DIM]; safe_prev_emotions = self.prev_emotions.detach().clone().unsqueeze(0); safe_prev_emotions.zero_() if not is_safe(safe_prev_emotions) else None; reward_tensor_single = torch.tensor([[current_reward]], device=DEVICE, dtype=torch.float32); current_emotions_batch = self.emotional_module(emotion_state_part.unsqueeze(0), reward_tensor_single, safe_prev_emotions); current_emotions = current_emotions_batch.squeeze(0)
        if not is_safe(current_emotions): current_emotions = torch.zeros_like(current_emotions)
        psi_input = attn_output_context
        if is_batch and psi_input.ndim == 1: psi_input = psi_input.unsqueeze(0).repeat(batch_size, 1)
        elif is_batch and psi_input.shape[0] != batch_size: logger.warning(f"Korp batch context mismatch."); psi_input = discretized_state
        elif not is_batch and psi_input.ndim == 2: logger.warning("Korp single context is batch?"); psi_input = discretized_state
        belief_raw = self.korporator(discretized_state, psi_input, level=2); belief = self.kaskade(belief_raw);
        if not is_safe(belief): belief = torch.zeros_like(belief_raw)
        value = self.value_head(belief);
        if not is_safe(value): value = torch.zeros_like(value)
        feedback_signal = self.feedback(belief);
        if not is_safe(feedback_signal): feedback_signal = torch.zeros_like(feedback_signal)
        I_S_norm = 0.0; R_acc_mean = 0.0; tau_t = Config.ACCESSIBILITY_THRESHOLD; rho_struct_val = 0.0; box_score = 0.0;
        self_consistency_batch = torch.zeros(batch_size, 1, device=DEVICE) if is_batch else torch.tensor(0.0, device=DEVICE)
        rho_score_batch = torch.zeros(batch_size, 1, device=DEVICE) if is_batch else torch.tensor(0.0, device=DEVICE)
        if belief.shape == belief_raw.shape and belief.numel() > 0:
             belief_flat = belief.flatten(start_dim=1).float() if is_batch else belief.flatten().float(); belief_raw_flat = belief_raw.flatten(start_dim=1).float() if is_batch else belief_raw.flatten().float()
             belief_norm = torch.linalg.norm(belief_flat, dim=-1, keepdim=True) + 1e-8; belief_raw_norm = torch.linalg.norm(belief_raw_flat, dim=-1, keepdim=True) + 1e-8
             cosine_sim = F.cosine_similarity(belief_flat, belief_raw_flat, dim=-1).unsqueeze(-1)
             self_consistency_batch = cosine_sim.detach(); rho_score_batch = torch.clamp((self_consistency_batch + 1.0) / 2.0, 0.0, 1.0).detach()
        else: logger.warning(f"Forward: Consistency shape mismatch.")
        if is_batch: self_consistency = self_consistency_batch.mean().item(); rho_score = rho_score_batch.mean().item()
        else:
            self_consistency = self_consistency_batch.item(); rho_score = rho_score_batch.item()
            if history_to_use is not None:
                R_accessibility = self.compute_accessibility(history_to_use.detach()); R_acc_mean = R_accessibility.mean().item() if R_accessibility.numel() > 0 else 0.0
                I_S_vector = self.lattice.S(history_to_use.detach(), 0, Config.HISTORY_SIZE - 1); I_S_norm = torch.linalg.norm(I_S_vector.float()).item() if I_S_vector.numel() > 0 else 0.0
                tau_t = Config.ACCESSIBILITY_THRESHOLD * (1 + att_score * 0.5)
                rho_struct_mem_short = self.memory.get_short_term_norm(); rho_struct_mem_long = self.memory.get_long_term_norm(); rho_struct_val = (rho_struct_mem_short * 0.3 + rho_struct_mem_long * 0.7)
                emotion_max = current_emotions.max().item() if current_emotions.numel() > 0 else 0.0; is_stable = emotion_max < Config.STABILITY_THRESHOLD; box_score = R_acc_mean if is_stable else 0.0;
        return (current_emotions, belief, feedback_signal, value, float(I_S_norm), float(rho_struct_val), float(att_score), float(self_consistency), float(rho_score), float(box_score), float(R_acc_mean), float(tau_t))

    def _get_default_outputs(self, batch_size=1) -> Tuple:
         zero_emo = torch.zeros(batch_size, Config.EMOTION_DIM, device=DEVICE) if batch_size > 1 else torch.zeros(Config.EMOTION_DIM, device=DEVICE)
         kaskade_out_dim = getattr(self.kaskade, '_output_dim', self.hidden_dim); zero_belief = torch.zeros(batch_size, kaskade_out_dim, device=DEVICE) if batch_size > 1 else torch.zeros(kaskade_out_dim, device=DEVICE)
         zero_feedback = torch.zeros(batch_size, self.state_dim, device=DEVICE) if batch_size > 1 else torch.zeros(self.state_dim, device=DEVICE)
         zero_value = torch.zeros(batch_size, 1, device=DEVICE) if batch_size > 1 else torch.zeros(1, device=DEVICE)
         return (zero_emo, zero_belief, zero_feedback, zero_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Config.ACCESSIBILITY_THRESHOLD)

    def step(self, state, reward, state_history, context) -> Tuple[torch.Tensor, str, torch.Tensor, float]:
        """ Processes simulation step. Returns outputs for simulation/memory."""
        # ... (step method remains the same as previous correction) ...
        self.eval()
        with torch.no_grad():
             if not isinstance(state, torch.Tensor) or not is_safe(state) or state.shape != (self.state_dim,): state = torch.zeros(self.state_dim, device=DEVICE)
             if state.device != DEVICE: state = state.to(DEVICE)
             forward_outputs = self.forward(state, reward, state_history)
             if len(forward_outputs) != 12: (emotions, belief, _, _, _, _, att_score, _, _, _, _, _) = self._get_default_outputs()
             else: (emotions, belief, _, _, _, _, att_score, _, _, _, _, _) = forward_outputs
             emotions = emotions.detach(); belief = belief.detach()
             try: response = self.gpt.generate(context, emotions)
             except Exception as e: logger.error(f"Error GPT gen step: {e}"); response = "..."
             discretized_state = self.lattice.discretize(state)
             self.state_history_deque.append(discretized_state.clone())
             self.prev_emotions = emotions.clone()
        return emotions, response, belief, att_score


    def learn(self, batch_size=32) -> float:
        """Samples batch, computes modulated loss using PER, value head, consistency, and optimizes."""
        self.train() # Ensure train mode

        # Anneal beta for PER IS weights
        self.beta = min(1.0, self.beta + self.beta_increment)

        if len(self.memory) < batch_size: return 0.0 # Not enough memory

        sample_result = self.memory.sample(batch_size, beta=self.beta)
        if sample_result is None: logger.warning("Learn: Memory sample returned None."); return 0.0
        batch, indices, weights = sample_result

        states = batch['states']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        if states.shape[0] != rewards.shape[0] or states.shape[0] != next_states.shape[0]: logger.error(f"Learn: Batch dim mismatch."); return 0.0
        current_batch_size = states.shape[0]
        if current_batch_size == 0: logger.warning("Learn: Batch size 0."); return 0.0

        # --- Calculate V(s) and Rho(s) for Current States (requires grad) ---
        try:
            # Forward returns: (emotions, belief, feedback, value, ..., rho_score, ...) - 12 items
            outputs = self.forward(states, torch.zeros_like(rewards), None)
            if len(outputs) != 12: raise ValueError("Forward learn V(s) output mismatch")

            current_value_pred = outputs[3] # Predicted value V(s)
            rho_score_batch = outputs[8]    # Rho score for the batch

            if not is_safe(current_value_pred) or current_value_pred.shape != (current_batch_size, 1): raise ValueError(f"Invalid V(s) shape/safety")
            if isinstance(rho_score_batch, float): rho_score_tensor = torch.full((current_batch_size, 1), rho_score_batch, device=DEVICE)
            else: rho_score_tensor = rho_score_batch.detach().clone()
            if rho_score_tensor.ndim == 1: rho_score_tensor = rho_score_tensor.unsqueeze(1)
            if rho_score_tensor.shape != (current_batch_size, 1): raise ValueError(f"Rho score tensor shape mismatch")
            if not is_safe(rho_score_tensor): raise ValueError("Invalid rho_score_tensor")

        except Exception as e: logger.error(f"Error V(s)/Metrics learn: {e}", exc_info=True); return -1.0

        # --- Calculate V(s') (no grad) ---
        next_value_pred = torch.zeros_like(rewards)
        with torch.no_grad():
            try:
                next_outputs = self.forward(next_states, torch.zeros_like(rewards), None)
                if len(next_outputs) == 12:
                     v_sp = next_outputs[3]; # Get V(s') prediction
                     if is_safe(v_sp) and v_sp.shape == (current_batch_size, 1): next_value_pred = v_sp
                     else: logger.warning("Learn: Invalid next_value prediction.")
                else: logger.warning("Learn: Forward V(s') mismatch.")
                next_value_pred[dones] = 0.0 # Terminal states value = 0
            except Exception as e: logger.error(f"Error V(s') learn: {e}", exc_info=True)

        # --- Calculate TD Target and TD Error ---
        # Incorporate intrinsic reward based on TD error magnitude (optional)
        intrinsic_reward = torch.abs(rewards + Config.GAMMA * next_value_pred.detach() - current_value_pred.detach()).clamp(max=1.0) * Config.INTRINSIC_REWARD_SCALE
        effective_rewards = rewards + intrinsic_reward.detach() # Add scaled intrinsic reward to extrinsic reward

        target_value = effective_rewards + Config.GAMMA * next_value_pred # Shape: (batch, 1)
        td_error = target_value - current_value_pred # Shape: (batch, 1), has grad

        # --- Modulated Loss ---
        loss = torch.tensor(0.0, device=DEVICE)
        if current_value_pred.requires_grad:
            base_loss_elementwise = F.mse_loss(current_value_pred, target_value.detach(), reduction='none') # Use detached target
            consistency_penalty = (1.0 + (1.0 - rho_score_tensor.clamp(0,1)))
            # Apply PER IS weights and consistency penalty
            modulated_loss = base_loss_elementwise * weights * consistency_penalty
            loss = modulated_loss.mean()
        else: logger.warning("Learn: current_value_pred does not require grad.")

        # --- Optimization ---
        loss_val = loss.item()
        if is_safe(loss) and loss.requires_grad:
             self.optimizer.zero_grad()
             loss.backward() # Backprop modulated loss
             agent_params_with_grad = [p for p in self.optimizer.param_groups[0]['params'] if p.grad is not None]
             if agent_params_with_grad: torch.nn.utils.clip_grad_norm_(agent_params_with_grad, max_norm=Config.GRADIENT_CLIP_AGENT)
             self.optimizer.step()

             # --- Update Memory Priorities ---
             self.memory.update_priorities(indices, td_error.detach()) # Update priorities with abs TD error

        elif not loss.requires_grad and abs(loss_val) > 1e-7 : logger.warning(f"Learn: Loss ({loss_val:.4f}) requires no grad.")
        elif not is_safe(loss): logger.warning(f"Learn: Unsafe loss ({loss_val:.4f})."); self.optimizer.zero_grad()
        else: pass # Loss is zero or safe but no grad

        return loss_val # Return loss for logging

# --- END OF FILE agent.py ---