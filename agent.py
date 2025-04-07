# --- START OF FILE agent.py ---

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Dict, Optional, List, Union, Deque, Any

# Use MasterConfig object and tokenizer variables
from config import MasterConfig as Config
# --- USE VALIDATED TRAIN_DATA from config ---
from config import DEVICE, logger, TRAIN_DATA, tokenizer, tokenize, detokenize, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID
# ---
from utils import MetronicLattice, MetaCognitiveMemory, is_safe, Experience
from ai_modules import EmotionalModule, SyntrixKorporator, StrukturKaskade, SimpleGPT


class ConsciousAgent(nn.Module):
    def __init__(self, state_dim: int = Config.Agent.STATE_DIM, hidden_dim: int = Config.Agent.HIDDEN_DIM, vocab_size: int = Config.NLP.VOCAB_SIZE):
        super().__init__();
        self.state_dim = state_dim; self.hidden_dim = hidden_dim; self.vocab_size = vocab_size
        logger.info(f"Agent Init: state_dim={state_dim}, hidden_dim={hidden_dim}, vocab_size={vocab_size}")

        # --- Initialize components ---
        # Lattice now uses the full combined state dimension for agent processing
        self.lattice = MetronicLattice(dim=state_dim, tau=Config.Agent.TAU);
        self.korporator = SyntrixKorporator(self.state_dim, self.hidden_dim, m=min(8, hidden_dim // 2) if hidden_dim >= 2 else 1); # Increased m slightly
        self.kaskade = StrukturKaskade(hidden_dim, hidden_dim, levels=Config.Agent.CASCADE_LEVELS);
        kaskade_out_dim = self.kaskade._output_dim;
        # Emotional module still uses only EMOTION_DIM + reward
        self.emotional_module = EmotionalModule(input_dim=Config.Agent.EMOTION_DIM + 1)
        self.attention: Optional[nn.MultiheadAttention] = None;
        if state_dim > 0:
            # Find suitable number of heads for the potentially large state_dim
            possible_heads=[h for h in [16, 12, 8, 6, 4, 2, 1] if state_dim % h == 0];
            num_heads = possible_heads[0] if possible_heads else 1;
            if not possible_heads: logger.warning(f"Agent Attention using fallback 1 head for state_dim {state_dim}.");
            try:
                # Increased dropout slightly for larger state
                self.attention=nn.MultiheadAttention(embed_dim=state_dim, num_heads=num_heads, batch_first=True, dropout=0.15);
                logger.info(f"Agent Attention init {num_heads} heads for state_dim {state_dim}.")
            except Exception as e: logger.error(f"Failed init Attention: {e}. Disabled."); self.attention=None
        self.feedback = nn.Linear(kaskade_out_dim, state_dim);
        self.value_head = nn.Linear(kaskade_out_dim, 1)
        self.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE)
        self.gpt = SimpleGPT(vocab_size=Config.NLP.VOCAB_SIZE, embed_dim=64, hidden_dim=128, num_heads=4)

        agent_params = list(self.korporator.parameters()) + list(self.kaskade.parameters()) + list(self.feedback.parameters()) + list(self.emotional_module.parameters()) + list(self.value_head.parameters())
        if self.attention: agent_params.extend(list(self.attention.parameters()))
        self.optimizer = optim.Adam(agent_params, lr=Config.RL.LR)
        self._base_lr = Config.RL.LR

        # State history now stores the *combined* state
        self.state_history_deque: Deque[torch.Tensor] = deque(maxlen=Config.Agent.HISTORY_SIZE);
        for _ in range(Config.Agent.HISTORY_SIZE): self.state_history_deque.append(torch.zeros(Config.Agent.STATE_DIM, device=DEVICE)) # Use combined STATE_DIM
        self.prev_emotions = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE); self.step_count = 0

        self.beta: float = Config.RL.PER_BETA_START
        beta_frames: int = Config.RL.PER_BETA_FRAMES
        self.beta_increment: float = (1.0 - self.beta) / beta_frames if beta_frames > 0 else 0.0

        # --- Use validated TRAIN_DATA loaded in config.py ---
        if TRAIN_DATA and tokenizer is not None: # Check if data is valid and tokenizer exists
            try:
                logger.info("Starting initial GPT training (using pre-validated data)...");
                self.gpt.train_model(TRAIN_DATA, epochs=Config.NLP.TRAIN_EPOCHS);
                logger.info("Initial GPT training complete.")
            except Exception as e:
                logger.error(f"Error during initial GPT training in Agent: {e}", exc_info=True)
        elif tokenizer is None:
             logger.warning("Skipping initial GPT training: Tokenizer not initialized.")
        else:
             logger.info("Skipping initial GPT training: No valid training data loaded or found.")
        # --- End GPT Training ---

        logger.info("ConsciousAgent initialized.")

    @property
    def state_history(self) -> torch.Tensor:
        """Provides the combined state history as a stacked torch tensor."""
        if not self.state_history_deque: return torch.zeros(Config.Agent.HISTORY_SIZE, Config.Agent.STATE_DIM, device=DEVICE, dtype=torch.float32)
        valid_elements = True;
        for t in self.state_history_deque:
             # Check against the combined STATE_DIM
             if not isinstance(t, torch.Tensor) or t.shape != (Config.Agent.STATE_DIM,):
                 logger.error(f"Agent history invalid shape. Expected ({Config.Agent.STATE_DIM},), Got {t.shape}. Reinitializing.");
                 valid_elements = False; break
        if not valid_elements:
             self.state_history_deque.clear();
             for _ in range(Config.Agent.HISTORY_SIZE): self.state_history_deque.append(torch.zeros(Config.Agent.STATE_DIM, device=DEVICE))
             return torch.zeros(Config.Agent.HISTORY_SIZE, Config.Agent.STATE_DIM, device=DEVICE)
        try: return torch.stack(list(self.state_history_deque)).to(device=DEVICE, dtype=torch.float32)
        except Exception as e: logger.error(f"Error stacking agent history: {e}."); return torch.zeros(Config.Agent.HISTORY_SIZE, Config.Agent.STATE_DIM, device=DEVICE)

    def compute_accessibility(self, history_tensor: torch.Tensor) -> torch.Tensor:
        """Computes accessibility based on the *combined* state history."""
        default_matrix = torch.zeros((Config.Agent.HISTORY_SIZE, Config.Agent.HISTORY_SIZE), device=DEVICE)
        # Check against combined STATE_DIM
        if not isinstance(history_tensor, torch.Tensor) or history_tensor.shape != (Config.Agent.HISTORY_SIZE, Config.Agent.STATE_DIM):
            logger.warning(f"Accessibility received invalid history shape {history_tensor.shape}. Expected {(Config.Agent.HISTORY_SIZE, Config.Agent.STATE_DIM)}.")
            return default_matrix
        if not is_safe(history_tensor): logger.warning("Accessibility skipped: Unsafe history."); return default_matrix
        try:
            history_float = history_tensor.float(); norms = torch.linalg.norm(history_float, dim=1, keepdim=True) + 1e-8
            normalized_history = history_float / norms; similarity_matrix = torch.matmul(normalized_history, normalized_history.t())
            similarity_matrix = (similarity_matrix + 1.0) / 2.0; similarity_matrix = torch.clamp(similarity_matrix, 0.0, 1.0)
            accessibility_matrix = torch.where(similarity_matrix > Config.Agent.ACCESSIBILITY_THRESHOLD, similarity_matrix, torch.zeros_like(similarity_matrix))
            if not is_safe(accessibility_matrix): logger.warning("Calculated accessibility matrix unsafe."); return default_matrix
            return accessibility_matrix
        except Exception as e: logger.error(f"Error accessibility calc: {e}", exc_info=True); return default_matrix

    ForwardReturnType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float, float, float, float, float]

    def forward(self, state_input: torch.Tensor, current_reward_input: Union[float, torch.Tensor], full_state_history_input: Optional[torch.Tensor]) -> ForwardReturnType:
        """Processes the combined state (base + lang embedding if enabled)."""
        is_batch = state_input.ndim == 2; batch_size = state_input.shape[0] if is_batch else 1
        # Check input state shape against combined STATE_DIM
        if not isinstance(state_input, torch.Tensor) or (is_batch and state_input.shape[1] != self.state_dim) or (not is_batch and state_input.shape[0] != self.state_dim):
             logger.error(f"Agent.forward: Invalid state shape. Expected dim {self.state_dim}, got {state_input.shape}.");
             return self._get_default_outputs(batch_size)

        state = state_input.to(DEVICE);
        if not is_safe(state): logger.warning("Agent.forward: Unsafe state input. Using zeros."); state = torch.zeros_like(state_input)

        # Reward processing (remains same)
        if is_batch:
            if isinstance(current_reward_input, torch.Tensor):
                if current_reward_input.shape == (batch_size, 1): current_reward = current_reward_input.to(DEVICE).float()
                elif current_reward_input.shape == (batch_size,): current_reward = current_reward_input.to(DEVICE).float().unsqueeze(1)
                else: logger.warning(f"Agent.forward batch: Reward shape mismatch {current_reward_input.shape}. Using zeros."); current_reward = torch.zeros(batch_size, 1, device=DEVICE, dtype=torch.float32)
            else:
                 try:
                     rewards_float = [float(r) for r in current_reward_input] # type: ignore
                     current_reward = torch.tensor(rewards_float, device=DEVICE, dtype=torch.float32).unsqueeze(1)
                     if current_reward.shape[0] != batch_size: raise ValueError("Batch size mismatch")
                 except (ValueError, TypeError): logger.warning(f"Agent.forward batch: Invalid non-tensor reward input type {type(current_reward_input)}. Using zeros."); current_reward = torch.zeros(batch_size, 1, device=DEVICE, dtype=torch.float32)
        else:
            try: current_reward_float = float(current_reward_input); current_reward = torch.tensor([[current_reward_float]], device=DEVICE, dtype=torch.float32)
            except (ValueError, TypeError): logger.warning(f"Agent.forward single: Invalid reward input type {type(current_reward_input)}. Using zero."); current_reward = torch.tensor([[0.0]], device=DEVICE, dtype=torch.float32)
        if not is_safe(current_reward): logger.warning("Agent.forward: Unsafe reward. Using zeros."); current_reward = torch.zeros_like(current_reward).float()

        # Discretize the *combined* state
        discretized_state = self.lattice.discretize(state.clone().detach());
        if not is_safe(discretized_state): discretized_state = torch.zeros_like(state)

        history_to_use = None; attn_output_context = discretized_state.clone(); attn_weights = None; att_score = 0.0
        if not is_batch:
            history_to_use = self.state_history # Get combined state history
            # Check external history shape against combined STATE_DIM
            if isinstance(full_state_history_input, torch.Tensor) and full_state_history_input.shape == (Config.Agent.HISTORY_SIZE, Config.Agent.STATE_DIM) and is_safe(full_state_history_input):
                 history_to_use = full_state_history_input.to(DEVICE)
            elif full_state_history_input is not None: logger.warning(f"Agent.forward single: Invalid external history shape {full_state_history_input.shape}.")

            if self.attention and history_to_use is not None and history_to_use.shape[0] == Config.Agent.HISTORY_SIZE:
                 state_seq = history_to_use.unsqueeze(0).float().detach()
                 try:
                     attn_output_b, attn_weights_b = self.attention(state_seq, state_seq, state_seq);
                     if is_safe(attn_output_b) and attn_output_b.shape == (1, Config.Agent.HISTORY_SIZE, self.state_dim): attn_output_context = attn_output_b[0, -1, :].detach()
                     else: logger.warning("Unsafe/invalid shape attn output."); attn_output_context = discretized_state[-1] if discretized_state.ndim == 2 else discretized_state
                     if attn_output_context.shape != (self.state_dim,): logger.warning(f"Attn context shape mismatch {attn_output_context.shape} vs {(self.state_dim,)}."); attn_output_context = discretized_state[-1] if discretized_state.ndim == 2 else discretized_state
                     attn_weights = attn_weights_b.squeeze(0)
                     if is_safe(attn_weights):
                         non_diag_mask = ~torch.eye(Config.Agent.HISTORY_SIZE, dtype=torch.bool, device=DEVICE); valid_weights = attn_weights[non_diag_mask];
                         if valid_weights.numel() > 0: att_score = valid_weights.mean().item()
                         else: logger.warning("No valid non-diagonal attention weights.")
                     else: logger.warning("Unsafe attention weights.")
                 except Exception as e: logger.error(f"Error self-attention: {e}"); attn_output_context = discretized_state[-1] if discretized_state.ndim == 2 else discretized_state
            elif not is_batch and self.attention and history_to_use is not None: logger.warning(f"Attn skipped: History size mismatch {history_to_use.shape[0]} vs {Config.Agent.HISTORY_SIZE}.")
        elif is_batch and self.attention:
            # Batch attention might need different handling if history is per-item
            # For now, just pass discretized state through
            attn_output_context = discretized_state; pass # Placeholder for potential batch attention implementation
        if is_batch:
            # Extract emotion part from the start of the combined state
            emotion_state_part = discretized_state[:, :Config.Agent.EMOTION_DIM]
            prev_emotions_for_module = self.prev_emotions.unsqueeze(0).repeat(batch_size, 1)
        else:
            # Extract emotion part from the start of the combined state
            emotion_state_part = discretized_state[:Config.Agent.EMOTION_DIM].unsqueeze(0)
            prev_emotions_for_module = self.prev_emotions.unsqueeze(0)

        if not is_safe(prev_emotions_for_module): prev_emotions_for_module = torch.zeros_like(prev_emotions_for_module)
        # Emotional module input shape remains EMOTION_DIM + 1
        current_emotions_batch = self.emotional_module(emotion_state_part, current_reward, prev_emotions_for_module)
        if not is_batch: current_emotions = current_emotions_batch.squeeze(0); self.prev_emotions = current_emotions.detach().clone()
        else: current_emotions = current_emotions_batch
        if not is_safe(current_emotions): logger.warning("Unsafe emotions from module."); current_emotions = torch.zeros_like(current_emotions)

        psi_input = attn_output_context # Use attention context if available, else discretized state
        # Korporator and Kaskade process the combined state/context
        belief_raw = self.korporator(discretized_state, psi_input, level=2); belief = self.kaskade(belief_raw);
        if not is_safe(belief): belief = torch.zeros_like(belief_raw)
        value = self.value_head(belief);
        if not is_safe(value): value = torch.zeros_like(value)
        # Feedback signal should match the combined state dimension
        feedback_signal = self.feedback(belief);
        if not is_safe(feedback_signal): feedback_signal = torch.zeros_like(feedback_signal)

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
            # Accessibility and I_S use the combined state history
            if history_to_use is not None:
                R_accessibility = self.compute_accessibility(history_to_use.detach()); R_acc_mean = R_accessibility.mean().item() if R_accessibility.numel() > 0 else 0.0
                I_S_vector = self.lattice.S(history_to_use.detach(), 0, Config.Agent.HISTORY_SIZE - 1); I_S_norm = torch.linalg.norm(I_S_vector.float()).item() if I_S_vector.numel() > 0 else 0.0
                tau_t = Config.Agent.ACCESSIBILITY_THRESHOLD * (1 + att_score * 0.5)
                rho_struct_mem_short = self.memory.get_short_term_norm(); rho_struct_mem_long = self.memory.get_long_term_norm(); rho_struct_val = (rho_struct_mem_short * 0.3 + rho_struct_mem_long * 0.7)
                emotion_max_val = current_emotions.max().item() if current_emotions.numel() > 0 else 0.0; is_stable = emotion_max_val < Config.Agent.STABILITY_THRESHOLD; box_score = R_acc_mean if is_stable else 0.0;
        metrics_float = tuple(float(m) for m in [I_S_norm, rho_struct_val, att_score, self_consistency, rho_score, box_score, R_acc_mean, tau_t])
        return (current_emotions, belief, feedback_signal, value, *metrics_float)

    def _get_default_outputs(self, batch_size=1) -> ForwardReturnType:
         zero_emo = torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE) if batch_size > 1 else torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
         kaskade_out_dim = getattr(self.kaskade, '_output_dim', self.hidden_dim); zero_belief = torch.zeros(batch_size, kaskade_out_dim, device=DEVICE) if batch_size > 1 else torch.zeros(kaskade_out_dim, device=DEVICE)
         # Feedback signal should match combined state dimension
         zero_feedback = torch.zeros(batch_size, self.state_dim, device=DEVICE) if batch_size > 1 else torch.zeros(self.state_dim, device=DEVICE)
         zero_value = torch.zeros(batch_size, 1, device=DEVICE) if batch_size > 1 else torch.zeros(1, device=DEVICE)
         zero_metrics = (0.0,) * 8
         return (zero_emo, zero_belief, zero_feedback, zero_value, *zero_metrics)

    StepReturnType = Tuple[torch.Tensor, str, torch.Tensor, float]

    def step(self, state: torch.Tensor, reward: float, state_history: torch.Tensor, context: Optional[str]) -> StepReturnType:
        """Takes the combined state, returns emotions, response, belief, attention."""
        self.eval()
        with torch.no_grad():
             # Check combined state dim
             if not isinstance(state, torch.Tensor) or not is_safe(state) or state.shape != (self.state_dim,):
                 logger.warning(f"Agent step received invalid state shape {state.shape}. Expected ({self.state_dim},). Using zeros.");
                 state = torch.zeros(self.state_dim, device=DEVICE)
             if state.device != DEVICE: state = state.to(DEVICE)

             # Forward pass uses combined state
             forward_outputs = self.forward(state, reward, state_history)
             current_emotions = forward_outputs[0]; belief_for_memory = forward_outputs[1]; att_score_metric = forward_outputs[6]
             try:
                 response_context = context if context else "..."; temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7); top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9)
                 response = self.gpt.generate(response_context, current_emotions, temperature=temp, top_p=top_p)
             except Exception as e: logger.error(f"Error GPT gen step: {e}"); response = "..."

             # Discretize the *combined* state for history storage
             discretized_state = self.lattice.discretize(state)
             # Append combined state to history
             self.state_history_deque.append(discretized_state.clone())
        return current_emotions, response, belief_for_memory, att_score_metric

    def learn(self, batch_size: int = Config.RL.AGENT_BATCH_SIZE) -> float:
        """Learns from a batch of experiences (which contain combined states)."""
        self.train()
        self.beta = min(1.0, self.beta + self.beta_increment)
        if len(self.memory) < batch_size: return 0.0
        sample_result = self.memory.sample(batch_size, beta=self.beta)
        if sample_result is None: logger.warning("Learn: Memory sample returned None."); return 0.0
        batch_data, indices, weights = sample_result

        # States and next_states from memory are already the *combined* states
        states = batch_data['states']; rewards = batch_data['rewards']; next_states = batch_data['next_states']; dones = batch_data['dones']
        if states.shape[0] != rewards.shape[0] or states.shape[0] != next_states.shape[0]: logger.error(f"Learn: Batch dim mismatch."); return 0.0
        current_batch_size = states.shape[0]
        if current_batch_size == 0: logger.warning("Learn: Batch size 0."); return 0.0

        # Check combined state dimension from memory
        if states.shape[1] != self.state_dim or next_states.shape[1] != self.state_dim:
            logger.error(f"Learn: State dimension mismatch in memory batch! Expected {self.state_dim}, got states {states.shape}, next_states {next_states.shape}. Returning -1.")
            return -1.0

        try:
            zero_rewards_batch = torch.zeros_like(rewards)
            # Pass combined states to forward pass
            outputs = self.forward(states, zero_rewards_batch, None)
            current_value_pred = outputs[3]; rho_score_batch = outputs[8]; box_score_batch = outputs[9]
            if not is_safe(current_value_pred) or current_value_pred.shape != (current_batch_size, 1): raise ValueError(f"Invalid V(s) shape/safety {current_value_pred.shape}")
            if isinstance(rho_score_batch, float): rho_score_tensor = torch.full((current_batch_size, 1), rho_score_batch, device=DEVICE)
            elif isinstance(rho_score_batch, torch.Tensor): rho_score_tensor = rho_score_batch.detach().clone()
            else: raise ValueError(f"Invalid rho_score type {type(rho_score_batch)}")
            if rho_score_tensor.ndim == 1: rho_score_tensor = rho_score_tensor.unsqueeze(1)
            if rho_score_tensor.shape != (current_batch_size, 1): raise ValueError(f"Rho score tensor shape mismatch: {rho_score_tensor.shape}")
            if not is_safe(rho_score_tensor): raise ValueError("Unsafe rho_score_tensor")
            if isinstance(box_score_batch, float): box_score_tensor = torch.full((current_batch_size, 1), box_score_batch, device=DEVICE)
            elif isinstance(box_score_batch, torch.Tensor): box_score_tensor = box_score_batch.detach().clone()
            else: raise ValueError(f"Invalid box_score type {type(box_score_batch)}")
            if box_score_tensor.ndim == 1: box_score_tensor = box_score_tensor.unsqueeze(1)
            if box_score_tensor.shape != (current_batch_size, 1): raise ValueError(f"Box score tensor shape mismatch: {box_score_tensor.shape}")
            if not is_safe(box_score_tensor): raise ValueError("Unsafe box_score_tensor")
        except Exception as e: logger.error(f"Error V(s)/Metrics learn: {e}", exc_info=True); return -1.0
        next_value_pred = torch.zeros_like(rewards)
        with torch.no_grad():
            try:
                zero_rewards_next_batch = torch.zeros_like(rewards)
                # Pass combined next_states to forward pass
                next_outputs = self.forward(next_states, zero_rewards_next_batch, None)
                v_sp = next_outputs[3];
                if is_safe(v_sp) and v_sp.shape == (current_batch_size, 1): next_value_pred = v_sp
                else: logger.warning(f"Learn: Invalid next_value prediction shape/safety {v_sp.shape}."); next_value_pred = torch.zeros_like(rewards)
                next_value_pred[dones.squeeze(-1)] = 0.0 # Ensure dones mask works correctly
            except Exception as e: logger.error(f"Error V(s') learn: {e}", exc_info=True); next_value_pred = torch.zeros_like(rewards)

        # Rewards calculations (remain same logic)
        intrinsic_reward_consistency = rho_score_tensor * Config.RL.INTRINSIC_REWARD_SCALE_CONSISTENCY
        intrinsic_reward_box = box_score_tensor * Config.RL.INTRINSIC_REWARD_SCALE_BOX
        intrinsic_reward_td_error = torch.abs(rewards + Config.RL.GAMMA * next_value_pred.detach() - current_value_pred.detach()).clamp(max=1.0) * Config.RL.INTRINSIC_REWARD_SCALE_TD
        effective_rewards = rewards + (intrinsic_reward_consistency + intrinsic_reward_box + intrinsic_reward_td_error).detach()
        target_value = effective_rewards + Config.RL.GAMMA * next_value_pred
        td_error = target_value - current_value_pred
        loss = torch.tensor(0.0, device=DEVICE)
        if current_value_pred.requires_grad:
            base_loss_elementwise = F.mse_loss(current_value_pred, target_value.detach(), reduction='none')
            consistency_penalty = (1.0 + (1.0 - rho_score_tensor.clamp(0,1)))
            modulated_loss = base_loss_elementwise * weights * consistency_penalty
            loss = modulated_loss.mean()
        else: logger.warning("Learn: current_value_pred does not require grad.")
        original_lrs = []
        dynamic_lr = self._base_lr
        if Config.RL.ADAPTIVE_LR_ENABLED and rho_score_tensor.numel() > 0:
            avg_rho_score = rho_score_tensor.mean().item()
            min_factor = Config.RL.LR_ADAPTIVE_MIN_FACTOR; max_factor = Config.RL.LR_ADAPTIVE_MAX_FACTOR
            dynamic_lr_factor = min_factor + (max_factor - min_factor) * max(0.0, min(1.0, avg_rho_score))
            dynamic_lr = self._base_lr * dynamic_lr_factor
            # logger.debug(f"Adaptive LR: Rho={avg_rho_score:.3f}, Factor={dynamic_lr_factor:.3f}, LR={dynamic_lr:.6f}") # Can be verbose
            for i, param_group in enumerate(self.optimizer.param_groups): original_lrs.append(param_group['lr']); param_group['lr'] = dynamic_lr
        loss_val = loss.item()
        if is_safe(loss) and loss.requires_grad:
             self.optimizer.zero_grad(); loss.backward()
             agent_params_with_grad = [p for p in self.optimizer.param_groups[0]['params'] if p.grad is not None]
             if agent_params_with_grad: torch.nn.utils.clip_grad_norm_(agent_params_with_grad, max_norm=Config.RL.GRADIENT_CLIP_AGENT)
             self.optimizer.step(); self.memory.update_priorities(indices, td_error.detach())
        elif not loss.requires_grad and abs(loss_val) > 1e-7 : logger.debug(f"Learn: Loss ({loss_val:.4f}) requires no grad.")
        elif not is_safe(loss): logger.warning(f"Learn: Unsafe loss ({loss_val:.4f})."); self.optimizer.zero_grad()
        else: pass
        if Config.RL.ADAPTIVE_LR_ENABLED and original_lrs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i < len(original_lrs): param_group['lr'] = original_lrs[i]
        return loss_val

# --- END OF FILE agent.py ---
