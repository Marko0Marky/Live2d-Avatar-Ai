# --- START OF FILE agent.py ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Dict, Optional, List, Union, Deque, Any
import os # Import os for save/load

# Use MasterConfig object and tokenizer variables
from config import MasterConfig as Config
# Use TRAIN_DATA loaded in config.py
from config import DEVICE, logger, TRAIN_DATA, tokenizer, tokenize, detokenize, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID
# Import save/load paths
from config import AGENT_SAVE_PATH, GPT_SAVE_PATH, OPTIMIZER_SAVE_PATH, TARGET_NET_SAVE_SUFFIX
# --- Import Head Movement Labels ---
from config import HEAD_MOVEMENT_LABELS, NUM_HEAD_MOVEMENTS, IDX_TO_HEAD_MOVEMENT, HEAD_MOVEMENT_TO_IDX
# ---
from utils import MetronicLattice, MetaCognitiveMemory, is_safe, Experience
from ai_modules import EmotionalModule, SyntrixKorporator, StrukturKaskade, SimpleGPT


class ConsciousAgent(nn.Module):
    def __init__(self, state_dim: int = Config.Agent.STATE_DIM, hidden_dim: int = Config.Agent.HIDDEN_DIM, vocab_size: int = Config.NLP.VOCAB_SIZE):
        super().__init__()
        self.state_dim = state_dim
        self.base_state_dim = Config.Agent.BASE_STATE_DIM
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        logger.info(f"Agent Init: Combined State Dim={state_dim}, Base State Dim={self.base_state_dim}, Hidden Dim={hidden_dim}, Vocab Size={vocab_size}")
        if self.base_state_dim > self.state_dim: raise ValueError("AgentConfig BASE_STATE_DIM cannot be larger than STATE_DIM")

        # --- Online Networks ---
        self.lattice = MetronicLattice(dim=self.state_dim, tau=Config.Agent.TAU)
        self.korporator = SyntrixKorporator(self.state_dim, self.hidden_dim, m=min(8, hidden_dim // 2) if hidden_dim >= 2 else 1).to(DEVICE)
        self.kaskade = StrukturKaskade(self.hidden_dim, self.hidden_dim, levels=Config.Agent.CASCADE_LEVELS).to(DEVICE)
        kaskade_out_dim = self.kaskade._output_dim
        self.emotional_module = EmotionalModule(input_dim=Config.Agent.EMOTION_DIM + 1).to(DEVICE)
        self.feedback = nn.Linear(kaskade_out_dim, self.state_dim).to(DEVICE)
        self.value_head = nn.Linear(kaskade_out_dim, 1).to(DEVICE)
        self.head_movement_head = nn.Linear(kaskade_out_dim, NUM_HEAD_MOVEMENTS).to(DEVICE)
        logger.info(f"Initialized Head Movement prediction head ({kaskade_out_dim} -> {NUM_HEAD_MOVEMENTS})")

        # --- Attention Network ---
        self.attention: Optional[nn.MultiheadAttention] = None
        if self.state_dim > 0:
            possible_heads=[h for h in [16, 12, 8, 6, 4, 2, 1] if self.state_dim % h == 0]; num_heads = possible_heads[0] if possible_heads else 1
            if not possible_heads: logger.warning(f"Agent Attention fallback 1 head dim {self.state_dim}.")
            try:
                self.attention = nn.MultiheadAttention(embed_dim=self.state_dim, num_heads=num_heads, batch_first=True, dropout=0.15).to(DEVICE)
                logger.info(f"Agent Attention init {num_heads} heads dim {self.state_dim}.")
            except Exception as e:
                logger.error(f"Failed init Attention: {e}. Disabled.")
                self.attention = None

        # --- Target Networks (for DDQN stability) ---
        self.target_korporator = SyntrixKorporator(self.state_dim, self.hidden_dim, m=min(8, hidden_dim // 2) if hidden_dim >= 2 else 1).to(DEVICE)
        self.target_kaskade = StrukturKaskade(self.hidden_dim, self.hidden_dim, levels=Config.Agent.CASCADE_LEVELS).to(DEVICE)
        self.target_value_head = nn.Linear(kaskade_out_dim, 1).to(DEVICE)
        # Target HM head might not be strictly necessary for DDQN value estimate, but good practice if using its output elsewhere
        self.target_head_movement_head = nn.Linear(kaskade_out_dim, NUM_HEAD_MOVEMENTS).to(DEVICE)
        self.update_target_network() # Initial hard copy
        self.target_korporator.eval() # Target networks are only for inference
        self.target_kaskade.eval()
        self.target_value_head.eval()
        self.target_head_movement_head.eval()
        self.soft_update_tau = Config.RL.TARGET_NETWORK_SOFT_UPDATE_TAU # Use value from config

        # --- Other Components ---
        self.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE)
        self.gpt = SimpleGPT(vocab_size=Config.NLP.VOCAB_SIZE, embed_dim=64, hidden_dim=128, num_heads=4) # Keep SimpleGPT for now

        # --- Optimizer (Optimizes ONLINE networks only) ---
        online_params = list(self.korporator.parameters()) + list(self.kaskade.parameters()) + \
                       list(self.feedback.parameters()) + list(self.emotional_module.parameters()) + \
                       list(self.value_head.parameters()) + list(self.head_movement_head.parameters())
        if self.attention: online_params.extend(list(self.attention.parameters()))
        self.optimizer = optim.Adam(online_params, lr=Config.RL.LR)
        self._base_lr = Config.RL.LR

        # --- State Tracking ---
        self.state_history_deque: Deque[torch.Tensor] = deque(maxlen=Config.Agent.HISTORY_SIZE)
        for _ in range(Config.Agent.HISTORY_SIZE):
            self.state_history_deque.append(torch.zeros(self.state_dim, device=DEVICE))
        self.prev_emotions = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
        self.step_count = 0 # Tracks number of learning steps performed
        self.beta: float = Config.RL.PER_BETA_START
        beta_frames: int = Config.RL.PER_BETA_FRAMES
        self.beta_increment: float = (1.0 - self.beta) / beta_frames if beta_frames > 0 else 0.0

        # --- Initial GPT Training ---
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

        logger.info(f"ConsciousAgent initialized with STATE_DIM={self.state_dim} (Target Networks ENABLED)")

    @property
    def state_history(self) -> torch.Tensor:
        if not self.state_history_deque:
            return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE, dtype=torch.float32)
        valid_elements = True
        for t in self.state_history_deque:
            if not isinstance(t, torch.Tensor) or t.shape != (self.state_dim,):
                logger.error(f"Agent history invalid shape {t.shape}, expected ({self.state_dim},).")
                valid_elements = False
                break
        if not valid_elements:
            self.state_history_deque.clear()
            for _ in range(Config.Agent.HISTORY_SIZE):
                self.state_history_deque.append(torch.zeros(self.state_dim, device=DEVICE))
            return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE)
        try:
            return torch.stack(list(self.state_history_deque)).to(device=DEVICE, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error stacking agent history: {e}.")
            return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE)

    def update_target_network(self):
        """Hard update: Copies weights from online to target networks."""
        logger.debug("Hard updating target networks...")
        self.target_korporator.load_state_dict(self.korporator.state_dict())
        self.target_kaskade.load_state_dict(self.kaskade.state_dict())
        self.target_value_head.load_state_dict(self.value_head.state_dict())
        self.target_head_movement_head.load_state_dict(self.head_movement_head.state_dict())

    def _polyak_update(self, base_net, target_net, tau):
         """Helper for soft updates."""
         with torch.no_grad():
            for target_param, base_param in zip(target_net.parameters(), base_net.parameters()):
                  target_param.data.mul_(1.0 - tau)
                  target_param.data.add_(tau * base_param.data)

    def soft_update_target_networks(self, tau=None):
         """Soft update target network parameters (Polyak averaging)."""
         update_tau = tau if tau is not None else self.soft_update_tau
         # logger.debug(f"Soft updating target networks with tau={update_tau:.4f}...") # Can be noisy
         self._polyak_update(self.korporator, self.target_korporator, update_tau)
         self._polyak_update(self.kaskade, self.target_kaskade, update_tau)
         self._polyak_update(self.value_head, self.target_value_head, update_tau)
         self._polyak_update(self.head_movement_head, self.target_head_movement_head, update_tau)

    def compute_accessibility(self, history_tensor: torch.Tensor) -> torch.Tensor:
        # ... (implementation unchanged) ...
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

    def forward(self, state_input: torch.Tensor, current_reward_input: Union[float, torch.Tensor], full_state_history_input: Optional[torch.Tensor], use_target: bool = False) -> ForwardReturnType:
        """
        Performs a forward pass through the agent's networks.

        Args:
            state_input: The current state tensor (batch or single).
            current_reward_input: The current reward (batch or single).
            full_state_history_input: Optional full state history tensor.
            use_target: If True, uses the target networks for value estimation.

        Returns:
            A tuple containing various outputs like emotions, belief, value, etc.
        """
        is_batch = state_input.ndim == 2; batch_size = state_input.shape[0] if is_batch else 1
        if not isinstance(state_input, torch.Tensor) or state_input.shape[-1] != self.state_dim: logger.error(f"Agent.forward: Invalid state type/shape. Expected (*, {self.state_dim}), got {state_input.shape}."); return self._get_default_outputs(batch_size)
        state = state_input.to(DEVICE);
        if not is_safe(state): logger.warning("Agent.forward: Unsafe state input. Using zeros."); state = torch.zeros_like(state_input)

        # --- Process Reward ---
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

        # --- Discretize & Attention ---
        discretized_state = self.lattice.discretize(state.clone().detach());
        if not is_safe(discretized_state): discretized_state = torch.zeros_like(state)
        history_to_use = None; attn_output_context = discretized_state.clone(); attn_weights = None; att_score = 0.0
        if not is_batch:
            history_to_use = self.state_history
            if isinstance(full_state_history_input, torch.Tensor) and full_state_history_input.shape == (Config.Agent.HISTORY_SIZE, self.state_dim) and is_safe(full_state_history_input): history_to_use = full_state_history_input.to(DEVICE)
            elif full_state_history_input is not None: logger.warning("Agent.forward single: Invalid external history.")
            if self.attention and history_to_use is not None and history_to_use.shape[0] == Config.Agent.HISTORY_SIZE:
                 state_seq = history_to_use.unsqueeze(0).float().detach()
                 try:
                     attn_output_b, attn_weights_b = self.attention(state_seq, state_seq, state_seq)
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
                         valid_weights = attn_weights[non_diag_mask]
                         if valid_weights.numel() > 0:
                             att_score = valid_weights.mean().item()
                     else:
                         logger.warning("Unsafe attention weights.")
                 except Exception as e:
                    logger.error(f"Error self-attention: {e}")
                    attn_output_context = discretized_state[-1] if discretized_state.ndim == 2 else discretized_state
            elif not is_batch and self.attention and history_to_use is not None: logger.warning(f"Attn skipped: History size mismatch.")
        elif is_batch and self.attention: attn_output_context = discretized_state; pass # Use discretized state directly for batch context

        # --- Emotion Update ---
        if is_batch: emotion_state_part = discretized_state[:, :Config.Agent.EMOTION_DIM]; prev_emotions_for_module = self.prev_emotions.unsqueeze(0).repeat(batch_size, 1)
        else: emotion_state_part = discretized_state[:Config.Agent.EMOTION_DIM].unsqueeze(0); prev_emotions_for_module = self.prev_emotions.unsqueeze(0)
        if not is_safe(prev_emotions_for_module): prev_emotions_for_module = torch.zeros_like(prev_emotions_for_module)
        current_emotions_batch = self.emotional_module(emotion_state_part, current_reward, prev_emotions_for_module)
        if not is_batch: current_emotions = current_emotions_batch.squeeze(0); self.prev_emotions = current_emotions.detach().clone() # Update agent's internal prev_emotions only for single step
        else: current_emotions = current_emotions_batch
        if not is_safe(current_emotions): logger.warning("Unsafe emotions from module."); current_emotions = torch.zeros_like(current_emotions)

        # --- Belief Formation & Value/HM Prediction ---
        # Select networks based on use_target flag
        korp = self.target_korporator if use_target else self.korporator
        kask = self.target_kaskade if use_target else self.kaskade
        val_head = self.target_value_head if use_target else self.value_head
        hm_head = self.target_head_movement_head if use_target else self.head_movement_head

        psi_input = attn_output_context # Context from attention (or just state if batch/no attention)
        belief_raw = korp(discretized_state, psi_input, level=2);
        belief = kask(belief_raw);
        if not is_safe(belief): belief = torch.zeros_like(belief_raw) # Use belief_raw shape as fallback

        value = val_head(belief);
        if not is_safe(value): value = torch.zeros_like(value)

        head_movement_logits = hm_head(belief)
        if not is_safe(head_movement_logits): logger.warning("Unsafe head movement logits."); hm_shape = (batch_size, NUM_HEAD_MOVEMENTS) if is_batch else (NUM_HEAD_MOVEMENTS,); head_movement_logits = torch.zeros(hm_shape, device=DEVICE)

        # Feedback signal uses online network regardless of target usage
        feedback_signal = self.feedback(belief);
        if not is_safe(feedback_signal): feedback_signal = torch.zeros_like(feedback_signal)

        # --- Metrics Calculation ---
        # These metrics reflect the state of the *online* agent, even if using target nets for value
        I_S_norm = 0.0; R_acc_mean = 0.0; tau_t = Config.Agent.ACCESSIBILITY_THRESHOLD; rho_struct_val = 0.0; box_score = 0.0;
        self_consistency_batch = torch.zeros(batch_size, 1, device=DEVICE) if is_batch else torch.tensor(0.0, device=DEVICE)
        rho_score_batch = torch.zeros(batch_size, 1, device=DEVICE) if is_batch else torch.tensor(0.0, device=DEVICE)
        if belief.shape == belief_raw.shape and belief.numel() > 0: # Compare online belief vs belief_raw
             belief_flat = belief.flatten(start_dim=1).float() if is_batch else belief.flatten().float(); belief_raw_flat = belief_raw.flatten(start_dim=1).float() if is_batch else belief_raw.flatten().float()
             belief_norm = torch.linalg.norm(belief_flat, dim=-1, keepdim=True) + 1e-8; belief_raw_norm = torch.linalg.norm(belief_raw_flat, dim=-1, keepdim=True) + 1e-8
             cosine_sim = F.cosine_similarity(belief_flat / belief_norm, belief_raw_flat / belief_raw_norm, dim=-1).unsqueeze(-1)
             self_consistency_batch = cosine_sim.detach(); rho_score_batch = torch.clamp((self_consistency_batch + 1.0) / 2.0, 0.0, 1.0).detach()
        else: logger.warning(f"Forward: Consistency shape mismatch {belief.shape} vs {belief_raw.shape}.")
        if is_batch:
            self_consistency = self_consistency_batch.mean().item(); rho_score = rho_score_batch.mean().item()
            # Metrics below require history, set defaults for batch mode
            I_S_norm, rho_struct_val, box_score, R_acc_mean, tau_t = 0.0, 0.0, 0.0, 0.0, Config.Agent.ACCESSIBILITY_THRESHOLD;
        else: # Calculate history-based metrics only for single instance forward pass
            self_consistency = self_consistency_batch.item(); rho_score = rho_score_batch.item()
            if history_to_use is not None:
                R_accessibility = self.compute_accessibility(history_to_use.detach()); R_acc_mean = R_accessibility.mean().item() if R_accessibility.numel() > 0 else 0.0
                I_S_vector = self.lattice.S(history_to_use.detach(), 0, Config.Agent.HISTORY_SIZE - 1); I_S_norm = torch.linalg.norm(I_S_vector.float()).item() if I_S_vector.numel() > 0 else 0.0
                tau_t = Config.Agent.ACCESSIBILITY_THRESHOLD * (1 + att_score * 0.5) # att_score calculated earlier
                rho_struct_mem_short = self.memory.get_short_term_norm(); rho_struct_mem_long = self.memory.get_long_term_norm(); rho_struct_val = (rho_struct_mem_short * 0.3 + rho_struct_mem_long * 0.7)
                emotion_max_val = current_emotions.max().item() if current_emotions.numel() > 0 else 0.0; is_stable = emotion_max_val < Config.Agent.STABILITY_THRESHOLD; box_score = R_acc_mean if is_stable else 0.0;

        metrics_float = tuple(float(m) for m in [I_S_norm, rho_struct_val, att_score, self_consistency, rho_score, box_score, R_acc_mean, tau_t])

        return (current_emotions, belief, feedback_signal, value, *metrics_float, head_movement_logits)

    def _get_default_outputs(self, batch_size=1) -> ForwardReturnType:
         # ... (implementation unchanged) ...
         zero_emo = torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE) if batch_size > 1 else torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
         kaskade_out_dim = getattr(self.kaskade, '_output_dim', self.hidden_dim); zero_belief = torch.zeros(batch_size, kaskade_out_dim, device=DEVICE) if batch_size > 1 else torch.zeros(kaskade_out_dim, device=DEVICE)
         zero_feedback = torch.zeros(batch_size, self.state_dim, device=DEVICE) if batch_size > 1 else torch.zeros(self.state_dim, device=DEVICE)
         zero_value = torch.zeros(batch_size, 1, device=DEVICE) if batch_size > 1 else torch.zeros(1, device=DEVICE)
         zero_metrics = (0.0,) * 8
         zero_hm_logits = torch.zeros(batch_size, NUM_HEAD_MOVEMENTS, device=DEVICE) if batch_size > 1 else torch.zeros(NUM_HEAD_MOVEMENTS, device=DEVICE)
         return (zero_emo, zero_belief, zero_feedback, zero_value, *zero_metrics, zero_hm_logits)

    StepReturnType = Tuple[torch.Tensor, str, torch.Tensor, float, str]

    def step(self, state: torch.Tensor, reward: float, state_history: torch.Tensor, context: Optional[str]) -> StepReturnType:
        """Performs a single agent step (inference only)."""
        self.eval() # Ensure model is in evaluation mode for step
        predicted_hm_label = "idle"
        with torch.no_grad():
             if not isinstance(state, torch.Tensor) or not is_safe(state) or state.shape != (self.state_dim,): logger.warning(f"Agent step invalid state {state.shape}. Exp ({self.state_dim},). Zeros."); state = torch.zeros(self.state_dim, device=DEVICE)
             if state.device != DEVICE: state = state.to(DEVICE)
             # Use online network for stepping/acting
             forward_outputs = self.forward(state, reward, state_history, use_target=False)
             current_emotions = forward_outputs[0]; belief_for_memory = forward_outputs[1]; att_score_metric = forward_outputs[6]; head_movement_logits = forward_outputs[-1]

             # Predict head movement
             try:
                 if head_movement_logits.ndim == 1 and head_movement_logits.shape[0] == NUM_HEAD_MOVEMENTS: predicted_hm_idx = torch.argmax(head_movement_logits).item(); predicted_hm_label = IDX_TO_HEAD_MOVEMENT.get(predicted_hm_idx, "idle")
                 elif head_movement_logits.ndim == 2 and head_movement_logits.shape[0] == 1 and head_movement_logits.shape[1] == NUM_HEAD_MOVEMENTS: predicted_hm_idx = torch.argmax(head_movement_logits.squeeze(0)).item(); predicted_hm_label = IDX_TO_HEAD_MOVEMENT.get(predicted_hm_idx, "idle")
                 else: logger.warning(f"Unexpected HM logits shape: {head_movement_logits.shape}. Default 'idle'."); predicted_hm_label = "idle"
             except Exception as e: logger.error(f"Error predicting head movement: {e}"); predicted_hm_label = "idle"

             # Generate response
             try: response_context = context if context else "..."; temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7); top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9); response = self.gpt.generate(response_context, current_emotions, temperature=temp, top_p=top_p)
             except Exception as e: logger.error(f"Error GPT gen step: {e}"); response = "..."

             # Update history (only in non-batch mode)
             discretized_state = self.lattice.discretize(state); self.state_history_deque.append(discretized_state.clone())

        # Note: self.prev_emotions was updated inside forward() for the single step
        return current_emotions, response, belief_for_memory, att_score_metric, predicted_hm_label

    def learn(self, batch_size: int = Config.RL.AGENT_BATCH_SIZE) -> float:
        """Performs a learning update using a batch from memory."""
        self.train() # Ensure model is in training mode
        self.beta = min(1.0, self.beta + self.beta_increment)
        if len(self.memory) < batch_size: return 0.0 # Not enough samples

        sample_result = self.memory.sample(batch_size, beta=self.beta)
        if sample_result is None: logger.warning("Learn: Memory sample returned None."); return 0.0
        batch_data, indices, weights = sample_result

        states = batch_data['states'].to(DEVICE); rewards = batch_data['rewards'].to(DEVICE)
        next_states = batch_data['next_states'].to(DEVICE); dones = batch_data['dones'].to(DEVICE)
        target_hm_indices = batch_data.get('target_hm_idx').to(DEVICE) if batch_data.get('target_hm_idx') is not None else None

        current_batch_size = states.shape[0]
        if current_batch_size == 0: logger.warning("Learn: Sampled batch size is 0."); return 0.0

        # --- Get V(s) and HM(s) using ONLINE network ---
        try:
            outputs_online = self.forward(states, torch.zeros_like(rewards), None, use_target=False)
            current_value_pred = outputs_online[3]
            head_movement_logits = outputs_online[-1]
            rho_score_batch = outputs_online[8]
            box_score_batch = outputs_online[9]

            if not is_safe(current_value_pred) or current_value_pred.shape != (current_batch_size, 1): raise ValueError(f"Invalid Online V(s) shape/safety {current_value_pred.shape}")
            if not is_safe(head_movement_logits) or head_movement_logits.shape != (current_batch_size, NUM_HEAD_MOVEMENTS): raise ValueError(f"Invalid Online HM logits shape/safety {head_movement_logits.shape}")

            # Process metrics tensors
            rho_score_tensor = torch.tensor(rho_score_batch, device=DEVICE).view(-1, 1) if isinstance(rho_score_batch, float) else rho_score_batch.detach()
            box_score_tensor = torch.tensor(box_score_batch, device=DEVICE).view(-1, 1) if isinstance(box_score_batch, float) else box_score_batch.detach()
            if not is_safe(rho_score_tensor) or rho_score_tensor.shape != (current_batch_size, 1): raise ValueError("Invalid Rho score tensor")
            if not is_safe(box_score_tensor) or box_score_tensor.shape != (current_batch_size, 1): raise ValueError("Invalid Box score tensor")

        except Exception as e: logger.error(f"Error Online Net calc in learn: {e}", exc_info=True); return -1.0

        # --- Get V(s') using TARGET network ---
        next_value_pred = torch.zeros_like(rewards)
        with torch.no_grad():
            try:
                # Use TARGET networks for next state value estimation
                outputs_target = self.forward(next_states, torch.zeros_like(rewards), None, use_target=True)
                v_sp_target = outputs_target[3]

                if is_safe(v_sp_target) and v_sp_target.shape == (current_batch_size, 1):
                    next_value_pred = v_sp_target
                else:
                    logger.warning(f"Learn: Invalid target next_value pred shape/safety {v_sp_target.shape}.")

                dones_squeezed = dones.squeeze(-1) if dones.ndim > 1 else dones
                next_value_pred[dones_squeezed] = 0.0 # Set value of terminal states to 0
            except Exception as e:
                logger.error(f"Error Target V(s') calc: {e}", exc_info=True)
                next_value_pred = torch.zeros_like(rewards) # Fallback

        # --- Calculate TD Target and Error ---
        # Calculate intrinsic rewards based on ONLINE network's metrics
        intrinsic_reward_consistency = rho_score_tensor * Config.RL.INTRINSIC_REWARD_SCALE_CONSISTENCY
        intrinsic_reward_box = box_score_tensor * Config.RL.INTRINSIC_REWARD_SCALE_BOX
        td_error_abs_estimate = torch.abs(rewards + Config.RL.GAMMA * next_value_pred.detach() - current_value_pred.detach())
        intrinsic_reward_td_error = td_error_abs_estimate.clamp(max=1.0) * Config.RL.INTRINSIC_REWARD_SCALE_TD
        effective_rewards = rewards + (intrinsic_reward_consistency + intrinsic_reward_box + intrinsic_reward_td_error).detach()

        target_value = effective_rewards + Config.RL.GAMMA * next_value_pred # Uses V(s') from TARGET network
        td_error = target_value - current_value_pred # TD Error based on ONLINE V(s) and TARGET V(s')

        # --- Calculate Losses ---
        # Value Loss (using PER weights)
        value_loss = torch.tensor(0.0, device=DEVICE)
        if current_value_pred.requires_grad:
            value_loss_elementwise = F.smooth_l1_loss(current_value_pred, target_value.detach(), reduction='none', beta=1.0)
            # consistency_penalty = (1.0 + (1.0 - rho_score_tensor.clamp(0,1))) # Optional penalty
            value_loss = (value_loss_elementwise * weights).mean() # Apply PER weights
        else:
            logger.warning("Learn: current_value_pred does not require grad. Value loss is 0.")

        # Head Movement Loss (Supervised from data, weighted by PER weights)
        movement_loss = torch.tensor(0.0, device=DEVICE)
        hm_loss_weight = Config.RL.HEAD_MOVEMENT_LOSS_WEIGHT
        if hm_loss_weight > 0 and target_hm_indices is not None and head_movement_logits.requires_grad:
             if target_hm_indices.shape[0] == current_batch_size:
                 try:
                     if target_hm_indices.ndim > 1: target_hm_indices = target_hm_indices.squeeze(-1)
                     hm_loss_func = nn.CrossEntropyLoss(reduction='none')
                     hm_loss_elementwise = hm_loss_func(head_movement_logits, target_hm_indices.long())
                     movement_loss = (hm_loss_elementwise * weights.squeeze(-1)).mean() # Apply PER weights
                     if not torch.isfinite(movement_loss): logger.warning(f"Movement loss is NaN/Inf!"); movement_loss = torch.tensor(0.0, device=DEVICE)
                 except Exception as hm_err: logger.error(f"Learn: Error calculating head movement loss: {hm_err}", exc_info=True); movement_loss = torch.tensor(0.0, device=DEVICE)
             else: logger.warning(f"Learn: Mismatch batch size vs target HM indices. Skip HM loss.")

        # Total Loss
        total_loss = value_loss + movement_loss * hm_loss_weight

        # --- Optimization Step ---
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
             # Clip gradients for ONLINE parameters only
             online_params_with_grad = [p for pg in self.optimizer.param_groups for p in pg['params'] if p.grad is not None]
             if online_params_with_grad: torch.nn.utils.clip_grad_norm_(online_params_with_grad, max_norm=Config.RL.GRADIENT_CLIP_AGENT)
             else: logger.warning("Learn: No gradients found for online network after backward pass.")
             self.optimizer.step();

             # Update priorities in PER buffer
             self.memory.update_priorities(indices, td_error.detach())

             # --- Update Target Network (using soft updates) ---
             self.soft_update_target_networks()

        elif not total_loss.requires_grad and abs(loss_val) > 1e-7 : logger.debug(f"Learn: Total Loss ({loss_val:.4f}) requires no grad.")
        elif not is_safe(total_loss): logger.warning(f"Learn: Unsafe total loss ({loss_val:.4f}). Skip step."); self.optimizer.zero_grad()
        else: pass # Zero loss or requires_grad is False, skip update

        if Config.RL.ADAPTIVE_LR_ENABLED and original_lrs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i < len(original_lrs): param_group['lr'] = original_lrs[i]

        self.step_count += 1 # Increment learn step counter

        return loss_val

    # --- Add save/load methods ---
    def save_state(self, agent_path: str = AGENT_SAVE_PATH, gpt_path: str = GPT_SAVE_PATH, optimizer_path: str = OPTIMIZER_SAVE_PATH, target_path_suffix: str = TARGET_NET_SAVE_SUFFIX):
        """Saves the state of the agent, GPT, optimizer, and target networks."""
        logger.info(f"Saving agent state to {agent_path}...")
        try:
            os.makedirs(os.path.dirname(agent_path), exist_ok=True)
            os.makedirs(os.path.dirname(gpt_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)

            # Save Online networks
            online_state = {
                'korporator_state_dict': self.korporator.state_dict(),
                'kaskade_state_dict': self.kaskade.state_dict(),
                'emotional_module_state_dict': self.emotional_module.state_dict(),
                'feedback_state_dict': self.feedback.state_dict(),
                'value_head_state_dict': self.value_head.state_dict(),
                'head_movement_head_state_dict': self.head_movement_head.state_dict(),
                'attention_state_dict': self.attention.state_dict() if self.attention else None,
                'prev_emotions': self.prev_emotions,
                'step_count': self.step_count, # Save learn step counter
                'beta': self.beta, # Save PER beta
            }
            torch.save(online_state, agent_path)

            # Save Target networks (append suffix)
            target_state = {
                'target_korporator_state_dict': self.target_korporator.state_dict(),
                'target_kaskade_state_dict': self.target_kaskade.state_dict(),
                'target_value_head_state_dict': self.target_value_head.state_dict(),
                'target_head_movement_head_state_dict': self.target_head_movement_head.state_dict(),
            }
            target_agent_path = agent_path.replace(".pth", f"{target_path_suffix}.pth")
            os.makedirs(os.path.dirname(target_agent_path), exist_ok=True) # Ensure dir exists
            torch.save(target_state, target_agent_path)


            # Save GPT separately
            self.gpt.save_model(gpt_path)

            # Save Optimizer
            torch.save(self.optimizer.state_dict(), optimizer_path)

            logger.info("Agent online, target, GPT, and optimizer states saved successfully.")
        except IOError as e: logger.error(f"IOError saving agent state: {e}", exc_info=True)
        except Exception as e: logger.error(f"Unexpected error saving agent state: {e}", exc_info=True)

    def load_state(self, agent_path: str = AGENT_SAVE_PATH, gpt_path: str = GPT_SAVE_PATH, optimizer_path: str = OPTIMIZER_SAVE_PATH, target_path_suffix: str = TARGET_NET_SAVE_SUFFIX):
        """Loads the state of the agent, GPT, optimizer, and target networks."""
        logger.info(f"Loading agent state from {agent_path}...")
        target_agent_path = agent_path.replace(".pth", f"{target_path_suffix}.pth")

        if not os.path.exists(agent_path):
            logger.warning(f"Agent online state file not found: {agent_path}. Skipping load.")
            return False
        if not os.path.exists(target_agent_path):
            logger.warning(f"Agent target state file not found: {target_agent_path}. Skipping target load.")
            load_targets = False
        else:
            load_targets = True

        try:
            # Load Online state
            agent_state = torch.load(agent_path, map_location=DEVICE)
            self.korporator.load_state_dict(agent_state['korporator_state_dict'])
            self.kaskade.load_state_dict(agent_state['kaskade_state_dict'])
            self.emotional_module.load_state_dict(agent_state['emotional_module_state_dict'])
            self.feedback.load_state_dict(agent_state['feedback_state_dict'])
            self.value_head.load_state_dict(agent_state['value_head_state_dict'])
            self.head_movement_head.load_state_dict(agent_state['head_movement_head_state_dict'])
            if self.attention and agent_state.get('attention_state_dict'):
                self.attention.load_state_dict(agent_state['attention_state_dict'])
            self.prev_emotions = agent_state.get('prev_emotions', self.prev_emotions).to(DEVICE)
            self.step_count = agent_state.get('step_count', self.step_count) # Load learn step counter
            self.beta = agent_state.get('beta', self.beta) # Load PER beta
            logger.info("Online agent networks loaded.")

            # Load Target state
            if load_targets:
                target_state = torch.load(target_agent_path, map_location=DEVICE)
                self.target_korporator.load_state_dict(target_state['target_korporator_state_dict'])
                self.target_kaskade.load_state_dict(target_state['target_kaskade_state_dict'])
                self.target_value_head.load_state_dict(target_state['target_value_head_state_dict'])
                self.target_head_movement_head.load_state_dict(target_state['target_head_movement_head_state_dict'])
                logger.info("Target agent networks loaded.")
            else: # If target file missing, copy from loaded online nets
                logger.warning("Target state file missing, performing hard copy from loaded online networks.")
                self.update_target_network()

            # Load GPT separately
            gpt_loaded = self.gpt.load_model(gpt_path)
            if not gpt_loaded: logger.warning("GPT model failed to load.")

            # Load Optimizer state
            if os.path.exists(optimizer_path):
                 self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
                 logger.info("Optimizer state loaded.")
                 # Re-link parameters after loading state dicts
                 online_params = list(self.korporator.parameters()) + list(self.kaskade.parameters()) + \
                                list(self.feedback.parameters()) + list(self.emotional_module.parameters()) + \
                                list(self.value_head.parameters()) + list(self.head_movement_head.parameters())
                 if self.attention: online_params.extend(list(self.attention.parameters()))
                 self.optimizer.param_groups[0]['params'] = online_params
            else:
                 logger.warning(f"Optimizer state file not found: {optimizer_path}. Optimizer not loaded (will start fresh).")


            logger.info("Agent state loading complete.")
            self.eval() # Set online networks to evaluation mode
            self.target_korporator.eval() # Ensure target networks are also in eval mode
            self.target_kaskade.eval()
            self.target_value_head.eval()
            self.target_head_movement_head.eval()
            return True

        except FileNotFoundError:
             logger.error(f"Error: State file not found at {agent_path} or related paths.")
             return False
        except KeyError as e:
            logger.error(f"Error loading state: Missing key {e}. State file might be incompatible.", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading agent state: {e}", exc_info=True)
            return False


# --- END OF FILE agent.py ---
