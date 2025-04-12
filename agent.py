# --- START OF FILE agent.py ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Dict, Optional, List, Union, Deque, Any
import sys # Import sys for exit
import os # Import os for save/load


# Use MasterConfig object and tokenizer variables
from config import MasterConfig as Config
# Use TRAIN_DATA loaded in config.py
from config import DEVICE, logger, TRAIN_DATA # Keep TRAIN_DATA
# --- REMOVED BPE tokenizer imports ---
# Import save/load paths
from config import AGENT_SAVE_PATH, GPT_SAVE_PATH, OPTIMIZER_SAVE_PATH, TARGET_NET_SAVE_SUFFIX
# --- Import Head Movement Labels ---
from config import HEAD_MOVEMENT_LABELS, NUM_HEAD_MOVEMENTS, IDX_TO_HEAD_MOVEMENT, HEAD_MOVEMENT_TO_IDX
# ---
from utils import MetronicLattice, MetaCognitiveMemory, is_safe, Experience
# --- MODIFIED: Import TransformerGPT instead of SimpleGPT ---
from ai_modules import EmotionalModule, SyntrixKorporator, StrukturKaskade, TransformerGPT
# ---

class ConsciousAgent(nn.Module):
    # REMOVED vocab_size from signature
    def __init__(self, state_dim: int = Config.Agent.STATE_DIM, hidden_dim: int = Config.Agent.HIDDEN_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.base_state_dim = Config.Agent.BASE_STATE_DIM
        self.hidden_dim = hidden_dim
        # REMOVED self.vocab_size assignment
        logger.info(f"Agent Init: Combined State Dim={state_dim}, Base State Dim={self.base_state_dim}, Hidden Dim={hidden_dim}")
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
            # Find suitable number of heads based on state dimension
            possible_heads=[h for h in [16, 12, 8, 6, 4, 2, 1] if self.state_dim % h == 0];
            num_heads = possible_heads[0] if possible_heads else 1
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
        self.target_head_movement_head = nn.Linear(kaskade_out_dim, NUM_HEAD_MOVEMENTS).to(DEVICE)
        self.update_target_network() # Initial hard copy
        self.target_korporator.eval() # Target networks are only for inference
        self.target_kaskade.eval()
        self.target_value_head.eval()
        self.target_head_movement_head.eval()
        self.soft_update_tau = Config.RL.TARGET_NETWORK_SOFT_UPDATE_TAU # Use value from config

        # --- Other Components ---
        self.memory = MetaCognitiveMemory(capacity=Config.Agent.MEMORY_SIZE)
        # --- MODIFIED: Instantiate TransformerGPT ---
        try:
            self.gpt = TransformerGPT() # Instantiate the new wrapper
            logger.info(f"Using TransformerGPT with model: {Config.NLP.HUGGINGFACE_MODEL}")
        except Exception as e:
             logger.critical(f"CRITICAL: Failed to initialize TransformerGPT: {e}. Cannot continue.")
             # Ensure sys is imported if using sys.exit
             # import sys
             sys.exit(1)
        # --- END MODIFIED ---

        # --- Optimizer (Optimizes ONLINE agent networks ONLY) ---
        # GPT fine-tuning is handled separately by its Trainer
        online_params = list(self.korporator.parameters()) + list(self.kaskade.parameters()) + \
                       list(self.feedback.parameters()) + list(self.emotional_module.parameters()) + \
                       list(self.value_head.parameters()) + list(self.head_movement_head.parameters())
        if self.attention: online_params.extend(list(self.attention.parameters()))
        # Ensure all parameters require gradients
        online_params = [p for p in online_params if p.requires_grad]
        if not online_params:
             logger.warning("Agent optimizer created with NO trainable parameters!")
        self.optimizer = optim.Adam(online_params, lr=Config.RL.LR)
        self._base_lr = Config.RL.LR # Store base LR for adaptive LR

        # --- State Tracking ---
        self.state_history_deque: Deque[torch.Tensor] = deque(maxlen=Config.Agent.HISTORY_SIZE)
        # Initialize with zero tensors matching the state dimension
        for _ in range(Config.Agent.HISTORY_SIZE):
            self.state_history_deque.append(torch.zeros(self.state_dim, device=DEVICE))
        self.prev_emotions = torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
        self.step_count = 0 # Tracks number of learning steps performed
        self.beta: float = Config.RL.PER_BETA_START # For PER
        beta_frames: int = Config.RL.PER_BETA_FRAMES
        self.beta_increment: float = (1.0 - self.beta) / beta_frames if beta_frames > 0 else 0.0

        # --- REMOVED Initial GPT Training Call ---
        logger.info("Skipping initial GPT training in agent init (handled separately or uses pre-trained).")

        logger.info(f"ConsciousAgent initialized with STATE_DIM={self.state_dim} (Target Networks ENABLED)")

    @property
    def state_history(self) -> torch.Tensor:
        """Returns the state history as a stacked tensor."""
        # Ensure deque has elements and they are valid tensors
        if not self.state_history_deque:
            return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE, dtype=torch.float32)

        valid_elements = True
        for t in self.state_history_deque:
            if not isinstance(t, torch.Tensor) or t.shape != (self.state_dim,):
                logger.error(f"Agent history contains invalid element shape {getattr(t, 'shape', 'None')}, expected ({self.state_dim},). Reinitializing history.")
                valid_elements = False
                break

        if not valid_elements:
            # Clear and refill with zeros if invalid elements were found
            self.state_history_deque.clear()
            for _ in range(Config.Agent.HISTORY_SIZE):
                self.state_history_deque.append(torch.zeros(self.state_dim, device=DEVICE))
            return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE, dtype=torch.float32)

        try:
            # Stack the tensors from the deque
            return torch.stack(list(self.state_history_deque)).to(device=DEVICE, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error stacking agent history: {e}. Returning zeros.", exc_info=True)
            # Fallback: return zeros if stacking fails
            return torch.zeros(Config.Agent.HISTORY_SIZE, self.state_dim, device=DEVICE, dtype=torch.float32)

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
        """Computes accessibility matrix based on state history similarity."""
        default_matrix = torch.zeros((Config.Agent.HISTORY_SIZE, Config.Agent.HISTORY_SIZE), device=DEVICE)
        if not isinstance(history_tensor, torch.Tensor) or history_tensor.shape != (Config.Agent.HISTORY_SIZE, self.state_dim):
            logger.warning(f"Accessibility input shape mismatch: Got {history_tensor.shape}, Expected ({Config.Agent.HISTORY_SIZE}, {self.state_dim})")
            return default_matrix
        if not is_safe(history_tensor):
            logger.warning("Accessibility skipped: Unsafe history tensor.")
            return default_matrix
        try:
            # Ensure float type and normalize rows
            history_float = history_tensor.float()
            norms = torch.linalg.norm(history_float, dim=1, keepdim=True) + 1e-8 # Add epsilon for stability
            normalized_history = history_float / norms
            # Calculate cosine similarity matrix
            similarity_matrix = torch.matmul(normalized_history, normalized_history.t())
            # Scale similarity to [0, 1] range (optional, depends on interpretation)
            similarity_matrix = (similarity_matrix + 1.0) / 2.0
            similarity_matrix = torch.clamp(similarity_matrix, 0.0, 1.0)
            # Apply accessibility threshold
            accessibility_matrix = torch.where(
                similarity_matrix > Config.Agent.ACCESSIBILITY_THRESHOLD,
                similarity_matrix,
                torch.zeros_like(similarity_matrix)
            )
            # Final safety check
            if not is_safe(accessibility_matrix):
                logger.warning("Calculated accessibility matrix is unsafe.")
                return default_matrix
            return accessibility_matrix
        except Exception as e:
            logger.error(f"Error calculating accessibility matrix: {e}", exc_info=True)
            return default_matrix

    # Define return type hint for clarity
    ForwardReturnType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, torch.Tensor, torch.Tensor, torch.Tensor, float, float, torch.Tensor]

    def forward(self, state_input: torch.Tensor, current_reward_input: Union[float, torch.Tensor], full_state_history_input: Optional[torch.Tensor], use_target: bool = False) -> ForwardReturnType:
        """
        Performs a forward pass through the agent's networks.

        Args:
            state_input: The current state tensor (batch or single).
            current_reward_input: The current reward (batch or single).
            full_state_history_input: Optional full state history tensor.
            use_target: If True, uses the target networks for value estimation.

        Returns:
            A tuple containing:
             (current_emotions, belief, feedback_signal, value,
              I_S_norm, rho_struct_val, att_score, self_consistency_batch, rho_score_batch, box_score_batch,
              R_acc_mean, tau_t, head_movement_logits)
        """
        is_batch = state_input.ndim == 2; batch_size = state_input.shape[0] if is_batch else 1
        if not isinstance(state_input, torch.Tensor) or state_input.shape[-1] != self.state_dim: logger.error(f"Agent.forward: Invalid state type/shape. Expected (*, {self.state_dim}), got {state_input.shape}."); return self._get_default_outputs(batch_size)
        state = state_input.to(DEVICE);
        if not is_safe(state): logger.warning("Agent.forward: Unsafe state input. Using zeros."); state = torch.zeros_like(state_input)

        # --- Process Reward ---
        # Handles both single float/tensor and batch tensor inputs
        if is_batch:
            if isinstance(current_reward_input, torch.Tensor):
                if current_reward_input.shape == (batch_size, 1): current_reward = current_reward_input.to(DEVICE).float()
                elif current_reward_input.shape == (batch_size,): current_reward = current_reward_input.to(DEVICE).float().unsqueeze(1)
                else: logger.warning(f"Agent.forward batch: Reward shape mismatch {current_reward_input.shape}. Using zeros."); current_reward = torch.zeros(batch_size, 1, device=DEVICE, dtype=torch.float32)
            # Handle list/tuple of rewards for batch
            elif isinstance(current_reward_input, (list, tuple)) and len(current_reward_input) == batch_size:
                 try: rewards_float = [float(r) for r in current_reward_input]; current_reward = torch.tensor(rewards_float, device=DEVICE, dtype=torch.float32).unsqueeze(1);
                 except (ValueError, TypeError): logger.warning(f"Agent.forward batch: Invalid non-tensor reward input type {type(current_reward_input)}. Using zeros."); current_reward = torch.zeros(batch_size, 1, device=DEVICE, dtype=torch.float32)
            else: logger.warning(f"Agent.forward batch: Invalid reward input type/length {type(current_reward_input)}. Using zeros."); current_reward = torch.zeros(batch_size, 1, device=DEVICE, dtype=torch.float32)
        else: # Single instance
            try: current_reward_float = float(current_reward_input); current_reward = torch.tensor([[current_reward_float]], device=DEVICE, dtype=torch.float32)
            except (ValueError, TypeError): logger.warning(f"Agent.forward single: Invalid reward input type {type(current_reward_input)}. Using zero."); current_reward = torch.tensor([[0.0]], device=DEVICE, dtype=torch.float32)

        if not is_safe(current_reward): logger.warning("Agent.forward: Unsafe reward. Using zeros."); current_reward = torch.zeros_like(current_reward).float()

        # --- Discretize & Attention ---
        discretized_state = self.lattice.discretize(state.clone().detach());
        if not is_safe(discretized_state): discretized_state = torch.zeros_like(state)
        attn_output_context = discretized_state.clone(); # Default context is just the state
        att_score = 0.0 # Default attention score

        # Attention only computed in single instance mode with valid history
        if not is_batch and self.attention:
            history_to_use = self.state_history # Use agent's internal history by default
            # Override with provided history if valid
            if isinstance(full_state_history_input, torch.Tensor) and full_state_history_input.shape == (Config.Agent.HISTORY_SIZE, self.state_dim) and is_safe(full_state_history_input):
                 history_to_use = full_state_history_input.to(DEVICE)
            elif full_state_history_input is not None:
                 logger.warning("Agent.forward single: Invalid external history provided.")

            if history_to_use is not None and history_to_use.shape[0] == Config.Agent.HISTORY_SIZE:
                 state_seq = history_to_use.unsqueeze(0).float().detach() # Add batch dim
                 try:
                     # Compute self-attention over history
                     attn_output_b, attn_weights_b = self.attention(state_seq, state_seq, state_seq) # Q, K, V are all history
                     # Use the last output token's context if safe
                     if is_safe(attn_output_b) and attn_output_b.shape == (1, Config.Agent.HISTORY_SIZE, self.state_dim):
                         attn_output_context = attn_output_b[0, -1, :].detach() # Get context from last time step
                     else: logger.warning("Unsafe or invalid shape attention output.")
                     # Ensure context shape matches state dim
                     if attn_output_context.shape != (self.state_dim,): attn_output_context = discretized_state # Fallback

                     # Calculate average attention score (excluding self-attention)
                     if attn_weights_b is not None and is_safe(attn_weights_b):
                         attn_weights = attn_weights_b.squeeze(0).detach() # Remove batch dim
                         non_diag_mask = ~torch.eye(Config.Agent.HISTORY_SIZE, dtype=torch.bool, device=DEVICE)
                         valid_weights = attn_weights[non_diag_mask]
                         if valid_weights.numel() > 0: att_score = valid_weights.mean().item()
                     else: logger.warning("Unsafe attention weights.")
                 except Exception as e:
                    logger.error(f"Error during self-attention calculation: {e}")
            elif history_to_use is not None:
                 logger.warning(f"Attention skipped: History size mismatch ({history_to_use.shape[0]} vs {Config.Agent.HISTORY_SIZE}).")
        # For batch mode, attn_output_context remains discretized_state

        # --- Emotion Update ---
        # Prepare inputs for batch/single instance
        if is_batch:
            emotion_state_part = discretized_state[:, :Config.Agent.EMOTION_DIM];
            prev_emotions_for_module = self.prev_emotions.unsqueeze(0).repeat(batch_size, 1) # Use internal prev_emotions as base for batch
        else:
            emotion_state_part = discretized_state[:Config.Agent.EMOTION_DIM].unsqueeze(0); # Add batch dim
            prev_emotions_for_module = self.prev_emotions.unsqueeze(0) # Use internal prev_emotions

        if not is_safe(prev_emotions_for_module): prev_emotions_for_module = torch.zeros_like(prev_emotions_for_module)
        current_emotions_batch = self.emotional_module(emotion_state_part, current_reward, prev_emotions_for_module)

        if not is_batch:
            current_emotions = current_emotions_batch.squeeze(0);
            self.prev_emotions = current_emotions.detach().clone() # Update agent's internal prev_emotions only for single step
        else:
            current_emotions = current_emotions_batch # Keep as batch

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
        if not is_safe(head_movement_logits):
            logger.warning("Unsafe head movement logits.");
            hm_shape = (batch_size, NUM_HEAD_MOVEMENTS) if is_batch else (NUM_HEAD_MOVEMENTS,)
            head_movement_logits = torch.zeros(hm_shape, device=DEVICE)

        # Feedback signal always uses online network
        feedback_signal = self.feedback(belief);
        if not is_safe(feedback_signal): feedback_signal = torch.zeros_like(feedback_signal)

        # --- Metrics Calculation ---
        # Initialize metrics (some only relevant for single instance)
        I_S_norm = 0.0; R_acc_mean = 0.0; tau_t = Config.Agent.ACCESSIBILITY_THRESHOLD; rho_struct_val = 0.0;
        # Use tensors for batch consistency in learn() step
        self_consistency_batch = torch.zeros(batch_size, 1, device=DEVICE) if is_batch else torch.tensor([[0.0]], device=DEVICE)
        rho_score_batch = torch.zeros(batch_size, 1, device=DEVICE) if is_batch else torch.tensor([[0.0]], device=DEVICE)
        box_score_batch = torch.zeros(batch_size, 1, device=DEVICE) if is_batch else torch.tensor([[0.0]], device=DEVICE)

        # Calculate self-consistency (cosine similarity between belief_raw and belief)
        # These metrics reflect the state of the *online* agent, even if using target nets for value
        if belief.shape == belief_raw.shape and belief.numel() > 0:
             belief_flat = belief.flatten(start_dim=1).float() if is_batch else belief.flatten().float().unsqueeze(0);
             belief_raw_flat = belief_raw.flatten(start_dim=1).float() if is_batch else belief_raw.flatten().float().unsqueeze(0)
             belief_norm = torch.linalg.norm(belief_flat, dim=-1, keepdim=True) + 1e-8;
             belief_raw_norm = torch.linalg.norm(belief_raw_flat, dim=-1, keepdim=True) + 1e-8
             safe_belief = belief_flat / belief_norm; safe_belief_raw = belief_raw_flat / belief_raw_norm
             cosine_sim = F.cosine_similarity(safe_belief, safe_belief_raw, dim=-1).unsqueeze(-1) # Ensure output shape (batch, 1)
             self_consistency_batch = cosine_sim.detach();
             # Rho score: Scale cosine similarity [0, 1]
             rho_score_batch = torch.clamp((self_consistency_batch + 1.0) / 2.0, 0.0, 1.0).detach()
        else: logger.warning(f"Forward: Consistency calculation shape mismatch {belief.shape} vs {belief_raw.shape}.")

        # Calculate history-based metrics only for single instance forward pass
        if not is_batch:
            history_for_metrics = history_to_use if 'history_to_use' in locals() and history_to_use is not None else self.state_history
            if history_for_metrics is not None and history_for_metrics.shape[0] == Config.Agent.HISTORY_SIZE:
                # Accessibility
                R_accessibility = self.compute_accessibility(history_for_metrics.detach())
                R_acc_mean = R_accessibility.mean().item() if R_accessibility.numel() > 0 else 0.0
                # Lattice Stability
                I_S_vector = self.lattice.S(history_for_metrics.detach(), 0, Config.Agent.HISTORY_SIZE - 1)
                I_S_norm = torch.linalg.norm(I_S_vector.float()).item() if I_S_vector.numel() > 0 else 0.0
                # Dynamic Threshold
                tau_t = Config.Agent.ACCESSIBILITY_THRESHOLD * (1 + att_score * 0.5) # Modulated by attention
                # Memory Structure Norm
                rho_struct_mem_short = self.memory.get_short_term_norm();
                rho_struct_mem_long = self.memory.get_long_term_norm();
                rho_struct_val = (rho_struct_mem_short * 0.3 + rho_struct_mem_long * 0.7)
                # Box Score (Stability-gated accessibility)
                emotion_max_val = current_emotions.max().item() if current_emotions.numel() > 0 else 0.0;
                is_stable = emotion_max_val < Config.Agent.STABILITY_THRESHOLD;
                box_score = R_acc_mean if is_stable else 0.0
                # Update the single-item tensor for box score
                box_score_batch = torch.tensor([[box_score]], device=DEVICE)
            else: # No valid history for single instance
                 I_S_norm, rho_struct_val, box_score, R_acc_mean, tau_t = 0.0, 0.0, 0.0, 0.0, Config.Agent.ACCESSIBILITY_THRESHOLD
                 box_score_batch = torch.tensor([[0.0]], device=DEVICE)

        # Return metrics (float for single, tensor for batch where applicable) and the logits
        # Note: Ensure the order matches ForwardReturnType
        return (current_emotions, belief, feedback_signal, value,
                float(I_S_norm), float(rho_struct_val), float(att_score),
                self_consistency_batch, # Return batch tensor
                rho_score_batch,      # Return batch tensor
                box_score_batch,      # Return batch tensor
                float(R_acc_mean), float(tau_t),
                head_movement_logits)

    def _get_default_outputs(self, batch_size=1) -> ForwardReturnType:
         """Returns zero tensors with correct shapes for default output."""
         zero_emo = torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE) if batch_size > 1 else torch.zeros(Config.Agent.EMOTION_DIM, device=DEVICE)
         kaskade_out_dim = getattr(self.kaskade, '_output_dim', Config.Agent.HIDDEN_DIM); # Use actual or default
         zero_belief = torch.zeros(batch_size, kaskade_out_dim, device=DEVICE) if batch_size > 1 else torch.zeros(kaskade_out_dim, device=DEVICE)
         zero_feedback = torch.zeros(batch_size, self.state_dim, device=DEVICE) if batch_size > 1 else torch.zeros(self.state_dim, device=DEVICE)
         zero_value = torch.zeros(batch_size, 1, device=DEVICE) if batch_size > 1 else torch.zeros(1, device=DEVICE)
         # Zero metrics (handle batch tensors)
         zero_I_S, zero_rho_s, zero_att, zero_R_acc, zero_tau = 0.0, 0.0, 0.0, 0.0, 0.0
         zero_self_consistency = torch.zeros(batch_size, 1, device=DEVICE)
         zero_rho_score = torch.zeros(batch_size, 1, device=DEVICE)
         zero_box_score = torch.zeros(batch_size, 1, device=DEVICE)
         # Zero logits
         zero_hm_logits = torch.zeros(batch_size, NUM_HEAD_MOVEMENTS, device=DEVICE) if batch_size > 1 else torch.zeros(NUM_HEAD_MOVEMENTS, device=DEVICE)

         return (zero_emo, zero_belief, zero_feedback, zero_value,
                 zero_I_S, zero_rho_s, zero_att,
                 zero_self_consistency, zero_rho_score, zero_box_score,
                 zero_R_acc, zero_tau,
                 zero_hm_logits)

    # Define return type hint for clarity
    StepReturnType = Tuple[torch.Tensor, str, torch.Tensor, float, str]

    def step(self, state: torch.Tensor, reward: float, state_history: torch.Tensor, context: Optional[str]) -> StepReturnType:
        """Performs a single agent step (inference only)."""
        self.eval() # Ensure model is in evaluation mode for step
        predicted_hm_label = "idle" # Default
        with torch.no_grad():
             # Validate input state
             if not isinstance(state, torch.Tensor) or not is_safe(state) or state.shape != (self.state_dim,):
                 logger.warning(f"Agent step received invalid state shape {getattr(state, 'shape', 'None')}. Expected ({self.state_dim},). Using zeros.");
                 state = torch.zeros(self.state_dim, device=DEVICE)
             if state.device != DEVICE: state = state.to(DEVICE)

             # Use online network for stepping/acting
             forward_outputs = self.forward(state, reward, state_history, use_target=False)

             # Correctly unpack 13 items
             if len(forward_outputs) == 13:
                 current_emotions = forward_outputs[0]
                 belief_for_memory = forward_outputs[1] # Belief for this state
                 att_score_metric = forward_outputs[6] # Attention score
                 head_movement_logits = forward_outputs[-1] # HM logits
             else:
                 logger.error(f"Agent.step: Forward returned {len(forward_outputs)} items, expected 13. Using defaults.")
                 defaults = self._get_default_outputs(batch_size=1)
                 current_emotions = defaults[0]
                 belief_for_memory = defaults[1]
                 att_score_metric = 0.0
                 head_movement_logits = defaults[-1]

             # Predict head movement based on the logits for *this* state
             try:
                 if head_movement_logits.ndim == 1 and head_movement_logits.shape[0] == NUM_HEAD_MOVEMENTS:
                     predicted_hm_idx = torch.argmax(head_movement_logits).item()
                 elif head_movement_logits.ndim == 2 and head_movement_logits.shape[0] == 1 and head_movement_logits.shape[1] == NUM_HEAD_MOVEMENTS:
                     predicted_hm_idx = torch.argmax(head_movement_logits.squeeze(0)).item()
                 else:
                     logger.warning(f"Unexpected HM logits shape: {head_movement_logits.shape}. Defaulting to 'idle'.")
                     predicted_hm_idx = HEAD_MOVEMENT_TO_IDX["idle"]
                 predicted_hm_label = IDX_TO_HEAD_MOVEMENT.get(predicted_hm_idx, "idle")
             except Exception as e:
                 logger.error(f"Error predicting head movement in step: {e}"); predicted_hm_label = "idle"

             # Generate response using TransformerGPT
             try:
                 response_context = context if context else "..."; # Use provided context
                 temp = getattr(Config.NLP, 'GPT_TEMPERATURE', 0.7);
                 top_p = getattr(Config.NLP, 'GPT_TOP_P', 0.9);
                 # Call TransformerGPT generate
                 response = self.gpt.generate(
                     context=response_context,
                     emotions=current_emotions, # Pass emotions for potential use in prompt building inside generate
                     temperature=temp,
                     top_p=top_p
                 )
             except Exception as e:
                 logger.error(f"Error during GPT generation in step: {e}"); response = "..."

             # Update internal state history (using discretized version of input state)
             discretized_state = self.lattice.discretize(state); # Discretize the state we acted upon
             self.state_history_deque.append(discretized_state.clone().detach())

        # Note: self.prev_emotions was updated inside forward() for the single step
        # Return emotions, response text, belief vector, attention score, and predicted head movement label
        return current_emotions, response, belief_for_memory, att_score_metric, predicted_hm_label

    def learn(self, batch_size: int = Config.RL.AGENT_BATCH_SIZE) -> float:
        """Performs a learning update using a batch from memory."""
        self.train() # Ensure model is in training mode
        # Anneal PER beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        if len(self.memory) < batch_size:
            # logger.debug(f"Learn Skipped: Not enough memory ({len(self.memory)}/{batch_size})")
            return 0.0 # Not enough samples

        # Sample batch from memory using PER
        sample_result = self.memory.sample(batch_size, beta=self.beta)
        if sample_result is None:
            logger.warning("Learn: Memory sample returned None."); return 0.0
        batch_data, indices, weights = sample_result

        # Extract data and move to device
        states = batch_data['states'].to(DEVICE);
        rewards = batch_data['rewards'].to(DEVICE)
        next_states = batch_data['next_states'].to(DEVICE);
        dones = batch_data['dones'].to(DEVICE)
        # Extract target head movement indices from the sampled batch data
        target_hm_indices = batch_data.get('target_hm_idx').to(DEVICE) # This corresponds to the `state` in the batch

        current_batch_size = states.shape[0]
        if current_batch_size == 0: logger.warning("Learn: Sampled batch size is 0."); return 0.0

        # --- Get V(s) and HM(s) using ONLINE network ---
        try:
            # Pass zeros for reward, None for history in batch forward pass
            outputs_online = self.forward(states, torch.zeros_like(rewards), None, use_target=False)
            # Correctly unpack 13 items
            if len(outputs_online) != 13: raise ValueError(f"Online Forward returned {len(outputs_online)} items, expected 13.")
            current_value_pred = outputs_online[3]        # V(s) prediction
            self_consistency_batch = outputs_online[7]    # Metric tensor
            rho_score_batch = outputs_online[8]           # Metric tensor
            box_score_batch = outputs_online[9]           # Metric tensor
            head_movement_logits = outputs_online[-1]     # HM(s) prediction

            # Validate shapes and safety
            if not is_safe(current_value_pred) or current_value_pred.shape != (current_batch_size, 1): raise ValueError(f"Invalid Online V(s) shape/safety {current_value_pred.shape}")
            if not is_safe(head_movement_logits) or head_movement_logits.shape != (current_batch_size, NUM_HEAD_MOVEMENTS): raise ValueError(f"Invalid Online HM logits shape/safety {head_movement_logits.shape}")

            # Ensure metric tensors are valid
            rho_score_tensor = rho_score_batch.detach()
            box_score_tensor = box_score_batch.detach()
            if not is_safe(rho_score_tensor) or rho_score_tensor.shape != (current_batch_size, 1): raise ValueError("Invalid Rho score tensor")
            if not is_safe(box_score_tensor) or box_score_tensor.shape != (current_batch_size, 1): raise ValueError("Invalid Box score tensor")

        except Exception as e:
            logger.error(f"Error during Online Net calculation in learn: {e}", exc_info=True); return -1.0 # Indicate error

        # --- Get V(s') using TARGET network ---
        next_value_pred = torch.zeros_like(rewards) # Initialize with zeros
        with torch.no_grad(): # Ensure no gradients for target network forward pass
            try:
                # Use TARGET networks for next state value estimation
                outputs_target = self.forward(next_states, torch.zeros_like(rewards), None, use_target=True)
                # Correctly unpack 13 items
                if len(outputs_target) != 13: raise ValueError(f"Target Forward returned {len(outputs_target)} items, expected 13.")
                v_sp_target = outputs_target[3] # Value V(s') is the 4th item (index 3)

                # Validate target value prediction
                if is_safe(v_sp_target) and v_sp_target.shape == (current_batch_size, 1):
                    next_value_pred = v_sp_target
                else:
                    logger.warning(f"Learn: Invalid target next_value prediction shape/safety {v_sp_target.shape}. Using zeros.")

                # Set value of terminal states to 0
                dones_squeezed = dones.squeeze(-1) if dones.ndim > 1 else dones # Ensure dones is 1D bool tensor
                next_value_pred[dones_squeezed] = 0.0
            except Exception as e:
                logger.error(f"Error during Target V(s') calculation: {e}", exc_info=True)
                next_value_pred = torch.zeros_like(rewards) # Fallback to zeros on error

        # --- Calculate TD Target and Error ---
        # Calculate intrinsic rewards based on ONLINE network's metrics for state 's'
        intrinsic_reward_consistency = rho_score_tensor * Config.RL.INTRINSIC_REWARD_SCALE_CONSISTENCY
        intrinsic_reward_box = box_score_tensor * Config.RL.INTRINSIC_REWARD_SCALE_BOX
        # Optional: TD-error based intrinsic reward (often unstable, set scale to 0 in config if unused)
        td_error_abs_estimate = torch.abs(rewards + Config.RL.GAMMA * next_value_pred.detach() - current_value_pred.detach())
        intrinsic_reward_td_error = td_error_abs_estimate.clamp(max=1.0) * Config.RL.INTRINSIC_REWARD_SCALE_TD

        # Combine external and intrinsic rewards
        effective_rewards = rewards + (intrinsic_reward_consistency + intrinsic_reward_box + intrinsic_reward_td_error).detach()

        # Calculate TD Target: R + gamma * V_target(s')
        target_value = effective_rewards + Config.RL.GAMMA * next_value_pred # Uses V(s') from TARGET network

        # Calculate TD Error: Target - V_online(s)
        td_error = target_value - current_value_pred # Error drives learning and PER updates

        # --- Calculate Losses ---
        # 1. Value Loss (using PER weights)
        value_loss = torch.tensor(0.0, device=DEVICE)
        if current_value_pred.requires_grad:
             # Using Smooth L1 Loss (Huber loss)
            value_loss_elementwise = F.smooth_l1_loss(current_value_pred, target_value.detach(), reduction='none', beta=1.0)
            # Apply PER weights
            value_loss = (value_loss_elementwise * weights).mean()
        else:
            logger.warning("Learn: current_value_pred does not require grad. Value loss is 0.")

        # 2. Head Movement Loss (Supervised Cross-Entropy, weighted by PER weights)
        movement_loss = torch.tensor(0.0, device=DEVICE)
        hm_loss_weight = Config.RL.HEAD_MOVEMENT_LOSS_WEIGHT
        if hm_loss_weight > 0 and target_hm_indices is not None and head_movement_logits.requires_grad:
             if target_hm_indices.shape[0] == current_batch_size:
                 try:
                     # Ensure targets are 1D LongTensor
                     if target_hm_indices.ndim > 1: target_hm_indices = target_hm_indices.squeeze(-1)
                     target_hm_indices = target_hm_indices.long()

                     # Calculate cross-entropy loss per element
                     hm_loss_func = nn.CrossEntropyLoss(reduction='none')
                     hm_loss_elementwise = hm_loss_func(head_movement_logits, target_hm_indices)

                     # Apply PER weights (ensure weights are squeezed to match elementwise loss shape)
                     movement_loss = (hm_loss_elementwise * weights.squeeze(-1)).mean()

                     # Safety check for NaN/Inf loss
                     if not torch.isfinite(movement_loss):
                         logger.warning(f"Head movement loss is NaN/Inf! Setting to 0. Logits: {head_movement_logits.detach().cpu().numpy()}, Targets: {target_hm_indices.cpu().numpy()}")
                         movement_loss = torch.tensor(0.0, device=DEVICE)

                 except Exception as hm_err:
                     logger.error(f"Learn: Error calculating head movement loss: {hm_err}", exc_info=True);
                     movement_loss = torch.tensor(0.0, device=DEVICE)
             else:
                 logger.warning(f"Learn: Mismatch batch size ({current_batch_size}) vs target HM indices ({target_hm_indices.shape[0]}). Skipping HM loss.")
        elif hm_loss_weight > 0 and target_hm_indices is None:
             logger.debug("Learn: Target HM indices were None in the batch. Skipping HM loss.")


        # 3. Total Loss
        total_loss = value_loss + movement_loss * hm_loss_weight

        # --- Optimization Step ---
        original_lrs = [] # Store original LRs if using adaptive LR
        dynamic_lr = self._base_lr
        # Apply adaptive LR based on average rho score if enabled
        if Config.RL.ADAPTIVE_LR_ENABLED and rho_score_tensor.numel() > 0:
            avg_rho_score = rho_score_tensor.mean().item()
            min_factor = Config.RL.LR_ADAPTIVE_MIN_FACTOR; max_factor = Config.RL.LR_ADAPTIVE_MAX_FACTOR
            # Scale LR factor between min and max based on rho score
            dynamic_lr_factor = min_factor + (max_factor - min_factor) * max(0.0, min(1.0, avg_rho_score))
            dynamic_lr = self._base_lr * dynamic_lr_factor
            # Apply dynamic LR to optimizer param groups
            for i, param_group in enumerate(self.optimizer.param_groups):
                original_lrs.append(param_group['lr'])
                param_group['lr'] = dynamic_lr

        # Perform optimizer step if loss is valid and requires grad
        loss_val = total_loss.item()
        if is_safe(total_loss) and total_loss.requires_grad and abs(loss_val) > 1e-9: # Avoid tiny/zero gradients
             self.optimizer.zero_grad()
             total_loss.backward()
             # Clip gradients for ONLINE parameters only to prevent explosion
             online_params_with_grad = [p for pg in self.optimizer.param_groups for p in pg['params'] if p.grad is not None]
             if online_params_with_grad:
                 torch.nn.utils.clip_grad_norm_(online_params_with_grad, max_norm=Config.RL.GRADIENT_CLIP_AGENT)
             else:
                 logger.warning("Learn: No gradients found for online network after backward pass.")
             self.optimizer.step();

             # Update priorities in PER buffer using the calculated TD errors
             self.memory.update_priorities(indices, td_error.detach()) # Use TD error before intrinsic rewards

             # --- Update Target Network (using soft updates by default) ---
             self.soft_update_target_networks()

             # --- Optional: Hard update every N steps ---
             # if self.step_count % Config.RL.TARGET_NETWORK_UPDATE_FREQ == 0:
             #      self.update_target_network()

        elif not total_loss.requires_grad and abs(loss_val) > 1e-7 :
             logger.debug(f"Learn: Total Loss ({loss_val:.4f}) requires no grad. Skipping optimizer step.")
        elif not is_safe(total_loss):
             logger.warning(f"Learn: Unsafe total loss encountered ({loss_val:.4f}). Skipping optimizer step.");
             self.optimizer.zero_grad() # Zero grads even if step is skipped
        else:
             # Loss is likely zero or too small, skip optimizer step but zero grads
             self.optimizer.zero_grad()
             logger.debug(f"Learn: Skipping optimizer step due to zero/tiny loss ({loss_val:.4f}) or requires_grad=False.")

        # Restore original LRs if adaptive LR was used
        if Config.RL.ADAPTIVE_LR_ENABLED and original_lrs:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i < len(original_lrs): param_group['lr'] = original_lrs[i]

        self.step_count += 1 # Increment learn step counter

        return loss_val # Return the loss value for logging/monitoring

    # --- Save/Load Methods ---
    def save_state(self, agent_path: str = AGENT_SAVE_PATH, gpt_path: str = GPT_SAVE_PATH, optimizer_path: str = OPTIMIZER_SAVE_PATH, target_path_suffix: str = TARGET_NET_SAVE_SUFFIX):
        """Saves the state of the agent, GPT, optimizer, and target networks."""
        logger.info(f"Saving agent state to {agent_path}...")
        try:
            # Ensure directories exist
            os.makedirs(os.path.dirname(agent_path), exist_ok=True)
            # GPT path is a directory, handled by gpt.save_model
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)

            # Save Online networks state + agent metadata
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
                'beta': self.beta,             # Save PER beta
                'state_dim': self.state_dim,   # Save model's state dim for validation on load
            }
            torch.save(online_state, agent_path)
            logger.debug(f"Online agent state saved to {agent_path}")

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
            logger.debug(f"Target agent state saved to {target_agent_path}")

            # Save GPT separately (wrapper handles directory vs file)
            self.gpt.save_model(gpt_path) # gpt_path should be directory path

            # Save Optimizer
            torch.save(self.optimizer.state_dict(), optimizer_path)
            logger.debug(f"Optimizer state saved to {optimizer_path}")

            logger.info("Agent online, target, GPT, and optimizer states saved successfully.")
        except IOError as e: logger.error(f"IOError saving agent state: {e}", exc_info=True)
        except Exception as e: logger.error(f"Unexpected error saving agent state: {e}", exc_info=True)

    def load_state(self, agent_path: str = AGENT_SAVE_PATH, gpt_path: str = GPT_SAVE_PATH, optimizer_path: str = OPTIMIZER_SAVE_PATH, target_path_suffix: str = TARGET_NET_SAVE_SUFFIX) -> bool:
        """
        Loads the state of the agent, GPT, optimizer, and target networks.
        Returns True if model loading was successful, False otherwise.
        """
        logger.info(f"Loading agent state from {agent_path}...")
        target_agent_path = agent_path.replace(".pth", f"{target_path_suffix}.pth")

        if not os.path.exists(agent_path):
            logger.warning(f"Agent online state file not found: {agent_path}. Cannot load model state.")
            return False # Critical failure if main agent file is missing

        load_targets = os.path.exists(target_agent_path)
        if not load_targets:
            logger.warning(f"Agent target state file not found: {target_agent_path}. Will copy from online nets after loading.")

        try:
            # Load Online state
            agent_state = torch.load(agent_path, map_location=DEVICE)

            # --- Validate loaded state dimension ---
            loaded_state_dim = agent_state.get('state_dim')
            current_config_state_dim = Config.Agent.STATE_DIM
            if loaded_state_dim is not None and loaded_state_dim != current_config_state_dim:
                 logger.critical(f"CRITICAL LOAD ERROR: Saved model state dimension ({loaded_state_dim}) "
                                 f"does not match current configuration state dimension ({current_config_state_dim}). "
                                 f"Aborting load. Check configuration or use compatible save file.")
                 return False # Prevent loading incompatible state
            elif loaded_state_dim is None:
                 logger.warning("Save file missing 'state_dim'. Cannot verify dimension compatibility.")
            # --- End Validation ---

            # Load state dicts carefully
            self.korporator.load_state_dict(agent_state['korporator_state_dict'])
            self.kaskade.load_state_dict(agent_state['kaskade_state_dict'])
            self.emotional_module.load_state_dict(agent_state['emotional_module_state_dict'])
            self.feedback.load_state_dict(agent_state['feedback_state_dict'])
            self.value_head.load_state_dict(agent_state['value_head_state_dict'])
            # Load head movement head safely
            if 'head_movement_head_state_dict' in agent_state:
                self.head_movement_head.load_state_dict(agent_state['head_movement_head_state_dict'])
            else:
                 logger.warning("Head movement head state not found in online save file. Initializing fresh.")

            if self.attention and agent_state.get('attention_state_dict'):
                try:
                     self.attention.load_state_dict(agent_state['attention_state_dict'])
                except RuntimeError as attn_err:
                     logger.error(f"Failed to load attention state (likely incompatible dimensions): {attn_err}. Reinitializing attention.")
                     # Reinitialize attention if loading fails
                     possible_heads=[h for h in [16, 12, 8, 6, 4, 2, 1] if self.state_dim % h == 0]; num_heads = possible_heads[0] if possible_heads else 1
                     self.attention = nn.MultiheadAttention(embed_dim=self.state_dim, num_heads=num_heads, batch_first=True, dropout=0.15).to(DEVICE)

            # Load metadata
            self.prev_emotions = agent_state.get('prev_emotions', self.prev_emotions).to(DEVICE)
            self.step_count = agent_state.get('step_count', self.step_count) # Load learn step counter
            self.beta = agent_state.get('beta', self.beta) # Load PER beta
            logger.info("Online agent networks and metadata loaded.")

            # Load Target state
            if load_targets:
                target_state = torch.load(target_agent_path, map_location=DEVICE)
                self.target_korporator.load_state_dict(target_state['target_korporator_state_dict'])
                self.target_kaskade.load_state_dict(target_state['target_kaskade_state_dict'])
                self.target_value_head.load_state_dict(target_state['target_value_head_state_dict'])
                # Load target head movement head safely
                if 'target_head_movement_head_state_dict' in target_state:
                    self.target_head_movement_head.load_state_dict(target_state['target_head_movement_head_state_dict'])
                else:
                     logger.warning("Target head movement head state not found in target save file. Copying from loaded online head.")
                     self.target_head_movement_head.load_state_dict(self.head_movement_head.state_dict()) # Copy from loaded online head
                logger.info("Target agent networks loaded.")
            else: # If target file missing, copy from loaded online nets
                logger.warning("Target state file missing, performing hard copy from loaded online networks.")
                self.update_target_network()

            # Load GPT separately (wrapper handles directory vs file)
            gpt_loaded = self.gpt.load_model(gpt_path)
            if not gpt_loaded:
                 logger.warning(f"GPT model failed to load from {gpt_path}. Using base model {Config.NLP.HUGGINGFACE_MODEL}.")
                 # Re-initialize GPT to ensure it's usable with base model
                 try:
                     self.gpt = TransformerGPT()
                     logger.info("Re-initialized TransformerGPT with base model.")
                 except Exception as gpt_reinit_err:
                     logger.critical(f"CRITICAL: Failed to re-initialize TransformerGPT after load failure: {gpt_reinit_err}")
                     return False # Cannot proceed without a functional GPT


            # Load Optimizer state
            if os.path.exists(optimizer_path):
                 self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
                 logger.info("Optimizer state loaded.")
                 # Re-link parameters after loading state dicts to ensure optimizer targets current model params
                 online_params = list(self.korporator.parameters()) + list(self.kaskade.parameters()) + \
                                list(self.feedback.parameters()) + list(self.emotional_module.parameters()) + \
                                list(self.value_head.parameters()) + list(self.head_movement_head.parameters())
                 if self.attention: online_params.extend(list(self.attention.parameters()))
                 online_params = [p for p in online_params if p.requires_grad] # Ensure only trainable

                 # Clear existing param groups and add the correct one referencing CURRENT model parameters
                 self.optimizer.param_groups.clear()
                 self.optimizer.add_param_group({'params': online_params})

                 # Restore learning rate from config (state dict might save old/dynamic LR)
                 # Also apply to any potentially loaded optimizer state (like momentum buffers)
                 for group in self.optimizer.param_groups:
                     group['lr'] = Config.RL.LR
                 # If adaptive LR was used, the state might contain different LR values per parameter.
                 # Resetting ensures consistency with the current config after loading.
                 # self.optimizer.load_state_dict(self.optimizer.state_dict()) # Optional: Re-apply loaded state AFTER resetting LR

                 self._base_lr = Config.RL.LR # Also reset base LR tracker
                 logger.info("Optimizer parameters re-linked and LR reset to config value.")
            else:
                 logger.warning(f"Optimizer state file not found: {optimizer_path}. Optimizer not loaded (will start fresh).")


            logger.info("Agent model state loading complete.")
            self.eval() # Set online networks to evaluation mode after loading
            self.target_korporator.eval() # Ensure target networks are also in eval mode
            self.target_kaskade.eval()
            self.target_value_head.eval()
            self.target_head_movement_head.eval()
            return True # Indicate successful model load

        except FileNotFoundError:
             # Should be caught earlier, but as safety
             logger.error(f"Error: State file not found at {agent_path} or related paths.")
             return False
        except KeyError as e:
            logger.error(f"Error loading state: Missing key {e}. State file might be incompatible or corrupt.", exc_info=True)
            return False
        except RuntimeError as e:
             # Catches errors like size mismatches in load_state_dict
            logger.error(f"RuntimeError loading state dict (likely model structure mismatch): {e}. State file might be incompatible.", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading agent state: {e}", exc_info=True)
            return False

# --- END OF FILE agent.py ---
