# --- START OF FILE utils.py ---
import torch
import numpy as np
from collections import deque, namedtuple
import random
import logging
import sys
import os
import re # Keep re if used elsewhere, otherwise remove
from typing import Optional, Dict, Tuple, List, Deque, Union, Any # Keep typing

# --- REMOVED: BPE Tokenizer Imports and Flag ---

# Keep necessary config imports (ensure NLPConfig is NOT imported if only BPE used it)
from config import logger, DEVICE, MasterConfig as Config
# Import Head Movement Labels
from config import HEAD_MOVEMENT_TO_IDX # Needed for default index

# --- MODIFIED: Add head_movement_idx ---
Experience = namedtuple('Experience', [
    'state',            # Combined state (CPU Tensor)
    'belief',           # Belief vector (Optional CPU Tensor)
    'reward',           # Scalar reward (float)
    'next_state',       # Next combined state (CPU Tensor)
    'done',             # Done flag (bool)
    'td_error',         # Priority (float, calculated from TD error + attention)
    'head_movement_idx' # Target head movement index for state (int) - NEW FIELD
])

def is_safe(tensor: Optional[Union[torch.Tensor , np.ndarray]]) -> bool:
    """Checks if a tensor or numpy array contains NaN or Inf values."""
    if tensor is None: return False
    if isinstance(tensor, np.ndarray):
        if np.issubdtype(tensor.dtype, np.number):
            return not (np.isnan(tensor).any() or np.isinf(tensor).any())
        else: return True # Non-numeric numpy arrays are considered safe
    elif isinstance(tensor, torch.Tensor):
        # Check for floating point types before calling isfinite
        if tensor.is_floating_point() or tensor.is_complex():
            if tensor.numel() == 0: return True # Empty tensor is safe
            return torch.isfinite(tensor).all()
        return True # Non-floating point tensors are considered safe
    else:
        try: tensor_conv = torch.tensor(tensor, device='cpu') # Try converting to tensor
        except Exception: return False # Conversion failed
        return is_safe(tensor_conv) # Recursive call


class MetronicLattice:
    """Discretizes continuous state space based on Syntrometrie concepts."""
    def __init__(self, dim: int = Config.Agent.STATE_DIM, tau: float = Config.Agent.TAU):
        if not isinstance(dim, int) or dim <= 0: logger.error(f"Lattice invalid dimension {dim}. Using default 1."); dim = 1
        self.dim = dim
        if not isinstance(tau, (float, int)) or tau <= 1e-6: logger.warning(f"Lattice invalid tau value {tau}. Using default 0.1."); self.tau = 0.1
        else: self.tau = float(tau)
        #logger.debug(f"MetronicLattice initialized with dim={self.dim}, tau={self.tau}") # Can be noisy

    def discretize(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            try: x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
            except Exception as e: logger.error(f"Lattice discretize convert error: {e}"); return torch.zeros(self.dim, device=DEVICE, dtype=torch.float32)
        if x.device != DEVICE: x = x.to(DEVICE)
        was_single_instance = x.ndim == 1; x_batch = x if not was_single_instance else x.unsqueeze(0)
        if x_batch.ndim != 2 or x_batch.shape[1] != self.dim:
            logger.warning(f"Lattice discretize received unexpected shape {x.shape}. Expected (-1, {self.dim}). Returning zeros.")
            out_shape = (x_batch.shape[0], self.dim) if x_batch.ndim==2 else (1, self.dim)
            return torch.zeros(out_shape, device=DEVICE, dtype=torch.float32).squeeze(0) if was_single_instance else torch.zeros(out_shape, device=DEVICE, dtype=torch.float32)
        if x_batch.dtype not in [torch.float32, torch.float64]: x_batch = x_batch.float()
        if not is_safe(x_batch): logger.warning("Lattice discretize received unsafe input. Using zeros."); x_batch = torch.zeros_like(x_batch)
        discretized_x_batch = torch.round(x_batch / self.tau) * self.tau
        if not is_safe(discretized_x_batch): logger.warning("Lattice discretize produced unsafe output. Using zeros."); discretized_x_batch = torch.zeros_like(x_batch)
        return discretized_x_batch.squeeze(0) if was_single_instance else discretized_x_batch

    def S(self, phi_history: torch.Tensor, n1: int, n2: int) -> torch.Tensor:
        if not isinstance(phi_history, torch.Tensor) or phi_history.ndim != 2 or phi_history.shape[1] != self.dim: logger.warning(f"Lattice S received invalid history shape {phi_history.shape}. Expected (HistorySize, {self.dim}). Returning zeros."); return torch.zeros(self.dim, device=DEVICE)
        history_len = phi_history.shape[0];
        if history_len == 0: return torch.zeros(self.dim, device=DEVICE)
        n1_clamped = max(0, min(n1, history_len - 1)); n2_clamped = max(n1_clamped, min(n2, history_len - 1))
        if n1_clamped > n2_clamped: return torch.zeros(self.dim, device=DEVICE)
        history_slice = phi_history[n1_clamped : n2_clamped + 1]
        if history_slice.numel() == 0: return torch.zeros(self.dim, device=DEVICE)
        if not is_safe(history_slice): logger.warning("Lattice S processing unsafe history slice. Returning zeros."); return torch.zeros(self.dim, device=DEVICE)
        summed_states = torch.sum(history_slice.float(), dim=0)
        if not is_safe(summed_states): logger.warning("Lattice S calculation resulted in unsafe sum. Returning zeros."); return torch.zeros(self.dim, device=DEVICE)
        return summed_states


class MetaCognitiveMemory:
    """Stores experiences and samples them using Prioritized Experience Replay."""
    INITIAL_TD_ERROR = 1.0 # Default priority for new experiences
    def __init__(self, capacity: int = Config.Agent.MEMORY_SIZE):
        self.capacity = max(10, capacity)
        self.short_term: Deque[Experience] = deque(maxlen=30) # For potential short-term specific logic
        self.long_term: Deque[Experience] = deque(maxlen=self.capacity) # Main replay buffer
        self.priorities: Deque[float] = deque(maxlen=self.capacity) # Stores priorities for PER
        # Get default index for 'idle' once
        self.idle_hm_idx = HEAD_MOVEMENT_TO_IDX.get("idle", 0)

    def add(self, experience: Experience):
        """Adds an experience to memory, validating and converting to CPU."""
        if not isinstance(experience, Experience):
             logger.warning(f"Memory.add: Invalid type {type(experience)}."); return

        priority = abs(experience.td_error) + 1e-5 # Use calculated priority, ensure > 0

        try:
            # Extract and validate components
            state=experience.state; belief=experience.belief; reward=experience.reward
            next_state=experience.next_state; done=experience.done
            td_error=experience.td_error; hm_idx=experience.head_movement_idx # Keep original priority value

            # Validate state
            if not isinstance(state, torch.Tensor): raise TypeError("State must be a Tensor")
            safe_state = state.detach().clone().cpu() # Store on CPU
            if not is_safe(safe_state): raise ValueError("Unsafe state tensor")
            if safe_state.shape[0] != Config.Agent.STATE_DIM: logger.warning(f"Mem Add: State dim mismatch ({safe_state.shape[0]} vs {Config.Agent.STATE_DIM}).")

            # Validate belief (optional)
            safe_belief = None
            if belief is not None:
                 if not isinstance(belief, torch.Tensor): raise TypeError("Belief must be a Tensor or None")
                 safe_belief = belief.detach().clone().cpu() # Store on CPU
                 if not is_safe(safe_belief): raise ValueError("Unsafe belief tensor")

            # Validate next_state
            if not isinstance(next_state, torch.Tensor): raise TypeError("Next State must be a Tensor")
            safe_next_state = next_state.detach().clone().cpu() # Store on CPU
            if not is_safe(safe_next_state): raise ValueError("Unsafe next_state tensor")
            if safe_next_state.shape[0] != Config.Agent.STATE_DIM: logger.warning(f"Mem Add: Next State dim mismatch ({safe_next_state.shape[0]} vs {Config.Agent.STATE_DIM}).")

            # Validate hm_idx (must be int)
            safe_hm_idx = int(hm_idx)

            # Validate others
            safe_reward = float(reward); safe_done = bool(done); safe_td_error = float(td_error) # Store original priority

            # Create safe experience tuple with CPU tensors and validated types
            safe_exp = Experience(safe_state, safe_belief, safe_reward, safe_next_state, safe_done, safe_td_error, safe_hm_idx)

        except (ValueError, TypeError, AttributeError) as e:
             logger.error(f"Memory.add: Error processing/validating experience: {e}. Exp: {experience}. Not added."); return
        except Exception as e:
             logger.error(f"Memory.add: Unexpected error processing experience: {e}. Not added."); return

        # Add to buffers
        self.short_term.append(safe_exp) # Add to short term buffer too
        if len(self.long_term) >= self.capacity:
             self.long_term.popleft()    # Remove oldest experience if full
             self.priorities.popleft()   # Remove corresponding priority
        self.long_term.append(safe_exp);
        self.priorities.append(priority) # Store the calculated priority

    def __len__(self) -> int:
        """Return the current size of samples in the long-term memory."""
        return len(self.long_term)

    def sample(self, batch_size: int, beta: float = Config.RL.PER_BETA_START) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
         """Samples a batch using Prioritized Experience Replay (PER)."""
         if len(self.long_term) < batch_size:
             # logger.debug(f"Memory Sample: Not enough samples ({len(self.long_term)}/{batch_size}).")
             return None

         # Calculate probabilities based on priorities
         priorities_float = np.array([float(p) for p in self.priorities], dtype=np.float64)
         if np.sum(priorities_float) <= 1e-9 : # Use small epsilon for sum check
             # Fallback to uniform sampling if priorities are zero or negative
             probs = np.ones_like(priorities_float) / len(priorities_float) if len(priorities_float) > 0 else None;
             # logger.warning("Memory Sample: Sum of priorities is near zero. Using uniform probabilities.")
         else:
             # Apply PER alpha
             probs = priorities_float ** Config.RL.PER_ALPHA;
             probs /= probs.sum() # Normalize probabilities

         # Final check on probabilities
         if probs is None or not np.isclose(probs.sum(), 1.0, atol=1e-7):
              logger.error(f"Memory Sample: Invalid probabilities (Sum={probs.sum() if probs is not None else 'None'}). Using uniform fallback.");
              probs = np.ones_like(priorities_float) / len(priorities_float) if len(priorities_float) > 0 else None;
         if probs is None: return None # Cannot sample if no valid probabilities

         # Sample indices based on calculated probabilities
         try:
             indices = np.random.choice(len(self.long_term), batch_size, p=probs, replace=False) # No replacement for unique samples
         except ValueError as e:
             logger.error(f"Memory Sample: Error during np.random.choice (check probs sum? Size: {len(self.long_term)}, BS: {batch_size}): {e}");
             return None

         # Calculate Importance Sampling (IS) weights
         total_samples = len(self.long_term);
         weights = (total_samples * probs[indices]) ** (-beta);
         weights /= weights.max(); # Normalize weights for stability
         weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1) # Add dimension for broadcasting

         # Retrieve experiences for sampled indices
         states_list, beliefs_list, rewards_list, next_states_list, dones_list, head_movement_idx_list = [], [], [], [], [], []
         valid_count = 0; skipped_count = 0
         # Dynamically determine expected belief dim from agent config or first sample
         expected_belief_dim = Config.Agent.HIDDEN_DIM # Use config as default
         # Find first valid belief to confirm dim
         first_valid_belief = next((exp.belief for exp in [self.long_term[i] for i in indices] if exp.belief is not None), None)
         if first_valid_belief is not None:
             expected_belief_dim = first_valid_belief.shape[0]
         # Pre-create default belief with correct dim and device
         default_belief = torch.zeros(expected_belief_dim, device=DEVICE, dtype=torch.float32)

         for idx in indices:
             exp = self.long_term[idx]
             try:
                 # Move CPU tensors from memory to the target DEVICE for batching
                 state_dev = exp.state.to(DEVICE);
                 # Handle belief: use default if None or shape mismatch
                 if exp.belief is not None and exp.belief.shape[0] == expected_belief_dim:
                     belief_dev = exp.belief.to(DEVICE)
                 else:
                     if exp.belief is not None: logger.debug(f"Belief shape mismatch for index {idx}. Got {exp.belief.shape}, expected ({expected_belief_dim},). Using default.")
                     belief_dev = default_belief # Use pre-created default tensor

                 next_state_dev = exp.next_state.to(DEVICE)
                 hm_idx = exp.head_movement_idx # Retrieve index

                 # Check safety AFTER moving to device
                 if not is_safe(state_dev) or not is_safe(next_state_dev) or not is_safe(belief_dev):
                     logger.warning(f"Memory Sample: Unsafe tensor detected on device for index {idx}. Skipping."); skipped_count += 1; continue

                 # Append validated device tensors and other data
                 states_list.append(state_dev); beliefs_list.append(belief_dev); rewards_list.append(exp.reward); next_states_list.append(next_state_dev); dones_list.append(exp.done);
                 head_movement_idx_list.append(hm_idx) # Add index to its list
                 valid_count += 1
             except Exception as e:
                 logger.error(f"Memory Sample: Error processing experience index {idx}: {e}"); skipped_count += 1

         if valid_count == 0:
             logger.warning("Memory Sample: No valid experiences found in the sampled batch."); return None
         if skipped_count > 0:
             logger.debug(f"Memory Sample: Skipped {skipped_count}/{batch_size} experiences during batch creation.")

         # Stack lists into batch tensors
         try:
             states_batch=torch.stack(states_list);
             beliefs_batch=torch.stack(beliefs_list);
             rewards_batch=torch.tensor(rewards_list,dtype=torch.float32,device=DEVICE).unsqueeze(1);
             next_states_batch=torch.stack(next_states_list);
             dones_batch=torch.tensor(dones_list,dtype=torch.bool,device=DEVICE).unsqueeze(1) # Use bool for dones
             head_movement_idx_batch = torch.tensor(head_movement_idx_list, dtype=torch.long, device=DEVICE) # Target indices should be Long

             # Final safety check on stacked tensors
             if not is_safe(states_batch) or not is_safe(beliefs_batch) or not is_safe(rewards_batch) or not is_safe(next_states_batch) or not is_safe(dones_batch) or not is_safe(head_movement_idx_batch):
                 logger.error("Memory Sample: Unsafe tensor detected after stacking batch."); return None

             # Prepare dictionary for return
             final_batch_dict = {
                 'states': states_batch,
                 'beliefs': beliefs_batch, # Include beliefs, even if default
                 'rewards': rewards_batch,
                 'next_states': next_states_batch,
                 'dones': dones_batch,
                 'target_hm_idx': head_movement_idx_batch # Add target HM indices
             }
             indices_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE) # Indices used for priority updates

             return final_batch_dict, indices_tensor, weights_tensor
         except Exception as e:
             logger.error(f"Memory Sample: Error stacking batch tensors: {e}"); return None

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Updates priorities for sampled indices based on new TD errors."""
        if not isinstance(indices, torch.Tensor) or not isinstance(td_errors, torch.Tensor):
            logger.warning(f"update_priorities: Invalid types ({type(indices)}, {type(td_errors)})."); return
        if indices.numel() == 0 or td_errors.numel() == 0 or indices.shape[0] != td_errors.shape[0]:
            logger.warning(f"update_priorities: Mismatch/empty tensors (Indices: {indices.shape}, Errors: {td_errors.shape})."); return

        # Use absolute TD error for priority update, add epsilon
        new_priorities_raw = torch.abs(td_errors).cpu().numpy().flatten() + 1e-5
        # Apply alpha factor for sampling probability calculation (stored priority)
        new_priorities_final = new_priorities_raw ** Config.RL.PER_ALPHA

        np_indices = indices.cpu().numpy()
        updated_count = 0; skipped_count = 0
        for i, idx in enumerate(np_indices):
            if 0 <= idx < len(self.priorities):
                 self.priorities[idx] = new_priorities_final[i]; updated_count += 1
            else:
                 logger.warning(f"update_priorities: Invalid index {idx} encountered (Memory size: {len(self.priorities)}). Skipping update for this index."); skipped_count += 1
        # if skipped_count > 0: logger.debug(f"Priorities updated for {updated_count} indices. Skipped {skipped_count}.")

    def get_belief_norm(self, memory_deque: Deque[Experience]) -> float:
        """Calculates the average L2 norm of belief vectors in a deque."""
        valid_beliefs = [exp.belief for exp in memory_deque if exp.belief is not None and isinstance(exp.belief, torch.Tensor) and exp.belief.numel() > 0 and is_safe(exp.belief)]
        if not valid_beliefs: return 0.0
        try:
             # Ensure tensors are float and on CPU for norm calculation
             individual_norms = [torch.linalg.norm(b.float().cpu()).item() for b in valid_beliefs];
             return sum(individual_norms) / len(individual_norms) if individual_norms else 0.0
        except Exception as e:
             logger.error(f"Error calculating belief norm: {e}"); return 0.0

    def get_short_term_norm(self) -> float:
        """Calculates the average L2 norm of beliefs in short-term memory."""
        return self.get_belief_norm(self.short_term)

    def get_long_term_norm(self) -> float:
        """Calculates the average L2 norm of beliefs in long-term memory."""
        return self.get_belief_norm(self.long_term)

# --- REMOVED: BPE Tokenizer Variables and Functions ---

# --- END OF FILE utils.py ---
