# --- START OF FILE utils.py ---
import torch
import numpy as np
from collections import deque, namedtuple
import random
import logging
from typing import Optional, Dict, Tuple, List, Deque, Union, Any

from config import logger, DEVICE, MasterConfig as Config

# Experience tuple definition (remains same)
Experience = namedtuple('Experience', ['state', 'belief', 'reward', 'next_state', 'done', 'td_error'])

def is_safe(tensor: Optional[Union[torch.Tensor , np.ndarray]]) -> bool:
    """Checks if a tensor or numpy array contains NaN or Inf values."""
    if tensor is None: return False
    if isinstance(tensor, np.ndarray):
        # Check numpy array directly for NaNs/Infs
        if np.issubdtype(tensor.dtype, np.number): # Check only numeric types
            return not (np.isnan(tensor).any() or np.isinf(tensor).any())
        else:
            return True # Non-numeric arrays are considered safe
    elif isinstance(tensor, torch.Tensor):
        # Check PyTorch tensor
        if not torch.is_floating_point(tensor) and not torch.is_complex(tensor) and not hasattr(tensor, 'is_signed') : return True # Integer types are safe
        if tensor.numel() == 0: return True # Empty tensors are safe
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
             # isfinite checks for both NaN and Inf
             return torch.isfinite(tensor).all()
        return True # Should cover other types like boolean
    else:
        # Try converting other types (like lists/tuples) to tensor for checking
        try: tensor_conv = torch.tensor(tensor, device='cpu') # Check on CPU to avoid device errors
        except Exception: return False # Cannot convert, assume unsafe
        return is_safe(tensor_conv) # Recursively check the converted tensor


# --- MODIFIED: Lattice uses combined state dim by default ---
class MetronicLattice:
    """Discretizes continuous state space based on Syntrometrie concepts.
       Uses the agent's combined STATE_DIM by default.
    """
    def __init__(self, dim: int = Config.Agent.STATE_DIM, tau: float = Config.Agent.TAU): # Default uses combined STATE_DIM
        if not isinstance(dim, int) or dim <= 0:
            logger.error(f"Lattice invalid dimension {dim}. Using default 1.")
            dim = 1
        self.dim = dim
        if not isinstance(tau, (float, int)) or tau <= 1e-6:
            logger.warning(f"Lattice invalid tau value {tau}. Using default 0.1.")
            self.tau = 0.1
        else:
            self.tau = float(tau)
        logger.debug(f"MetronicLattice initialized with dim={self.dim}, tau={self.tau}")

    def discretize(self, x: torch.Tensor) -> torch.Tensor:
        """Discretizes a tensor to the nearest multiple of tau."""
        if not isinstance(x, torch.Tensor):
            try: x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
            except Exception as e: logger.error(f"Lattice discretize convert error: {e}"); return torch.zeros(self.dim, device=DEVICE, dtype=torch.float32)
        if x.device != DEVICE: x = x.to(DEVICE)

        was_single_instance = x.ndim == 1;
        x_batch = x if not was_single_instance else x.unsqueeze(0)

        # Check input dimension against self.dim
        if x_batch.ndim != 2 or x_batch.shape[1] != self.dim:
            logger.warning(f"Lattice discretize received unexpected shape {x.shape}. Expected (-1, {self.dim}). Returning zeros.")
            out_shape = (x_batch.shape[0], self.dim) if x_batch.ndim==2 else (1, self.dim) # Ensure batch dim for output
            return torch.zeros(out_shape, device=DEVICE, dtype=torch.float32).squeeze(0) if was_single_instance else torch.zeros(out_shape, device=DEVICE, dtype=torch.float32)

        if x_batch.dtype not in [torch.float32, torch.float64]: x_batch = x_batch.float()
        if not is_safe(x_batch): logger.warning("Lattice discretize received unsafe input. Using zeros."); x_batch = torch.zeros_like(x_batch)

        discretized_x_batch = torch.round(x_batch / self.tau) * self.tau
        if not is_safe(discretized_x_batch): logger.warning("Lattice discretize produced unsafe output. Using zeros."); discretized_x_batch = torch.zeros_like(x_batch)

        return discretized_x_batch.squeeze(0) if was_single_instance else discretized_x_batch

    def S(self, phi_history: torch.Tensor, n1: int, n2: int) -> torch.Tensor:
        """Calculates the Syntrop K structure vector S over a history slice."""
        # Check history dimension against self.dim
        if not isinstance(phi_history, torch.Tensor) or phi_history.ndim != 2 or phi_history.shape[1] != self.dim:
            logger.warning(f"Lattice S received invalid history shape {phi_history.shape}. Expected (HistorySize, {self.dim}). Returning zeros.");
            return torch.zeros(self.dim, device=DEVICE)

        history_len = phi_history.shape[0];
        if history_len == 0: return torch.zeros(self.dim, device=DEVICE) # Handle empty history

        n1_clamped = max(0, min(n1, history_len - 1));
        n2_clamped = max(n1_clamped, min(n2, history_len - 1)) # Ensure n2 >= n1 after clamping

        if n1_clamped > n2_clamped: return torch.zeros(self.dim, device=DEVICE) # Should not happen with above clamp

        history_slice = phi_history[n1_clamped : n2_clamped + 1]
        if history_slice.numel() == 0: return torch.zeros(self.dim, device=DEVICE) # Handle empty slice

        if not is_safe(history_slice): logger.warning("Lattice S processing unsafe history slice. Returning zeros."); return torch.zeros(self.dim, device=DEVICE)

        summed_states = torch.sum(history_slice.float(), dim=0) # Ensure float for sum
        if not is_safe(summed_states): logger.warning("Lattice S calculation resulted in unsafe sum. Returning zeros."); return torch.zeros(self.dim, device=DEVICE)

        return summed_states


# --- MetaCognitiveMemory (Stores CPU tensors, handles belief optionally) ---
class MetaCognitiveMemory:
    INITIAL_TD_ERROR = 1.0 # Default TD error for new experiences before calculation
    def __init__(self, capacity: int = Config.Agent.MEMORY_SIZE):
        self.capacity = max(10, capacity) # Ensure minimum capacity
        self.short_term: Deque[Experience] = deque(maxlen=30) # Short-term buffer (optional use)
        self.long_term: Deque[Experience] = deque(maxlen=self.capacity) # Main prioritized replay buffer
        self.priorities: Deque[float] = deque(maxlen=self.capacity) # Stores priorities corresponding to long_term

    def add(self, experience: Experience):
        """Adds an experience to the memory, converting tensors to CPU and calculating priority."""
        if not isinstance(experience, Experience):
             logger.warning(f"Memory.add: Received invalid type {type(experience)}. Expected Experience namedtuple."); return

        priority = abs(experience.td_error) + 1e-5 # Use provided priority (TD error + epsilon)

        try:
            # --- Validate and Convert to CPU Tensors ---
            state=experience.state; belief=experience.belief; reward=experience.reward; next_state=experience.next_state; done=experience.done; td_error=experience.td_error

            # State (Must be Tensor)
            if not isinstance(state, torch.Tensor): raise TypeError("State must be a Tensor")
            safe_state = state.detach().clone().cpu()
            if not is_safe(safe_state): raise ValueError("Unsafe state tensor")
            # Check dimension consistency (optional, but helpful for debugging)
            if safe_state.shape[0] != Config.Agent.STATE_DIM:
                 logger.warning(f"Memory Add: State dimension mismatch ({safe_state.shape[0]} vs {Config.Agent.STATE_DIM}). Check where experience is created.")


            # Belief (Optional Tensor)
            safe_belief = None
            if belief is not None:
                 if not isinstance(belief, torch.Tensor): raise TypeError("Belief must be a Tensor or None")
                 safe_belief = belief.detach().clone().cpu()
                 if not is_safe(safe_belief): raise ValueError("Unsafe belief tensor")
                 # Optional: Check belief dim consistency with hidden_dim?

            # Reward (Float)
            safe_reward = float(reward);

            # Next State (Must be Tensor)
            if not isinstance(next_state, torch.Tensor): raise TypeError("Next State must be a Tensor")
            safe_next_state = next_state.detach().clone().cpu()
            if not is_safe(safe_next_state): raise ValueError("Unsafe next_state tensor")
            if safe_next_state.shape[0] != Config.Agent.STATE_DIM:
                 logger.warning(f"Memory Add: Next State dimension mismatch ({safe_next_state.shape[0]} vs {Config.Agent.STATE_DIM}).")


            # Done (Bool)
            safe_done = bool(done);

            # TD Error (Float - already used for priority)
            safe_td_error = float(td_error)

            # Create safe experience tuple with CPU tensors
            safe_exp = Experience(safe_state, safe_belief, safe_reward, safe_next_state, safe_done, safe_td_error)

        except (ValueError, TypeError, AttributeError) as e:
             logger.error(f"Memory.add: Error processing/validating experience: {e}. Experience not added."); return
        except Exception as e:
             logger.error(f"Memory.add: Unexpected error processing experience: {e}. Experience not added."); return

        # Add to short-term buffer (always)
        self.short_term.append(safe_exp)

        # Add to long-term buffer and handle capacity limit
        if len(self.long_term) >= self.capacity:
             self.long_term.popleft()
             self.priorities.popleft()
        self.long_term.append(safe_exp);
        self.priorities.append(priority) # Store the calculated priority

    def __len__(self) -> int:
        """Returns the number of experiences in the long-term memory."""
        return len(self.long_term)

    def sample(self, batch_size: int, beta: float = Config.RL.PER_BETA_START) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
         """Samples a batch of experiences using Prioritized Experience Replay (PER)."""
         if len(self.long_term) < batch_size: return None # Not enough samples

         # Calculate sampling probabilities based on stored priorities
         priorities_float = np.array([float(p) for p in self.priorities], dtype=np.float64)
         if np.sum(priorities_float) <= 0 :
             # If all priorities are zero, use uniform sampling
             probs = np.ones_like(priorities_float) / len(priorities_float) if len(priorities_float) > 0 else None;
             logger.debug("Memory Sample: Zero priorities detected. Using uniform probabilities.")
         else:
             # Apply PER alpha parameter
             probs = priorities_float ** Config.RL.PER_ALPHA;
             probs /= probs.sum() # Normalize probabilities

         if probs is None or not np.isclose(probs.sum(), 1.0):
              logger.error(f"Memory Sample: Invalid probabilities (Sum={probs.sum() if probs is not None else 'None'}). Check priorities/alpha. Using uniform fallback.");
              probs = np.ones_like(priorities_float) / len(priorities_float) if len(priorities_float) > 0 else None;
         if probs is None: return None # Still failed

         # Sample indices based on calculated probabilities
         try: indices = np.random.choice(len(self.long_term), batch_size, p=probs, replace=False) # Sample without replacement
         except ValueError as e: logger.error(f"Memory Sample: Error during np.random.choice (check probs sum?): {e}"); return None

         # Calculate Importance Sampling (IS) weights
         total_samples = len(self.long_term);
         weights = (total_samples * probs[indices]) ** (-beta); # Apply PER beta parameter
         weights /= weights.max(); # Normalize weights for stability
         weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)

         # Gather sampled experiences and convert back to device tensors
         states_list, beliefs_list, rewards_list, next_states_list, dones_list = [], [], [], [], [];
         valid_count = 0; skipped_count = 0
         default_belief = torch.zeros(Config.Agent.HIDDEN_DIM, device=DEVICE) # Pre-create default belief

         for idx in indices:
             exp = self.long_term[idx]
             try:
                 # Move tensors to the target device
                 state_dev = exp.state.to(DEVICE);
                 belief_dev = exp.belief.to(DEVICE) if exp.belief is not None else default_belief # Use default if None
                 next_state_dev = exp.next_state.to(DEVICE)

                 # Final safety check on device tensors (optional but safer)
                 if not is_safe(state_dev) or not is_safe(next_state_dev) or not is_safe(belief_dev):
                     logger.warning(f"Memory Sample: Unsafe tensor detected on device for index {idx}. Skipping.");
                     skipped_count += 1; continue

                 # Append tensors to lists
                 states_list.append(state_dev);
                 beliefs_list.append(belief_dev);
                 rewards_list.append(exp.reward);
                 next_states_list.append(next_state_dev);
                 dones_list.append(exp.done);
                 valid_count += 1
             except Exception as e:
                 logger.error(f"Memory Sample: Error processing experience index {idx} during device transfer: {e}");
                 skipped_count += 1

         if valid_count == 0: logger.warning("Memory Sample: No valid experiences found in the sampled batch."); return None
         if skipped_count > 0: logger.debug(f"Memory Sample: Skipped {skipped_count} experiences during batch creation.")

         # Stack lists into batch tensors
         try:
             states_batch=torch.stack(states_list);
             beliefs_batch=torch.stack(beliefs_list); # Beliefs are now guaranteed to be tensors
             rewards_batch=torch.tensor(rewards_list,dtype=torch.float32,device=DEVICE).unsqueeze(1);
             next_states_batch=torch.stack(next_states_list);
             dones_batch=torch.tensor(dones_list,dtype=torch.bool,device=DEVICE).unsqueeze(1)

             # Final check on stacked batch tensors
             if not is_safe(states_batch) or not is_safe(beliefs_batch) or not is_safe(rewards_batch) or not is_safe(next_states_batch) or not is_safe(dones_batch):
                 logger.error("Memory Sample: Unsafe tensor detected after stacking batch."); return None

             final_batch_dict = { 'states': states_batch, 'beliefs': beliefs_batch, 'rewards': rewards_batch, 'next_states': next_states_batch, 'dones': dones_batch }
             indices_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE) # Indices for priority update

             return final_batch_dict, indices_tensor, weights_tensor
         except Exception as e: logger.error(f"Memory Sample: Error stacking batch tensors: {e}"); return None

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Updates the priorities of sampled experiences based on their TD errors."""
        if not isinstance(indices, torch.Tensor) or not isinstance(td_errors, torch.Tensor):
             logger.warning(f"update_priorities: Invalid input types ({type(indices)}, {type(td_errors)})."); return
        if indices.numel() == 0 or td_errors.numel() == 0 or indices.shape[0] != td_errors.shape[0]:
             logger.warning(f"update_priorities: Mismatch/empty tensors (Indices: {indices.shape}, Errors: {td_errors.shape}). Cannot update."); return

        # Calculate new priorities based on absolute TD error + epsilon
        new_priorities_raw = torch.abs(td_errors).cpu().numpy().flatten() + 1e-5
        # Apply PER alpha (already applied during sampling for weights, but store raw priority adjusted by alpha for next sampling)
        # Storing p^a directly simplifies sampling probability calculation next time.
        new_priorities_final = new_priorities_raw ** Config.RL.PER_ALPHA

        np_indices = indices.cpu().numpy() # Convert indices to numpy for easier iteration
        updated_count = 0
        skipped_count = 0
        for i, idx in enumerate(np_indices):
            if 0 <= idx < len(self.priorities):
                 self.priorities[idx] = new_priorities_final[i]; # Update priority at the specific index
                 updated_count += 1
            else:
                 # This should ideally not happen if sampling is correct
                 logger.warning(f"update_priorities: Invalid index {idx} encountered (Memory size: {len(self.priorities)}). Skipping update for this index.")
                 skipped_count += 1
        # logger.debug(f"Priorities updated for {updated_count} indices. Skipped {skipped_count}.")

    def get_belief_norm(self, memory_deque: Deque[Experience]) -> float:
        """Calculates the average L2 norm of belief vectors in a given deque."""
        valid_beliefs = [exp.belief for exp in memory_deque if exp.belief is not None and isinstance(exp.belief, torch.Tensor) and exp.belief.numel() > 0 and is_safe(exp.belief)]
        if not valid_beliefs: return 0.0
        try:
             # Calculate norms on CPU to avoid potential GPU memory issues with many small tensors
             individual_norms = [torch.linalg.norm(b.float().cpu()).item() for b in valid_beliefs];
             return sum(individual_norms) / len(individual_norms) if individual_norms else 0.0
        except Exception as e: logger.error(f"Error calculating belief norm: {e}"); return 0.0

    def get_short_term_norm(self) -> float:
        """Calculates the average belief norm in short-term memory."""
        return self.get_belief_norm(self.short_term)

    def get_long_term_norm(self) -> float:
        """Calculates the average belief norm in long-term memory."""
        return self.get_belief_norm(self.long_term)

# --- END OF FILE utils.py ---
