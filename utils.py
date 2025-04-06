# --- START OF FILE utils.py ---
import torch
import numpy as np
from collections import deque, namedtuple
import random
import logging
from typing import Optional, Dict, Tuple

# Use updated Config including PER parameters
from config import logger, DEVICE, Config

# Experience Namedtuple definition
Experience = namedtuple('Experience', ['state', 'belief', 'reward', 'next_state', 'done', 'td_error'])

# --- Helper Function (is_safe) ---
def is_safe(tensor):
    if tensor is None: return False
    if not isinstance(tensor, torch.Tensor):
        try: tensor = torch.tensor(tensor, device='cpu')
        except Exception: return False
    if tensor.numel() == 0: return True
    return torch.isfinite(tensor).all()

# --- Metronic Lattice (Optimized batch handling) ---
class MetronicLattice:
    """Discretizes continuous state space based on Syntrometrie concepts."""
    def __init__(self, dim=Config.STATE_DIM, tau=Config.TAU):
        if not isinstance(dim, int) or dim <= 0: logger.error(f"Lattice dim err {dim}"); dim = 1
        self.dim = dim
        if not isinstance(tau, (float, int)) or tau <= 1e-6: logger.warning(f"Lattice tau err {tau}"); self.tau = 0.1
        else: self.tau = float(tau)

    def discretize(self, x):
        """Applies discretization: x_discrete = round(x / tau) * tau. Handles single instance or batch."""
        if not isinstance(x, torch.Tensor):
            try: x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
            except Exception as e: logger.error(f"Lattice convert err: {e}"); return torch.zeros(self.dim, device=DEVICE, dtype=torch.float32)

        if x.device != DEVICE: x = x.to(DEVICE)

        was_single_instance = x.ndim == 1
        x_batch = x if not was_single_instance else x.unsqueeze(0) # Ensure batch dim

        # --- Input Shape Validation ---
        # Expect input shape (batch_size, self.dim)
        if x_batch.ndim != 2 or x_batch.shape[1] != self.dim:
            logger.warning(f"Lattice discretize received unexpected shape {x.shape}. Expected (-1, {self.dim}). Returning zeros.")
            # Return zeros matching input batch size if possible, otherwise single default
            out_shape = (x.shape[0], self.dim) if x.ndim==2 else (self.dim,)
            return torch.zeros(out_shape, device=DEVICE, dtype=torch.float32)

        if x_batch.dtype not in [torch.float32, torch.float64]: x_batch = x_batch.float()
        if not is_safe(x_batch): logger.warning("Lattice unsafe input."); x_batch = torch.zeros_like(x_batch)

        # --- Vectorized Discretization ---
        discretized_x_batch = torch.round(x_batch / self.tau) * self.tau

        if not is_safe(discretized_x_batch): logger.warning("Lattice unsafe output."); discretized_x_batch = torch.zeros_like(x_batch)

        # Return matching original dimension (single or batch)
        return discretized_x_batch.squeeze(0) if was_single_instance else discretized_x_batch

    def S(self, phi_history, n1, n2):
        """Calculates the structural measure S (sum) over a slice of history phi."""
        # ... (S method remains the same) ...
        if not isinstance(phi_history, torch.Tensor) or phi_history.ndim != 2 or phi_history.shape[1] != self.dim: logger.warning("Lattice S shape err."); return torch.zeros(self.dim, device=DEVICE)
        history_len = phi_history.shape[0];
        if history_len == 0: return torch.zeros(self.dim, device=DEVICE)
        n1_clamped = max(0, min(n1, history_len)); n2_clamped = max(0, min(n2, history_len - 1))
        if n1_clamped > n2_clamped: return torch.zeros(self.dim, device=DEVICE)
        history_slice = phi_history[n1_clamped : n2_clamped + 1]
        if history_slice.numel() == 0: return torch.zeros(self.dim, device=DEVICE)
        if not is_safe(history_slice): logger.warning("Lattice S unsafe slice."); return torch.zeros(self.dim, device=DEVICE)
        summed_states = torch.sum(history_slice, dim=0)
        if not is_safe(summed_states): logger.warning("Lattice S unsafe sum."); return torch.zeros(self.dim, device=DEVICE)
        return summed_states


# --- MetaCognitive Memory (Updated update_priorities) ---
class MetaCognitiveMemory:
    """Stores experiences including TD error and samples using Prioritized Replay."""
    INITIAL_TD_ERROR = 1.0

    def __init__(self, capacity=Config.MEMORY_SIZE):
        self.capacity = max(10, capacity)
        self.short_term = deque(maxlen=30)
        self.long_term = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)

    def add(self, experience: Experience):
        """Adds experience. Priority is based on initial TD error."""
        # ... (add method remains the same as previous correction) ...
        if not isinstance(experience, Experience): logger.warning(f"Memory.add: Invalid type {type(experience)}."); return
        priority = abs(experience.td_error) + 1e-5
        try:
            state=experience.state; belief=experience.belief; reward=experience.reward; next_state=experience.next_state; done=experience.done; td_error=experience.td_error
            safe_state = state.detach().clone().cpu() if isinstance(state, torch.Tensor) else torch.tensor(state, device='cpu', dtype=torch.float32)
            if not is_safe(safe_state): raise ValueError("Unsafe state")
            safe_belief = None
            if belief is not None:
                 safe_belief = belief.detach().clone().cpu() if isinstance(belief, torch.Tensor) else torch.tensor(belief, device='cpu', dtype=torch.float32)
                 if safe_belief is not None and safe_belief.shape[0] != Config.HIDDEN_DIM: corrected_belief = torch.zeros(Config.HIDDEN_DIM, device='cpu'); copy_len = min(safe_belief.shape[0], Config.HIDDEN_DIM); corrected_belief[:copy_len] = safe_belief.flatten()[:copy_len]; safe_belief = corrected_belief
                 if safe_belief is not None and not is_safe(safe_belief): raise ValueError("Unsafe belief")
            safe_reward = float(reward); safe_done = bool(done);
            safe_next_state = next_state.detach().clone().cpu() if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, device='cpu', dtype=torch.float32)
            if not is_safe(safe_next_state): raise ValueError("Unsafe next_state")
            safe_td_error = float(td_error)
            safe_exp = Experience(safe_state, safe_belief, safe_reward, safe_next_state, safe_done, safe_td_error)
        except (ValueError, TypeError, AttributeError) as e: logger.error(f"Memory.add: Error processing: {e}."); return
        except Exception as e: logger.error(f"Memory.add: Unexpected error: {e}."); return
        self.short_term.append(safe_exp)
        if len(self.long_term) >= self.capacity: self.long_term.popleft(); self.priorities.popleft()
        self.long_term.append(safe_exp); self.priorities.append(priority)

    def __len__(self):
        return len(self.long_term)

    def sample(self, batch_size, beta=0.4) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
         """Samples a batch using prioritized replay."""
        # ... (sample method remains the same as previous correction) ...
         if len(self.long_term) < batch_size: return None
         priorities = np.array(self.priorities, dtype=np.float64)
         if np.sum(priorities) <= 0 : probs = np.ones_like(priorities) / len(priorities) if len(priorities) > 0 else None; logger.debug("Using uniform probs for sampling.")
         else: probs = priorities ** Config.PER_ALPHA; probs /= probs.sum()
         if probs is None: logger.error("Mem Sample: Cannot calc probs."); return None
         try: indices = np.random.choice(len(self.long_term), batch_size, p=probs, replace=False)
         except ValueError as e: logger.error(f"Mem Sample: Error during np.random.choice (check probs?): {e}"); return None # Handle case where probs might sum != 1 due to float issues
         total_samples = len(self.long_term); weights = (total_samples * probs[indices]) ** (-beta); weights /= weights.max()
         weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)
         batch_dict = {'states': [], 'beliefs': [], 'rewards': [], 'next_states': [], 'dones': []}
         valid_count = 0; skipped_count = 0
         for i, idx in enumerate(indices):
             exp = self.long_term[idx]
             try:
                 state_dev = exp.state.to(DEVICE); belief_dev = exp.belief.to(DEVICE) if exp.belief is not None else None; next_state_dev = exp.next_state.to(DEVICE)
                 if not is_safe(state_dev) or not is_safe(next_state_dev): skipped_count += 1; continue
                 if belief_dev is not None and not is_safe(belief_dev): skipped_count += 1; continue
                 belief_to_add = belief_dev if belief_dev is not None else torch.zeros(Config.HIDDEN_DIM, device=DEVICE)
                 if belief_to_add.shape[0] != Config.HIDDEN_DIM: corrected_belief = torch.zeros(Config.HIDDEN_DIM, device=DEVICE); copy_len = min(belief_to_add.shape[0], Config.HIDDEN_DIM); corrected_belief[:copy_len] = belief_to_add.flatten()[:copy_len]; belief_to_add = corrected_belief
                 batch_dict['states'].append(state_dev); batch_dict['beliefs'].append(belief_to_add); batch_dict['rewards'].append(exp.reward); batch_dict['next_states'].append(next_state_dev); batch_dict['dones'].append(exp.done); valid_count += 1
             except Exception as e: logger.error(f"Error processing exp {idx}: {e}"); skipped_count += 1
         if valid_count == 0: logger.warning("Mem Sample: No valid experiences."); return None
         if skipped_count > 0: logger.debug(f"Mem Sample: Skipped {skipped_count}.")
         try:
             states_batch=torch.stack(batch_dict['states']); beliefs_batch=torch.stack(batch_dict['beliefs']); rewards_batch=torch.tensor(batch_dict['rewards'],dtype=torch.float32,device=DEVICE).unsqueeze(1); next_states_batch=torch.stack(batch_dict['next_states']); dones_batch=torch.tensor(batch_dict['dones'],dtype=torch.bool,device=DEVICE).unsqueeze(1)
             if not is_safe(states_batch) or not is_safe(beliefs_batch) or not is_safe(rewards_batch) or not is_safe(next_states_batch) or not is_safe(dones_batch): logger.error("Mem Sample: Unsafe stacked tensor."); return None
             final_batch_dict = { 'states': states_batch, 'beliefs': beliefs_batch, 'rewards': rewards_batch, 'next_states': next_states_batch, 'dones': dones_batch }
             indices_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
             return final_batch_dict, indices_tensor, weights_tensor
         except Exception as e: logger.error(f"Error stacking batch: {e}"); return None


    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Update the priorities of sampled experiences after a learning step."""
        if indices.numel() == 0 or td_errors.numel() == 0 or indices.shape[0] != td_errors.shape[0]:
            logger.warning(f"update_priorities: Mismatch/empty ({indices.shape}, {td_errors.shape}).")
            return

        # Calculate new priorities: (|td_error| + epsilon)^alpha
        new_priorities_raw = torch.abs(td_errors).cpu().numpy().flatten() + 1e-5 # Add epsilon, move to CPU NumPy
        new_priorities_alpha = new_priorities_raw ** Config.PER_ALPHA # Apply alpha

        np_indices = indices.cpu().numpy() # Indices on CPU NumPy

        updated_count = 0
        for i, idx in enumerate(np_indices):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = new_priorities_alpha[i]
                updated_count += 1
            else:
                logger.warning(f"update_priorities: Invalid index {idx} (deque len: {len(self.priorities)}).")
        # logger.debug(f"Updated priorities for {updated_count}/{len(np_indices)} indices.") # Optional: Verbose logging

    # get_belief_norm methods remain the same
    def get_belief_norm(self, memory_deque):
        valid_beliefs = [exp.belief for exp in memory_deque if exp.belief is not None and isinstance(exp.belief, torch.Tensor) and exp.belief.numel() > 0 and is_safe(exp.belief)]
        if not valid_beliefs: return 0.0
        try: individual_norms = [torch.linalg.norm(b.float()).item() for b in valid_beliefs]; return sum(individual_norms) / len(individual_norms) if individual_norms else 0.0
        except Exception as e: logger.error(f"Error get_belief_norm: {e}"); return 0.0
    def get_short_term_norm(self): return self.get_belief_norm(self.short_term)
    def get_long_term_norm(self): return self.get_belief_norm(self.long_term)

# --- END OF FILE utils.py ---