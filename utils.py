# --- START OF FILE utils.py ---

import torch
import numpy as np
from collections import deque, namedtuple
import random
import logging
import sys
import os
import re
from typing import Optional, Dict, Tuple, List, Deque, Union, Any

from config import logger, DEVICE, MasterConfig as Config

# --- Experience NamedTuple (Simplified: No head_movement_idx) ---
Experience = namedtuple('Experience', [
    'state',            # Combined state (CPU Tensor, 12D)
    'belief',           # Belief vector (GNN embedding?) (Optional CPU Tensor)
    'reward',           # Scalar reward (float)
    'next_state',       # Next combined state (CPU Tensor, 12D)
    'done',             # Done flag (bool)
    'td_error',         # Priority for PER (float, based on TD Error)
])

def is_safe(tensor: Optional[Union[torch.Tensor , np.ndarray]]) -> bool:
    """Checks if a tensor or numpy array contains NaN or Inf values."""
    if tensor is None: return False
    if isinstance(tensor, np.ndarray):
        if np.issubdtype(tensor.dtype, np.number): return not (np.isnan(tensor).any() or np.isinf(tensor).any())
        else: return True
    elif isinstance(tensor, torch.Tensor):
        if tensor.is_floating_point() or tensor.is_complex():
            if tensor.numel() == 0: return True
            return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())
        return True
    else:
        try: tensor_conv = torch.tensor(tensor, device='cpu'); return is_safe(tensor_conv)
        except Exception: return False

class MetronicLattice:
    """Discretizes continuous state space based on Syntrometrie concepts."""
    def __init__(self, dim: int = Config.Agent.STATE_DIM, tau: float = Config.Agent.TAU): # Use Config.Agent.STATE_DIM (12)
        self.dim = dim if isinstance(dim, int) and dim > 0 else Config.Agent.STATE_DIM
        self.tau = float(tau) if isinstance(tau, (float, int)) and tau > 1e-6 else Config.Agent.TAU

    def discretize(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            try: x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
            except Exception as e: logger.error(f"Lattice convert error: {e}"); return torch.zeros(self.dim, device=DEVICE, dtype=torch.float32)
        if x.device != DEVICE: x = x.to(DEVICE)
        is_single = x.ndim == 1; x_batch = x.unsqueeze(0) if is_single else x
        if x_batch.shape[-1] != self.dim:
            logger.warning(f"Lattice discretize shape mismatch: {x.shape}. Expected (*, {self.dim}). Padding/Truncating.")
            out_shape = (*x_batch.shape[:-1], self.dim); padded_x = torch.zeros(out_shape, device=DEVICE, dtype=torch.float32)
            copy_len = min(x_batch.shape[-1], self.dim); padded_x[..., :copy_len] = x_batch[..., :copy_len]; x_batch = padded_x
        if x_batch.dtype not in [torch.float32, torch.float64]: x_batch = x_batch.float()
        if not is_safe(x_batch): logger.warning("Lattice discretize unsafe input."); x_batch = torch.zeros_like(x_batch)
        discretized_x_batch = torch.round(x_batch / self.tau) * self.tau
        if not is_safe(discretized_x_batch): logger.warning("Lattice discretize unsafe output."); discretized_x_batch = torch.zeros_like(x_batch)
        return discretized_x_batch.squeeze(0) if is_single else discretized_x_batch

    def S(self, phi_history: torch.Tensor, n1: int, n2: int) -> torch.Tensor:
        """ Calculates sum of states in history slice (legacy, potentially useful). """
        if not isinstance(phi_history, torch.Tensor) or phi_history.ndim != 2 or phi_history.shape[1] != self.dim: logger.warning(f"Lattice S invalid history shape {phi_history.shape}."); return torch.zeros(self.dim, device=DEVICE)
        history_len = phi_history.shape[0];
        if history_len == 0: return torch.zeros(self.dim, device=DEVICE)
        n1_clamped = max(0, min(n1, history_len - 1)); n2_clamped = max(n1_clamped, min(n2, history_len - 1))
        if n1_clamped > n2_clamped: return torch.zeros(self.dim, device=DEVICE)
        history_slice = phi_history[n1_clamped : n2_clamped + 1]
        if history_slice.numel() == 0: return torch.zeros(self.dim, device=DEVICE)
        if not is_safe(history_slice): logger.warning("Lattice S unsafe history slice."); return torch.zeros(self.dim, device=DEVICE)
        summed_states = torch.sum(history_slice.float(), dim=0)
        return summed_states if is_safe(summed_states) else torch.zeros(self.dim, device=DEVICE)


class MetaCognitiveMemory:
    """Stores experiences and samples them using Prioritized Experience Replay."""
    INITIAL_TD_ERROR = 1.0
    def __init__(self, capacity: int = Config.Agent.MEMORY_SIZE):
        self.capacity = max(10, capacity)
        self.short_term: Deque[Experience] = deque(maxlen=30)
        self.long_term: Deque[Experience] = deque(maxlen=self.capacity)
        self.priorities: Deque[float] = deque(maxlen=self.capacity)
        logger.info(f"MetaCognitiveMemory initialized with capacity {self.capacity}.")

    def add(self, experience: Experience):
        """Adds an experience to memory, validating and converting to CPU."""
        if not isinstance(experience, Experience): logger.warning(f"Memory.add: Invalid type {type(experience)}."); return
        priority = abs(experience.td_error) + 1e-5
        try:
            # Unpack the simplified tuple
            state, belief, reward, next_state, done, td_error = experience
            if not isinstance(state, torch.Tensor) or state.shape != (Config.Agent.STATE_DIM,): raise TypeError(f"Invalid State shape {getattr(state, 'shape', 'N/A')}")
            if not isinstance(next_state, torch.Tensor) or next_state.shape != (Config.Agent.STATE_DIM,): raise TypeError(f"Invalid Next State shape {getattr(next_state, 'shape', 'N/A')}")
            safe_state = state.detach().clone().cpu(); safe_next_state = next_state.detach().clone().cpu()
            if not is_safe(safe_state) or not is_safe(safe_next_state): raise ValueError("Unsafe State/Next State")
            safe_belief = None
            if belief is not None:
                if not isinstance(belief, torch.Tensor): raise TypeError("Invalid Belief type")
                safe_belief = belief.detach().clone().cpu()
                if not is_safe(safe_belief): raise ValueError("Unsafe Belief")
            # Create the simplified experience tuple
            safe_exp = Experience(safe_state, safe_belief, float(reward), safe_next_state, bool(done), float(td_error))
        except (ValueError, TypeError, AttributeError) as e: logger.error(f"Memory.add: Error validating experience: {e}. Exp: {experience}."); return
        except Exception as e: logger.error(f"Memory.add: Unexpected error during validation: {e}."); return

        self.short_term.append(safe_exp)
        if len(self.long_term) >= self.capacity: self.long_term.popleft(); self.priorities.popleft()
        self.long_term.append(safe_exp); self.priorities.append(priority)

    def __len__(self) -> int: return len(self.long_term)

    def sample(self, batch_size: int, beta: float = Config.RL.PER_BETA_START) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        """Samples a batch using PER. Returns dict with states, beliefs, rewards, next_states, dones."""
        if len(self.long_term) < batch_size: return None
        priorities_float = np.array([float(p) for p in self.priorities], dtype=np.float64) + 1e-9
        if priorities_float.sum() <= 1e-8: probs = np.ones_like(priorities_float) / max(1, len(priorities_float))
        else: probs = priorities_float ** Config.RL.PER_ALPHA; probs /= probs.sum()
        if not np.isclose(probs.sum(), 1.0, atol=1e-7): probs = np.ones_like(priorities_float) / max(1, len(priorities_float))
        try: indices = np.random.choice(len(self.long_term), batch_size, p=probs, replace=False)
        except ValueError as e: logger.error(f"Memory Sample choice error: {e}"); return None
        total_samples = len(self.long_term); weights = (total_samples * probs[indices]) ** (-beta)
        weights /= max(1e-9, weights.max()); weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Retrieve and prepare experiences
        states_list, beliefs_list, rewards_list, next_states_list, dones_list = [], [], [], [], []
        skipped_count = 0
        expected_belief_dim = Config.Agent.GNN.GNN_HIDDEN_DIM # Use GNN dim
        first_valid_belief = next((exp.belief for exp in [self.long_term[i] for i in indices] if exp.belief is not None), None)
        if first_valid_belief is not None: expected_belief_dim = first_valid_belief.shape[0]
        default_belief = torch.zeros(expected_belief_dim, device=DEVICE, dtype=torch.float32)

        for idx in indices:
            exp = self.long_term[idx]
            try:
                if exp.state.shape != (Config.Agent.STATE_DIM,) or exp.next_state.shape != (Config.Agent.STATE_DIM,): skipped_count += 1; continue
                if not is_safe(exp.state) or not is_safe(exp.next_state) or (exp.belief is not None and not is_safe(exp.belief)): skipped_count += 1; continue

                state_dev = exp.state.to(DEVICE); next_state_dev = exp.next_state.to(DEVICE)
                belief_dev = default_belief
                if exp.belief is not None and exp.belief.shape[0] == expected_belief_dim: belief_dev = exp.belief.to(DEVICE)

                states_list.append(state_dev); beliefs_list.append(belief_dev); rewards_list.append(exp.reward);
                next_states_list.append(next_state_dev); dones_list.append(exp.done)
            except Exception as e: logger.error(f"Mem Sample error processing idx {idx}: {e}"); skipped_count += 1

        if len(states_list) == 0: logger.warning("Mem Sample: No valid experiences after processing."); return None
        if skipped_count > 0: logger.debug(f"Mem Sample: Skipped {skipped_count}/{batch_size}.")

        # Stack lists into batch tensors
        try:
            states_batch = torch.stack(states_list); beliefs_batch = torch.stack(beliefs_list)
            rewards_batch = torch.tensor(rewards_list,dtype=torch.float32,device=DEVICE).unsqueeze(1)
            next_states_batch = torch.stack(next_states_list); dones_batch = torch.tensor(dones_list,dtype=torch.bool,device=DEVICE).unsqueeze(1)
            if not all(map(is_safe, [states_batch, beliefs_batch, rewards_batch, next_states_batch, dones_batch])): raise ValueError("Unsafe tensor after stacking")
            # Return simplified dictionary
            final_batch_dict = {'states': states_batch, 'beliefs': beliefs_batch, 'rewards': rewards_batch, 'next_states': next_states_batch, 'dones': dones_batch}
            indices_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
            return final_batch_dict, indices_tensor, weights_tensor
        except Exception as e: logger.error(f"Mem Sample stacking error: {e}"); return None

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Updates priorities based on TD errors (magnitude used)."""
        if not isinstance(indices, torch.Tensor) or not isinstance(td_errors, torch.Tensor): logger.warning(f"update_priorities: Invalid types"); return
        if indices.numel() == 0 or td_errors.numel() == 0 or indices.shape[0] != td_errors.shape[0]: logger.warning(f"update_priorities: Mismatch/empty tensors"); return
        # Use abs TD error + epsilon for priority, then apply alpha
        new_priorities_raw = torch.abs(td_errors).clamp(min=1e-6).cpu().numpy().flatten() + 1e-5
        new_priorities_final = np.power(new_priorities_raw, Config.RL.PER_ALPHA)
        np_indices = indices.cpu().numpy()
        for i, idx in enumerate(np_indices):
            if 0 <= idx < len(self.priorities): self.priorities[idx] = new_priorities_final[i]

    def get_belief_norm(self, memory_deque: Deque[Experience]) -> float:
        valid_beliefs = [exp.belief for exp in memory_deque if exp.belief is not None and isinstance(exp.belief, torch.Tensor) and exp.belief.numel() > 0 and is_safe(exp.belief)]
        if not valid_beliefs: return 0.0
        try: norms = [torch.linalg.norm(b.float().cpu()).item() for b in valid_beliefs]; return sum(norms) / len(norms) if norms else 0.0
        except Exception as e: logger.error(f"Error calculating belief norm: {e}"); return 0.0
    def get_short_term_norm(self) -> float: return self.get_belief_norm(self.short_term)
    def get_long_term_norm(self) -> float: return self.get_belief_norm(self.long_term)

# --- END OF FILE utils.py ---
