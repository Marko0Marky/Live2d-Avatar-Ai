# --- START OF FILE utils.py ---
# ... (imports remain the same) ...
import torch
import numpy as np
from collections import deque, namedtuple
import random
import logging
from typing import Optional, Dict, Tuple, List, Deque, Union, Any # Added Union, Any

from config import logger, DEVICE, MasterConfig as Config

Experience = namedtuple('Experience', ['state', 'belief', 'reward', 'next_state', 'done', 'td_error'])

def is_safe(tensor: Optional[Union[torch.Tensor , np.ndarray]]) -> bool: # Use Union
    if tensor is None: return False
    if not isinstance(tensor, torch.Tensor):
        try: tensor = torch.tensor(tensor, device='cpu')
        except Exception: return False
    if not torch.is_floating_point(tensor) and not torch.is_complex(tensor) and not hasattr(tensor, 'is_signed') : return True
    if tensor.numel() == 0: return True
    if torch.is_floating_point(tensor) or torch.is_complex(tensor): return torch.isfinite(tensor).all()
    return True

# --- MODIFIED: Lattice uses combined state dim ---
class MetronicLattice:
    """Discretizes continuous state space based on Syntrometrie concepts."""
    def __init__(self, dim: int = Config.Agent.STATE_DIM, tau: float = Config.Agent.TAU): # Default uses combined dim
        if not isinstance(dim, int) or dim <= 0: logger.error(f"Lattice dim err {dim}"); dim = 1
        self.dim = dim
        if not isinstance(tau, (float, int)) or tau <= 1e-6: logger.warning(f"Lattice tau err {tau}"); self.tau = 0.1
        else: self.tau = float(tau)

    # ... (discretize and S methods remain the same, they use self.dim) ...
    def discretize(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            try: x = torch.tensor(x, device=DEVICE, dtype=torch.float32)
            except Exception as e: logger.error(f"Lattice convert err: {e}"); return torch.zeros(self.dim, device=DEVICE, dtype=torch.float32)
        if x.device != DEVICE: x = x.to(DEVICE)
        was_single_instance = x.ndim == 1; x_batch = x if not was_single_instance else x.unsqueeze(0)
        if x_batch.ndim != 2 or x_batch.shape[1] != self.dim:
            logger.warning(f"Lattice discretize received unexpected shape {x.shape}. Expected (-1, {self.dim}). Returning zeros.")
            out_shape = (x.shape[0], self.dim) if x.ndim==2 else (self.dim,)
            return torch.zeros(out_shape, device=DEVICE, dtype=torch.float32)
        if x_batch.dtype not in [torch.float32, torch.float64]: x_batch = x_batch.float()
        if not is_safe(x_batch): logger.warning("Lattice unsafe input."); x_batch = torch.zeros_like(x_batch)
        discretized_x_batch = torch.round(x_batch / self.tau) * self.tau
        if not is_safe(discretized_x_batch): logger.warning("Lattice unsafe output."); discretized_x_batch = torch.zeros_like(x_batch)
        return discretized_x_batch.squeeze(0) if was_single_instance else discretized_x_batch

    def S(self, phi_history: torch.Tensor, n1: int, n2: int) -> torch.Tensor:
        if not isinstance(phi_history, torch.Tensor) or phi_history.ndim != 2 or phi_history.shape[1] != self.dim: logger.warning(f"Lattice S shape err (Expected dim {self.dim})."); return torch.zeros(self.dim, device=DEVICE)
        history_len = phi_history.shape[0];
        if history_len == 0: return torch.zeros(self.dim, device=DEVICE)
        n1_clamped = max(0, min(n1, history_len - 1)); n2_clamped = max(0, min(n2, history_len - 1))
        if n1_clamped > n2_clamped: return torch.zeros(self.dim, device=DEVICE)
        history_slice = phi_history[n1_clamped : n2_clamped + 1]
        if history_slice.numel() == 0: return torch.zeros(self.dim, device=DEVICE)
        if not is_safe(history_slice): logger.warning("Lattice S unsafe slice."); return torch.zeros(self.dim, device=DEVICE)
        summed_states = torch.sum(history_slice, dim=0)
        if not is_safe(summed_states): logger.warning("Lattice S unsafe sum."); return torch.zeros(self.dim, device=DEVICE)
        return summed_states


# --- MetaCognitiveMemory (Remains the same) ---
class MetaCognitiveMemory:
    INITIAL_TD_ERROR = 1.0
    def __init__(self, capacity: int = Config.Agent.MEMORY_SIZE):
        self.capacity = max(10, capacity)
        self.short_term: Deque[Experience] = deque(maxlen=30)
        self.long_term: Deque[Experience] = deque(maxlen=self.capacity)
        self.priorities: Deque[float] = deque(maxlen=self.capacity)

    def add(self, experience: Experience):
        if not isinstance(experience, Experience): logger.warning(f"Memory.add: Invalid type {type(experience)}."); return
        priority = abs(experience.td_error) + 1e-5
        try:
            state=experience.state; belief=experience.belief; reward=experience.reward; next_state=experience.next_state; done=experience.done; td_error=experience.td_error
            safe_state = state.detach().clone().cpu() if isinstance(state, torch.Tensor) else torch.tensor(state, device='cpu', dtype=torch.float32)
            if not is_safe(safe_state): raise ValueError("Unsafe state")
            safe_belief = None
            if belief is not None:
                 safe_belief = belief.detach().clone().cpu() if isinstance(belief, torch.Tensor) else torch.tensor(belief, device='cpu', dtype=torch.float32)
                 if safe_belief is not None and not is_safe(safe_belief): raise ValueError("Unsafe belief")
            safe_reward = float(reward); safe_done = bool(done);
            safe_next_state = next_state.detach().clone().cpu() if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, device='cpu', dtype=torch.float32)
            if not is_safe(safe_next_state): raise ValueError("Unsafe next_state")
            safe_td_error = float(td_error)
            safe_exp = Experience(safe_state, safe_belief, safe_reward, safe_next_state, safe_done, safe_td_error)
        except (ValueError, TypeError, AttributeError) as e: logger.error(f"Memory.add: Error processing/validating experience: {e}."); return
        except Exception as e: logger.error(f"Memory.add: Unexpected error processing experience: {e}."); return
        self.short_term.append(safe_exp)
        if len(self.long_term) >= self.capacity: self.long_term.popleft(); self.priorities.popleft()
        self.long_term.append(safe_exp); self.priorities.append(priority)

    def __len__(self) -> int: return len(self.long_term)

    def sample(self, batch_size: int, beta: float = Config.RL.PER_BETA_START) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
         if len(self.long_term) < batch_size: return None
         priorities_float = [float(p) for p in self.priorities]; priorities = np.array(priorities_float, dtype=np.float64)
         if np.sum(priorities) <= 0 : probs = np.ones_like(priorities) / len(priorities) if len(priorities) > 0 else None; logger.debug("Memory Sample: Zero priorities detected. Using uniform probabilities.")
         else: probs = priorities ** Config.RL.PER_ALPHA; probs /= probs.sum()
         if probs is None or not np.isclose(probs.sum(), 1.0): logger.error(f"Memory Sample: Invalid probabilities calculated (Sum={probs.sum() if probs is not None else 'None'}). Check priorities/alpha."); probs = np.ones_like(priorities) / len(priorities) if len(priorities) > 0 else None;
         if probs is None: return None
         try: indices = np.random.choice(len(self.long_term), batch_size, p=probs, replace=False)
         except ValueError as e: logger.error(f"Memory Sample: Error during np.random.choice (check probs sum?): {e}"); return None
         total_samples = len(self.long_term); weights = (total_samples * probs[indices]) ** (-beta); weights /= weights.max(); weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)
         states_list, beliefs_list, rewards_list, next_states_list, dones_list = [], [], [], [], []; valid_count = 0; skipped_count = 0
         for idx in indices:
             exp = self.long_term[idx]
             try:
                 state_dev = exp.state.to(DEVICE); belief_dev = exp.belief.to(DEVICE) if exp.belief is not None else None; next_state_dev = exp.next_state.to(DEVICE)
                 if not is_safe(state_dev) or not is_safe(next_state_dev): skipped_count += 1; continue
                 if belief_dev is not None and not is_safe(belief_dev): skipped_count += 1; continue
                 belief_to_add = belief_dev if belief_dev is not None else torch.zeros(Config.Agent.HIDDEN_DIM, device=DEVICE)
                 states_list.append(state_dev); beliefs_list.append(belief_to_add); rewards_list.append(exp.reward); next_states_list.append(next_state_dev); dones_list.append(exp.done); valid_count += 1
             except Exception as e: logger.error(f"Memory Sample: Error processing experience index {idx}: {e}"); skipped_count += 1
         if valid_count == 0: logger.warning("Memory Sample: No valid experiences found in the sampled batch."); return None
         if skipped_count > 0: logger.debug(f"Memory Sample: Skipped {skipped_count} experiences during batch creation.")
         try:
             states_batch=torch.stack(states_list); beliefs_batch=torch.stack(beliefs_list); rewards_batch=torch.tensor(rewards_list,dtype=torch.float32,device=DEVICE).unsqueeze(1); next_states_batch=torch.stack(next_states_list); dones_batch=torch.tensor(dones_list,dtype=torch.bool,device=DEVICE).unsqueeze(1)
             if not is_safe(states_batch) or not is_safe(beliefs_batch) or not is_safe(rewards_batch) or not is_safe(next_states_batch) or not is_safe(dones_batch): logger.error("Memory Sample: Unsafe tensor detected after stacking batch."); return None
             final_batch_dict = { 'states': states_batch, 'beliefs': beliefs_batch, 'rewards': rewards_batch, 'next_states': next_states_batch, 'dones': dones_batch }
             indices_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)
             return final_batch_dict, indices_tensor, weights_tensor
         except Exception as e: logger.error(f"Memory Sample: Error stacking batch tensors: {e}"); return None

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        if not isinstance(indices, torch.Tensor) or not isinstance(td_errors, torch.Tensor): logger.warning(f"update_priorities: Invalid input types ({type(indices)}, {type(td_errors)})."); return
        if indices.numel() == 0 or td_errors.numel() == 0 or indices.shape[0] != td_errors.shape[0]: logger.warning(f"update_priorities: Mismatch/empty tensors (Indices: {indices.shape}, Errors: {td_errors.shape}). Cannot update."); return
        new_priorities_raw = torch.abs(td_errors).cpu().numpy().flatten() + 1e-5
        new_priorities_alpha = new_priorities_raw ** Config.RL.PER_ALPHA
        np_indices = indices.cpu().numpy()
        updated_count = 0
        for i, idx in enumerate(np_indices):
            if 0 <= idx < len(self.priorities): self.priorities[idx] = new_priorities_alpha[i]; updated_count += 1
            else: logger.warning(f"update_priorities: Invalid index {idx} encountered (Memory size: {len(self.priorities)}). Skipping update for this index.")

    def get_belief_norm(self, memory_deque: Deque[Experience]) -> float:
        valid_beliefs = [exp.belief for exp in memory_deque if exp.belief is not None and isinstance(exp.belief, torch.Tensor) and exp.belief.numel() > 0 and is_safe(exp.belief)]
        if not valid_beliefs: return 0.0
        try: individual_norms = [torch.linalg.norm(b.float()).item() for b in valid_beliefs]; return sum(individual_norms) / len(individual_norms) if individual_norms else 0.0
        except Exception as e: logger.error(f"Error get_belief_norm: {e}"); return 0.0

    def get_short_term_norm(self) -> float: return self.get_belief_norm(self.short_term)
    def get_long_term_norm(self) -> float: return self.get_belief_norm(self.long_term)

# --- END OF FILE utils.py ---
