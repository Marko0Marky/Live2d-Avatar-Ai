# --- START OF FILE utils.py ---
import torch
import numpy as np
from collections import deque, namedtuple
import random
import logging
import sys
import os
import re # Import re for detokenize
from typing import Optional, Dict, Tuple, List, Deque, Union, Any

# --- ADDED: Tokenizer Imports ---
try:
    from tokenizers import Tokenizer, decoders, AddedToken
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from typing import Optional as TokenizerOptional # Rename to avoid conflict
    TOKENIZERS_LIB_AVAILABLE = True
except ImportError:
    TOKENIZERS_LIB_AVAILABLE = False
# --- END ADDED ---

from config import logger, DEVICE, MasterConfig as Config, NLPConfig, TRAIN_DATA # Import specific config classes needed
# --- Import Head Movement Labels ---
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
    INITIAL_TD_ERROR = 1.0
    def __init__(self, capacity: int = Config.Agent.MEMORY_SIZE):
        self.capacity = max(10, capacity)
        self.short_term: Deque[Experience] = deque(maxlen=30)
        self.long_term: Deque[Experience] = deque(maxlen=self.capacity)
        self.priorities: Deque[float] = deque(maxlen=self.capacity)
        # Get default index for 'idle' once
        self.idle_hm_idx = HEAD_MOVEMENT_TO_IDX.get("idle", 0)

    def add(self, experience: Experience):
        """Adds an experience to memory, validating and converting to CPU."""
        if not isinstance(experience, Experience):
             logger.warning(f"Memory.add: Invalid type {type(experience)}."); return

        priority = abs(experience.td_error) + 1e-5 # Use calculated priority

        try:
            state=experience.state; belief=experience.belief; reward=experience.reward
            next_state=experience.next_state; done=experience.done
            td_error=experience.td_error; hm_idx=experience.head_movement_idx

            # Validate state
            if not isinstance(state, torch.Tensor): raise TypeError("State must be a Tensor")
            safe_state = state.detach().clone().cpu()
            if not is_safe(safe_state): raise ValueError("Unsafe state tensor")
            if safe_state.shape[0] != Config.Agent.STATE_DIM: logger.warning(f"Mem Add: State dim mismatch ({safe_state.shape[0]} vs {Config.Agent.STATE_DIM}).")

            # Validate belief (optional)
            safe_belief = None
            if belief is not None:
                 if not isinstance(belief, torch.Tensor): raise TypeError("Belief must be a Tensor or None")
                 safe_belief = belief.detach().clone().cpu()
                 if not is_safe(safe_belief): raise ValueError("Unsafe belief tensor")

            # Validate next_state
            if not isinstance(next_state, torch.Tensor): raise TypeError("Next State must be a Tensor")
            safe_next_state = next_state.detach().clone().cpu()
            if not is_safe(safe_next_state): raise ValueError("Unsafe next_state tensor")
            if safe_next_state.shape[0] != Config.Agent.STATE_DIM: logger.warning(f"Mem Add: Next State dim mismatch ({safe_next_state.shape[0]} vs {Config.Agent.STATE_DIM}).")

            # Validate hm_idx (must be int)
            safe_hm_idx = int(hm_idx)

            # Validate others
            safe_reward = float(reward); safe_done = bool(done); safe_td_error = float(td_error)

            # Create safe experience tuple with CPU tensors and validated types
            safe_exp = Experience(safe_state, safe_belief, safe_reward, safe_next_state, safe_done, safe_td_error, safe_hm_idx)

        except (ValueError, TypeError, AttributeError) as e:
             logger.error(f"Memory.add: Error processing/validating experience: {e}. Exp: {experience}. Not added."); return
        except Exception as e:
             logger.error(f"Memory.add: Unexpected error processing experience: {e}. Not added."); return

        self.short_term.append(safe_exp) # Add to short term buffer too
        if len(self.long_term) >= self.capacity:
             self.long_term.popleft()
             self.priorities.popleft()
        self.long_term.append(safe_exp);
        self.priorities.append(priority) # Store the calculated priority

    def __len__(self) -> int: return len(self.long_term)

    def sample(self, batch_size: int, beta: float = Config.RL.PER_BETA_START) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
         """Samples a batch using Prioritized Experience Replay (PER)."""
         if len(self.long_term) < batch_size: return None

         priorities_float = np.array([float(p) for p in self.priorities], dtype=np.float64)
         if np.sum(priorities_float) <= 1e-9 : # Use small epsilon for sum check
             probs = np.ones_like(priorities_float) / len(priorities_float) if len(priorities_float) > 0 else None;
         else:
             probs = priorities_float ** Config.RL.PER_ALPHA;
             probs /= probs.sum()

         if probs is None or not np.isclose(probs.sum(), 1.0):
              logger.error(f"Memory Sample: Invalid probabilities (Sum={probs.sum() if probs is not None else 'None'}). Using uniform fallback.");
              probs = np.ones_like(priorities_float) / len(priorities_float) if len(priorities_float) > 0 else None;
         if probs is None: return None

         try: indices = np.random.choice(len(self.long_term), batch_size, p=probs, replace=False)
         except ValueError as e: logger.error(f"Memory Sample: Error during np.random.choice (check probs sum?): {e}"); return None

         total_samples = len(self.long_term);
         weights = (total_samples * probs[indices]) ** (-beta); weights /= weights.max();
         weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)

         states_list, beliefs_list, rewards_list, next_states_list, dones_list, head_movement_idx_list = [], [], [], [], [], []
         valid_count = 0; skipped_count = 0
         # Dynamically determine expected belief dim from agent config or first sample
         expected_belief_dim = Config.Agent.HIDDEN_DIM # Use config as default
         # Find first valid belief to confirm dim
         first_valid_belief = next((exp.belief for exp in [self.long_term[i] for i in indices] if exp.belief is not None), None)
         if first_valid_belief is not None:
             expected_belief_dim = first_valid_belief.shape[0]
         default_belief = torch.zeros(expected_belief_dim, device=DEVICE) # Pre-create default belief with correct dim

         for idx in indices:
             exp = self.long_term[idx]
             try:
                 state_dev = exp.state.to(DEVICE);
                 # Handle belief: use default if None or shape mismatch
                 if exp.belief is not None and exp.belief.shape[0] == expected_belief_dim:
                     belief_dev = exp.belief.to(DEVICE)
                 else:
                     if exp.belief is not None: logger.warning(f"Belief shape mismatch for index {idx}. Got {exp.belief.shape}, expected ({expected_belief_dim},). Using default.")
                     belief_dev = default_belief

                 next_state_dev = exp.next_state.to(DEVICE)
                 hm_idx = exp.head_movement_idx # Retrieve index

                 if not is_safe(state_dev) or not is_safe(next_state_dev) or not is_safe(belief_dev):
                     logger.warning(f"Memory Sample: Unsafe tensor detected on device for index {idx}. Skipping."); skipped_count += 1; continue

                 states_list.append(state_dev); beliefs_list.append(belief_dev); rewards_list.append(exp.reward); next_states_list.append(next_state_dev); dones_list.append(exp.done);
                 head_movement_idx_list.append(hm_idx) # Add index to its list
                 valid_count += 1
             except Exception as e: logger.error(f"Memory Sample: Error processing experience index {idx}: {e}"); skipped_count += 1

         if valid_count == 0: logger.warning("Memory Sample: No valid experiences found in the sampled batch."); return None
         if skipped_count > 0: logger.debug(f"Memory Sample: Skipped {skipped_count} experiences during batch creation.")

         try:
             states_batch=torch.stack(states_list); beliefs_batch=torch.stack(beliefs_list); rewards_batch=torch.tensor(rewards_list,dtype=torch.float32,device=DEVICE).unsqueeze(1); next_states_batch=torch.stack(next_states_list); dones_batch=torch.tensor(dones_list,dtype=torch.bool,device=DEVICE).unsqueeze(1)
             head_movement_idx_batch = torch.tensor(head_movement_idx_list, dtype=torch.long, device=DEVICE) # Target indices should be Long

             # Final safety check
             if not is_safe(states_batch) or not is_safe(beliefs_batch) or not is_safe(rewards_batch) or not is_safe(next_states_batch) or not is_safe(dones_batch) or not is_safe(head_movement_idx_batch):
                 logger.error("Memory Sample: Unsafe tensor detected after stacking batch."); return None

             final_batch_dict = {
                 'states': states_batch, 'beliefs': beliefs_batch, 'rewards': rewards_batch,
                 'next_states': next_states_batch, 'dones': dones_batch,
                 'target_hm_idx': head_movement_idx_batch # Add target indices to dict
             }
             indices_tensor = torch.tensor(indices, dtype=torch.long, device=DEVICE)

             return final_batch_dict, indices_tensor, weights_tensor
         except Exception as e: logger.error(f"Memory Sample: Error stacking batch tensors: {e}"); return None

    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Updates priorities based on TD errors."""
        if not isinstance(indices, torch.Tensor) or not isinstance(td_errors, torch.Tensor): logger.warning(f"update_priorities: Invalid types ({type(indices)}, {type(td_errors)})."); return
        if indices.numel() == 0 or td_errors.numel() == 0 or indices.shape[0] != td_errors.shape[0]: logger.warning(f"update_priorities: Mismatch/empty tensors (Indices: {indices.shape}, Errors: {td_errors.shape})."); return

        # Use absolute TD error for priority update
        new_priorities_raw = torch.abs(td_errors).cpu().numpy().flatten() + 1e-5
        # Priorities stored in the deque should reflect alpha influence for sampling bias
        new_priorities_final = new_priorities_raw ** Config.RL.PER_ALPHA

        np_indices = indices.cpu().numpy()
        updated_count = 0; skipped_count = 0
        for i, idx in enumerate(np_indices):
            if 0 <= idx < len(self.priorities):
                 self.priorities[idx] = new_priorities_final[i]; updated_count += 1
            else: logger.warning(f"update_priorities: Invalid index {idx} (Mem size: {len(self.priorities)}). Skip."); skipped_count += 1
        # if skipped_count > 0: logger.debug(f"Priorities updated for {updated_count} indices. Skipped {skipped_count}.")

    def get_belief_norm(self, memory_deque: Deque[Experience]) -> float:
        """Calculates the average L2 norm of belief vectors in a deque."""
        valid_beliefs = [exp.belief for exp in memory_deque if exp.belief is not None and isinstance(exp.belief, torch.Tensor) and exp.belief.numel() > 0 and is_safe(exp.belief)]
        if not valid_beliefs: return 0.0
        try:
             individual_norms = [torch.linalg.norm(b.float().cpu()).item() for b in valid_beliefs];
             return sum(individual_norms) / len(individual_norms) if individual_norms else 0.0
        except Exception as e: logger.error(f"Error calculating belief norm: {e}"); return 0.0

    def get_short_term_norm(self) -> float: return self.get_belief_norm(self.short_term)
    def get_long_term_norm(self) -> float: return self.get_belief_norm(self.long_term)


# --- BPE Tokenizer Variables and Functions (Moved from config.py) ---
bpe_tokenizer: TokenizerOptional[Tokenizer] = None
BPE_PAD_TOKEN_ID: Optional[int] = None
BPE_START_TOKEN_ID: Optional[int] = None
BPE_END_TOKEN_ID: Optional[int] = None
BPE_UNK_TOKEN_ID: Optional[int] = None

def train_or_load_bpe_tokenizer(data: List[Dict[str, Any]], config: NLPConfig) -> Optional[Tokenizer]:
    """Loads or trains the BPE tokenizer. Returns None if loading/training fails."""
    global BPE_PAD_TOKEN_ID, BPE_START_TOKEN_ID, BPE_END_TOKEN_ID, BPE_UNK_TOKEN_ID, bpe_tokenizer

    if not TOKENIZERS_LIB_AVAILABLE:
        logger.error("Cannot load/train BPE tokenizer: 'tokenizers' library not installed.")
        return None

    tokenizer_path = config.TOKENIZER_PATH
    if not tokenizer_path:
        logger.warning("BPE Tokenizer path not set in config. Skipping BPE tokenizer.")
        return None

    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    special_token_objects = [AddedToken(token, single_word=True) for token in config.SPECIAL_TOKENS]

    loaded_tokenizer: Optional[Tokenizer] = None
    try:
        if os.path.exists(tokenizer_path):
            logger.info(f"Loading existing BPE tokenizer from {tokenizer_path}")
            loaded_tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            logger.info(f"BPE Tokenizer not found at {tokenizer_path}. Training a new one...")
            if not data:
                 logger.warning("Cannot train BPE tokenizer: No training data provided. Skipping."); return None

            text_corpus = [item.get(k, "") for item in data for k in ["output", "situation"] if isinstance(item.get(k), str)] # Extract text safely
            if not text_corpus: logger.error("Cannot train BPE tokenizer: No valid text found in training data."); return None

            temp_bpe_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
            temp_bpe_tokenizer.pre_tokenizer = Whitespace()
            temp_bpe_tokenizer.decoder = decoders.BPEDecoder()
            # Use vocab size from config, BpeTrainer handles it
            trainer = BpeTrainer(vocab_size=config.VOCAB_SIZE, special_tokens=special_token_objects)
            logger.info(f"Training BPE tokenizer with target vocab size {config.VOCAB_SIZE} and {len(special_token_objects)} special tokens...")
            temp_bpe_tokenizer.train_from_iterator(text_corpus, trainer=trainer)
            logger.info(f"BPE Tokenizer training complete.")
            temp_bpe_tokenizer.save(tokenizer_path)
            logger.info(f"BPE Tokenizer saved to {tokenizer_path}")
            loaded_tokenizer = temp_bpe_tokenizer

        if loaded_tokenizer:
            if loaded_tokenizer.decoder is None: logger.warning("Loaded BPE tokenizer missing decoder, setting BPEDecoder."); loaded_tokenizer.decoder = decoders.BPEDecoder()
            actual_vocab_size = loaded_tokenizer.get_vocab_size()
            # config.VOCAB_SIZE = actual_vocab_size # Don't overwrite config if HF is primary
            logger.info(f"BPE Tokenizer ready. Actual Vocab size: {actual_vocab_size}")
            bpe_tokenizer = loaded_tokenizer
            BPE_PAD_TOKEN_ID = bpe_tokenizer.token_to_id("<PAD>")
            BPE_START_TOKEN_ID = bpe_tokenizer.token_to_id("<START>")
            BPE_END_TOKEN_ID = bpe_tokenizer.token_to_id("<END>")
            BPE_UNK_TOKEN_ID = bpe_tokenizer.token_to_id("<UNK>")
            if None in [BPE_PAD_TOKEN_ID, BPE_START_TOKEN_ID, BPE_END_TOKEN_ID, BPE_UNK_TOKEN_ID]:
                missing_tokens = [t for t, tid in [("<PAD>", BPE_PAD_TOKEN_ID), ("<START>", BPE_START_TOKEN_ID), ("<END>", BPE_END_TOKEN_ID), ("<UNK>", BPE_UNK_TOKEN_ID)] if tid is None]
                logger.error(f"Failed to get IDs for BPE special tokens: {missing_tokens}.");
                bpe_tokenizer = None # Invalidate tokenizer if specials are missing
                return None
            logger.debug(f"BPE Special Token IDs: PAD={BPE_PAD_TOKEN_ID}, START={BPE_START_TOKEN_ID}, END={BPE_END_TOKEN_ID}, UNK={BPE_UNK_TOKEN_ID}")
            return bpe_tokenizer
        else:
            return None

    except Exception as e:
         logger.error(f"Error during BPE tokenizer loading/training: {e}", exc_info=True)
         bpe_tokenizer = None
         return None

# --- BPE Tokenization functions (Keep separate in case needed elsewhere) ---
def tokenize_bpe(text: str, max_length: int = Config.NLP.MAX_RESPONSE_LEN - 2) -> List[int]:
    if bpe_tokenizer is None: logger.error("BPE Tokenizer not initialized for tokenize_bpe."); return []
    if not isinstance(text, str): logger.warning(f"Invalid input to tokenize_bpe: type {type(text)}."); return []
    try:
        encoding = bpe_tokenizer.encode(text.lower(), add_special_tokens=False)
        truncated_ids = encoding.ids[:max_length]
        return truncated_ids
    except Exception as e:
        logger.error(f"Error during BPE tokenization of '{text[:50]}...': {e}", exc_info=True)
        return []

def detokenize_bpe(indices: List[int]) -> str:
    if bpe_tokenizer is None: logger.error("BPE Tokenizer not initialized for detokenize_bpe."); return ""
    if isinstance(indices, torch.Tensor):
        try: indices = indices.cpu().tolist()
        except Exception as e: logger.warning(f"Error converting tensor to list for detokenize_bpe: {e}"); return "[Tensor Conversion Error]"
    if not isinstance(indices, (list, tuple)): logger.warning(f"Invalid input to detokenize_bpe: type {type(indices)}."); return ""
    if BPE_PAD_TOKEN_ID is None or BPE_START_TOKEN_ID is None or BPE_END_TOKEN_ID is None: logger.error("Cannot detokenize_bpe: Special tokens not set."); return ""
    try:
        special_ids_to_filter = {BPE_PAD_TOKEN_ID, BPE_START_TOKEN_ID, BPE_END_TOKEN_ID}
        valid_indices = [int(idx) for idx in indices if idx is not None and idx not in special_ids_to_filter] # Check for None too
        decoded_text = bpe_tokenizer.decode(valid_indices, skip_special_tokens=True)
        processed_text = re.sub(r'(?<=[a-zA-Z0-9])([.,!?;:])', r' \1', decoded_text)
        processed_text = re.sub(r'\s+', ' ', processed_text)
        return processed_text.strip()
    except Exception as e:
        logger.error(f"Error during BPE detokenization of indices {indices}: {e}", exc_info=True)
        return "[Detokenization Error]"

def initialize_bpe_tokenizer(data: List[Dict[str, Any]], config: NLPConfig):
    """Call this once during startup to initialize the BPE tokenizer."""
    logger.info("Initializing BPE tokenizer...")
    if train_or_load_bpe_tokenizer(data, config) is None:
        logger.warning("BPE tokenizer initialization failed or was skipped.")
    else:
        logger.info("BPE tokenizer initialized successfully.")
# --- END ADDED ---


# --- END OF FILE utils.py ---
