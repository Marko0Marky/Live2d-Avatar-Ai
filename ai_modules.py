# --- START OF FILE ai_modules.py ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
from typing import Dict, Optional, Tuple, List

# Use MasterConfig and tokenizer variables
from config import MasterConfig as Config
from config import DEVICE, logger, TRAIN_DATA, tokenizer, tokenize, detokenize, PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, UNK_TOKEN_ID
from utils import is_safe

# --- EmotionalModule ---
class EmotionalModule(nn.Module):
    def __init__(self, input_dim: int = Config.Agent.EMOTION_DIM + 1): # Added type hint
         super().__init__();
         self.input_dim=input_dim;
         self.fc=nn.Sequential(nn.Linear(self.input_dim, 32), nn.ReLU(), nn.Linear(32, Config.Agent.EMOTION_DIM), nn.Sigmoid());
         self.decay=0.85
    def forward(self, state_emo_part_batch: torch.Tensor, reward_batch: torch.Tensor, prev_emotions_batch: torch.Tensor) -> torch.Tensor: # Added type hints
        # --- Improved Shape Check and Logging ---
        if not isinstance(state_emo_part_batch, torch.Tensor) or \
           not isinstance(reward_batch, torch.Tensor) or \
           not isinstance(prev_emotions_batch, torch.Tensor):
            logger.error(f"EmoMod Batch Type Err: state={type(state_emo_part_batch)}, rew={type(reward_batch)}, prev={type(prev_emotions_batch)}")
            bs = state_emo_part_batch.shape[0] if isinstance(state_emo_part_batch, torch.Tensor) else \
                 (reward_batch.shape[0] if isinstance(reward_batch, torch.Tensor) and reward_batch.ndim > 0 else 1)
            return torch.zeros(bs, Config.Agent.EMOTION_DIM, device=DEVICE)

        # Batch size should now always be >= 1
        batch_size = state_emo_part_batch.shape[0]
        expected_state_shape = (batch_size, Config.Agent.EMOTION_DIM)
        expected_reward_shape = (batch_size, 1)
        expected_prev_shape = (batch_size, Config.Agent.EMOTION_DIM)

        if state_emo_part_batch.shape != expected_state_shape or \
           reward_batch.shape != expected_reward_shape or \
           prev_emotions_batch.shape != expected_prev_shape:
             logger.error(f"EmoMod Batch Shape Err. Expected:"
                          f" State={expected_state_shape}, Reward={expected_reward_shape}, Prev={expected_prev_shape}. "
                          f"Got:"
                          f" State={state_emo_part_batch.shape}, Reward={reward_batch.shape}, Prev={prev_emotions_batch.shape}")
             return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        # --- End Improved Check ---

        if not is_safe(state_emo_part_batch) or not is_safe(reward_batch) or not is_safe(prev_emotions_batch): logger.warning("EmoMod Batch Unsafe."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)

        try:
            emotion_inputs = torch.cat([state_emo_part_batch, reward_batch.float()], dim=1)
        except Exception as e: logger.error(f"EmoMod Batch Concat Err: {e}."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)

        expected_input_dim = self.fc[0].in_features
        if emotion_inputs.shape != (batch_size, expected_input_dim): logger.error(f"EmoMod Batch Dim Err after concat. Exp: {(batch_size, expected_input_dim)}, Got: {emotion_inputs.shape}"); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)

        try: scaled_emotions = self.fc(emotion_inputs); scaled_emotions = torch.clamp(scaled_emotions, 0, 1)
        except Exception as e: logger.error(f"EmoMod Batch FC Err: {e}."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)

        smoothed_emotions = self.decay * prev_emotions_batch + (1 - self.decay) * scaled_emotions; final_emotions = torch.clamp(smoothed_emotions, 0, 1)
        if not is_safe(final_emotions): logger.error("EmoMod Batch Unsafe Out."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        return final_emotions

# --- SyntrixKorporator ---
class SyntrixKorporator(nn.Module):
    """Implements the Korporator component with optimized batch processing."""
    def __init__(self, input_dim: int, hidden_dim: int, m: int = 6): # Added type hints
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0 or \
           not isinstance(hidden_dim, int) or hidden_dim <= 0: raise ValueError(f"Korporator: invalid dims.")
        self.input_dim = input_dim; self.hidden_dim = hidden_dim
        self.metrophor = nn.Parameter(torch.randn(hidden_dim, device=DEVICE) * 0.1) # Metrophor should be hidden_dim
        if hidden_dim < 2: self.m = 1 if hidden_dim == 1 else 0
        else: self.m = min(m, hidden_dim // 2)
        if self.m <= 0: raise ValueError(f"Korporator: m={self.m} invalid for hidden_dim={hidden_dim}.")
        self.k_input_dim = self.m * 2
        self.Km = nn.Linear(self.k_input_dim, hidden_dim); self.Cm = nn.Linear(self.k_input_dim, hidden_dim)
        self.input_projector = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()).to(DEVICE)

    def forward(self, phi_input: torch.Tensor, psi_input: torch.Tensor, level: int = 1) -> torch.Tensor: # Added type hints
        """ Performs Korporator operation, optimized for batches."""
        if not isinstance(phi_input, torch.Tensor) or not isinstance(psi_input, torch.Tensor): logger.warning("Korp Fwd: Invalid types."); return torch.zeros(1, self.hidden_dim, device=DEVICE)
        was_single = phi_input.ndim == 1
        phi_batch = phi_input.unsqueeze(0) if was_single else phi_input
        psi_batch = psi_input.unsqueeze(0) if was_single else psi_input
        batch_size = phi_batch.shape[0]

        if psi_batch.shape[0] != batch_size or phi_batch.shape[1] != self.input_dim or psi_batch.shape[1] != self.input_dim:
             logger.error(f"Korp Fwd Batch: Shape mismatch. Phi={phi_batch.shape}, Psi={psi_batch.shape}, Expected Input Dim={self.input_dim}. Returning zeros.");
             return torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        if not is_safe(phi_batch) or not is_safe(psi_batch): logger.warning("Korp Fwd Batch: Unsafe inputs."); return torch.zeros(batch_size, self.hidden_dim, device=DEVICE)

        try:
             hidden_phi = self.input_projector(phi_batch);
             hidden_psi = self.input_projector(psi_batch)
        except Exception as e: logger.error(f"Korp Fwd Batch: Projection error: {e}."); return torch.zeros(batch_size, self.hidden_dim, device=DEVICE)

        if not is_safe(hidden_phi) or not is_safe(hidden_psi) or hidden_phi.shape != (batch_size, self.hidden_dim) or hidden_psi.shape != (batch_size, self.hidden_dim): logger.warning("Korp Fwd Batch: Unsafe/wrong shape projection."); return torch.zeros(batch_size, self.hidden_dim, device=DEVICE)

        current_structure = self.metrophor.unsqueeze(0).repeat(batch_size, 1)
        if not is_safe(current_structure): logger.warning("Korp Fwd Batch: Unsafe metrophor."); current_structure = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        m = self.m

        for i in range(max(1, level)):
            try:
                phi_part = hidden_phi[:, :m]; psi_part = hidden_psi[:, :m]
                combined_comp = torch.cat((phi_part, psi_part), dim=1)
                composition_result = torch.relu(self.Cm(combined_comp))
                if not is_safe(composition_result) or composition_result.shape[1] != self.hidden_dim: logger.warning(f"Korp L{i+1}: Unsafe compose."); composition_result = torch.zeros_like(current_structure)

                struct_part = current_structure[:, :m]; comp_res_part = composition_result[:, :m]
                combined_coup = torch.cat((struct_part, comp_res_part), dim=1)
                coupled_structure = torch.relu(self.Km(combined_coup))
                if not is_safe(coupled_structure) or coupled_structure.shape[1] != self.hidden_dim: logger.warning(f"Korp L{i+1}: Unsafe couple."); break
                current_structure = coupled_structure
            except Exception as e: logger.error(f"Korp L{i+1}: Loop error: {e}."); break

        final_structure = current_structure
        if not is_safe(final_structure): logger.error("Korp Fwd Batch: Final unsafe."); final_structure = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        if final_structure.shape != (batch_size, self.hidden_dim): logger.error(f"Korp Fwd Batch: Final shape mismatch."); final_structure = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        return final_structure.squeeze(0) if was_single else final_structure


# --- StrukturKaskade ---
class StrukturKaskade(nn.Module):
    """Implements a cascade of linear layers with ReLU activations for hierarchical processing."""
    def __init__(self, input_dim: int, hidden_dim: int, levels: int = Config.Agent.CASCADE_LEVELS): # Added type hints
        super().__init__()
        self.levels=max(1, int(levels))
        layers=[]
        current_dim=input_dim
        self._output_dim=input_dim # Default output dim is input dim
        self.num_actual_layers=0
        self._input_dim=input_dim

        if not isinstance(input_dim, int) or input_dim <= 0 or \
           not isinstance(hidden_dim, int) or hidden_dim <= 0:
            logger.warning(f"Kaskade: Invalid dims (in={input_dim}, hidden={hidden_dim}). Using Identity.");
            self.network = nn.Identity()
            self._output_dim=input_dim if input_dim > 0 else 1; self.levels=0;
            self._input_dim = self._output_dim if input_dim <= 0 else input_dim
        else:
            for i in range(self.levels):
                output_dim=hidden_dim;
                layers.append(nn.Linear(current_dim, output_dim));
                layers.append(nn.ReLU());
                current_dim=output_dim;
                self._output_dim=output_dim

            if not layers:
                logger.warning("Kaskade: No layers created despite valid dims/levels>0? Using Identity.");
                self.network=nn.Identity();
                self._output_dim=input_dim
            else:
                self.network=nn.Sequential(*layers);
                self.num_actual_layers=len([l for l in layers if isinstance(l, nn.Linear)])

    def forward(self, x_input: torch.Tensor) -> torch.Tensor: # Added type hints
        if not isinstance(x_input, torch.Tensor): logger.warning("Kaskade NaN input"); return torch.zeros(1, self._output_dim, device=DEVICE)
        if x_input.device != DEVICE: x_input = x_input.to(DEVICE)
        if not is_safe(x_input): logger.warning("Kaskade Unsafe input"); return torch.zeros_like(x_input)

        was_single_instance = x_input.ndim == 1;
        x_batch = x_input.unsqueeze(0) if was_single_instance else x_input

        expected_input_dim = -1
        if hasattr(self, 'network'):
            if isinstance(self.network, nn.Sequential) and self.num_actual_layers > 0 and isinstance(self.network[0], nn.Linear): expected_input_dim = self.network[0].in_features
            elif isinstance(self.network, nn.Identity): expected_input_dim = self._input_dim
            elif isinstance(self.network, nn.Linear): expected_input_dim = self.network.in_features
        else:
             logger.error("Kaskade Forward: self.network not initialized!")
             out_shape = (x_batch.shape[0], self._output_dim) if self._output_dim > 0 else (x_batch.shape[0], 1)
             return torch.zeros(out_shape, device=DEVICE)

        if expected_input_dim > 0 and x_batch.shape[1] != expected_input_dim:
             logger.error(f"Kaskade Dim Err: Expected input {expected_input_dim}, got {x_batch.shape[1]}");
             return torch.zeros(x_batch.shape[0], self._output_dim, device=DEVICE)

        try: output_batch = self.network(x_batch)
        except Exception as e: logger.error(f"Kaskade Fwd Err: {e}"); return torch.zeros(x_batch.shape[0], self._output_dim, device=DEVICE)

        if not is_safe(output_batch): logger.error("Kaskade Unsafe output"); return torch.zeros_like(output_batch)

        if output_batch.ndim != 2 or output_batch.shape[1] != self._output_dim:
             logger.error(f"Kaskade Shape Err: Expected output ({self._output_dim},), got {output_batch.shape}");
             try: output_batch = output_batch.view(x_batch.shape[0], self._output_dim)
             except RuntimeError: return torch.zeros(x_batch.shape[0], self._output_dim, device=DEVICE)

        return output_batch.squeeze(0) if was_single_instance else output_batch


# --- SimpleGPT ---
class SimpleGPT(nn.Module):
    """A basic Transformer-based model using the BPE tokenizer."""
    def __init__(self, vocab_size: int = Config.NLP.VOCAB_SIZE, embed_dim: int = 64, hidden_dim: int = 128, num_heads: int = 4): # Added type hints
        super().__init__();
        logger.info(f"Initializing SimpleGPT with vocab_size={vocab_size} (from Tokenizer)")
        if not isinstance(vocab_size, int) or vocab_size <= 0 or \
           not isinstance(embed_dim, int) or embed_dim <= 0 or \
           not isinstance(hidden_dim, int) or hidden_dim <= 0 or \
           not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"GPT: vocab_size ({vocab_size}), embed_dim ({embed_dim}), hidden_dim ({hidden_dim}), and num_heads ({num_heads}) must be positive integers.")

        self.vocab_size = vocab_size; self.embed_dim = embed_dim; self.hidden_dim = hidden_dim; self.num_heads = num_heads;
        self.max_len = Config.NLP.MAX_RESPONSE_LEN

        # Use PAD_TOKEN_ID from config (set after tokenizer load)
        if PAD_TOKEN_ID is None: raise ValueError("PAD_TOKEN_ID not set in config before GPT init")
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=PAD_TOKEN_ID)

        if embed_dim % num_heads != 0:
             original_heads = num_heads;
             possible_heads = [h for h in range(original_heads, 0, -1) if embed_dim % h == 0]
             if possible_heads:
                 self.num_heads = possible_heads[0];
                 logger.warning(f"GPT embed_dim ({embed_dim}) not divisible by num_heads ({original_heads}). Adjusting heads to {self.num_heads}.")
             else:
                 logger.error(f"Invalid GPT embed_dim ({embed_dim}) - no possible head count found <= {original_heads}. Falling back to 1 head.")
                 self.num_heads = 1


        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=self.num_heads, dim_feedforward=hidden_dim,
                batch_first=True, dropout=0.1, activation=F.relu
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        except Exception as e:
             logger.critical(f"Failed to initialize Transformer Encoder Layer/Stack: {e}", exc_info=True);
             raise RuntimeError("Transformer initialization failed") from e

        self.output = nn.Linear(self.embed_dim, self.vocab_size)
        self.optimizer = optim.Adam(self.parameters(), lr=Config.NLP.GPT_LR)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
        self.emotion_bias_layer = nn.Linear(4, self.embed_dim).to(DEVICE)

    def _get_positional_encoding(self, seq_len: int, batch_size: int = 1) -> torch.Tensor: # Added type hints
        """Generates standard sinusoidal positional encoding."""
        if not isinstance(seq_len, int) or seq_len <= 0 or self.embed_dim <= 0:
            logger.warning(f"Cannot get positional encoding for seq_len={seq_len}, embed_dim={self.embed_dim}. Returning zeros.");
            return torch.zeros(batch_size, seq_len, self.embed_dim, device=DEVICE)

        position = torch.arange(seq_len, dtype=torch.float, device=DEVICE).unsqueeze(1);
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=DEVICE) * (-math.log(10000.0) / self.embed_dim))

        pe = torch.zeros(seq_len, self.embed_dim, device=DEVICE);
        pe[:, 0::2] = torch.sin(position * div_term)
        num_cos_terms = pe[:, 1::2].shape[1]
        if num_cos_terms > 0:
            pe[:, 1::2] = torch.cos(position * div_term[:num_cos_terms])

        return pe.unsqueeze(0).repeat(batch_size, 1, 1)

    def _generate_square_subsequent_mask(self, sz: int) -> Optional[torch.Tensor]: # Added type hints
        """Generates a square causal mask for the Transformer."""
        if not isinstance(sz, int) or sz <= 0: return None
        mask = torch.triu(torch.ones(sz, sz, device=DEVICE) * float('-inf'), diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor, emotion_vector: Optional[torch.Tensor] = None) -> torch.Tensor: # Added type hints
        """Forward pass through the GPT model."""
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 1:
                 input_ids = input_ids.unsqueeze(0)
            else:
                 shape_info = input_ids.shape if isinstance(input_ids, torch.Tensor) else 'N/A'
                 raise ValueError(f"Invalid GPT input_ids: Expected 2D Tensor (Batch, Seq), got {type(input_ids)} shape {shape_info}")

        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            logger.warning("GPT Forward: Received empty input sequence (seq_len=0). Returning empty logits.");
            return torch.zeros(batch_size, 0, self.vocab_size, device=DEVICE)
        if input_ids.device != DEVICE: input_ids = input_ids.to(DEVICE)

        try:
            embedded = self.embedding(input_ids) * math.sqrt(self.embed_dim);
            pos_encoding = self._get_positional_encoding(seq_len, batch_size);
            x = embedded + pos_encoding;
        except IndexError as e:
             max_id = input_ids.max().item() if input_ids.numel() > 0 else 'N/A'
             logger.error(f"GPT Forward: Embedding IndexError (likely invalid token ID {max_id} outside [0, {self.vocab_size-1}]): {e}. Input shape: {input_ids.shape}. Returning zeros.");
             return torch.zeros(batch_size, seq_len, self.vocab_size, device=DEVICE)
        except Exception as e:
             logger.error(f"GPT Forward: Error during embedding/positional encoding: {e}", exc_info=True);
             return torch.zeros(batch_size, seq_len, self.vocab_size, device=DEVICE)
        if not is_safe(x):
            logger.warning("GPT Forward: Unsafe tensor after embedding+pos encoding. Using zeros."); x = torch.zeros_like(x)

        if emotion_vector is not None:
             if not isinstance(emotion_vector, torch.Tensor):
                 try: emotion_vector = torch.tensor(emotion_vector, dtype=torch.float32)
                 except Exception: logger.warning("GPT Forward: Could not convert emotion_vector. Ignoring bias."); emotion_vector=None

             if emotion_vector is not None:
                emotion_vector = emotion_vector.to(DEVICE)
                if not is_safe(emotion_vector):
                     logger.warning("GPT Forward: Unsafe emotion vector provided. Ignoring bias.")
                else:
                    if emotion_vector.ndim == 1: emotion_vector = emotion_vector.unsqueeze(0)
                    if emotion_vector.shape[0] != batch_size:
                         logger.warning(f"GPT Emotion vector batch size mismatch ({emotion_vector.shape[0]} vs {batch_size}). Broadcasting first element.")
                         if emotion_vector.shape[0] > 0:
                             emotion_vector = emotion_vector[0].unsqueeze(0).repeat(batch_size, 1)
                         else:
                             emotion_vector = torch.zeros(batch_size, 4, device=DEVICE)

                    num_features_for_bias = 4
                    emo_for_bias = emotion_vector[:, :num_features_for_bias]

                    if emo_for_bias.shape[1] < num_features_for_bias:
                         pad_size = num_features_for_bias - emo_for_bias.shape[1]
                         emo_for_bias = F.pad(emo_for_bias, (0, pad_size))

                    try:
                        emotion_bias = self.emotion_bias_layer(emo_for_bias.float()).unsqueeze(1).repeat(1, seq_len, 1)
                        if is_safe(emotion_bias):
                            x = x + emotion_bias * 0.2
                        else:
                            logger.warning("GPT Forward: Calculated emotion bias is unsafe. Skipping.")
                    except Exception as e:
                        logger.warning(f"GPT Forward: Failed to apply emotion bias: {e}", exc_info=False)

        tgt_mask = self._generate_square_subsequent_mask(seq_len);
        if PAD_TOKEN_ID is None: raise ValueError("PAD_TOKEN_ID not set before forward pass")
        padding_mask = (input_ids == PAD_TOKEN_ID)

        try:
             transformer_output = self.transformer_encoder(x, mask=tgt_mask, src_key_padding_mask=padding_mask);
             if not is_safe(transformer_output):
                 logger.warning("GPT Forward: Unsafe output from Transformer Encoder. Returning zeros."); return torch.zeros(batch_size, seq_len, self.vocab_size, device=DEVICE)
        except Exception as e:
             logger.error(f"Error occurred in GPT Transformer Encoder: {e}", exc_info=True);
             return torch.zeros(batch_size, seq_len, self.vocab_size, device=DEVICE)

        try:
             logits = self.output(transformer_output);
             if not is_safe(logits):
                 logger.warning("GPT Forward: Unsafe output logits. Returning zeros."); return torch.zeros(batch_size, seq_len, self.vocab_size, device=DEVICE)
        except Exception as e:
             logger.error(f"Error occurred in GPT Output Layer: {e}", exc_info=True);
             return torch.zeros(batch_size, seq_len, self.vocab_size, device=DEVICE)

        return logits

    def train_model(self, dataset: List[Dict[str, str]], epochs: int = Config.NLP.TRAIN_EPOCHS): # Type hint dataset
        """Trains the GPT model on the provided dataset using the BPE tokenizer."""
        if not dataset or not isinstance(dataset, list) or len(dataset) == 0:
            logger.warning("GPT Training skipped: Invalid or empty dataset provided."); self.eval(); return
        if tokenizer is None or PAD_TOKEN_ID is None or START_TOKEN_ID is None or END_TOKEN_ID is None:
             logger.error("GPT Training skipped: Tokenizer or special tokens not initialized."); self.eval(); return

        logger.info(f"Starting SimpleGPT training for {epochs} epochs with {len(dataset)} samples (BPE Tokenizer)..."); self.train()

        for epoch in range(epochs):
            total_loss = 0; num_batches = 0; processed_samples = 0
            random.shuffle(dataset)
            for i, data_item in enumerate(dataset):
                if not isinstance(data_item, dict):
                    logger.warning(f"Skipping training item {i}: Not a dictionary."); continue
                output_text = data_item.get("output", "");
                if not output_text or not isinstance(output_text, str):
                    logger.warning(f"Skipping training item {i}: Invalid or missing 'output' string."); continue

                try:
                    token_ids_content = tokenize(output_text, max_length=self.max_len - 2)
                    token_ids = [START_TOKEN_ID] + token_ids_content + [END_TOKEN_ID]
                except Exception as e:
                    logger.warning(f"Skipping item {i}: Error tokenizing '{output_text}': {e}"); continue

                if len(token_ids) <= 2: continue

                input_token_ids = token_ids[:-1]
                target_token_ids = token_ids[1:]
                current_len = len(input_token_ids)

                pad_len = self.max_len - current_len
                if pad_len < 0:
                    logger.error(f"GPT Train: Negative padding length ({pad_len}) for item {i}. Skipping."); continue

                input_padded = input_token_ids + [PAD_TOKEN_ID] * pad_len
                target_padded = target_token_ids + [PAD_TOKEN_ID] * pad_len

                input_padded = input_padded[:self.max_len]
                target_padded = target_padded[:self.max_len]

                try:
                    input_tensor = torch.tensor([input_padded], dtype=torch.long, device=DEVICE);
                    target_tensor = torch.tensor([target_padded], dtype=torch.long, device=DEVICE);
                except Exception as e:
                    logger.error(f"GPT Train: Error creating tensors for item {i}: {e}. Skipping."); continue

                if input_tensor.shape != (1, self.max_len) or target_tensor.shape != (1, self.max_len):
                     logger.error(f"GPT Train: Shape mismatch after padding for item {i} ({input_tensor.shape}, {target_tensor.shape}). Expected (1, {self.max_len}). Skipping."); continue

                emotion_weights = data_item.get("emotion_weights", [0.0] * 4);
                if not isinstance(emotion_weights, (list, tuple)): emotion_weights = [0.0] * 4
                emo_input_list = (list(emotion_weights) + [0.0] * 4)[:4]
                try:
                    emotions_input = torch.tensor([emo_input_list], device=DEVICE, dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"Skipping item {i}: Failed to create emotion tensor: {e}. Using zeros.");
                    emotions_input = torch.zeros(1, 4, device=DEVICE)

                try:
                    self.optimizer.zero_grad();
                    logits = self(input_ids=input_tensor, emotion_vector=emotions_input)

                    logits_view = logits.view(-1, self.vocab_size);
                    target_view = target_tensor.view(-1)

                    if logits_view.shape[0] != target_view.shape[0]:
                        logger.error(f"GPT Train: Shape mismatch logits {logits_view.shape} vs target {target_view.shape} item {i}. Skipping.");
                        continue

                    loss = self.loss_fn(logits_view, target_view)

                    if is_safe(loss) and loss.requires_grad:
                        loss.backward();
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=Config.NLP.GRADIENT_CLIP_GPT);
                        self.optimizer.step();
                        total_loss += loss.item(); num_batches += 1; processed_samples += 1
                    elif not loss.requires_grad and abs(loss.item()) > 1e-7:
                         logger.debug(f"Loss requires no grad: {loss.item()}")
                         pass
                    elif not is_safe(loss):
                        logger.warning(f"Unsafe loss detected ({loss.item()}) during training epoch {epoch+1}, item {i}. Skipping gradient update.");
                        self.optimizer.zero_grad()

                except Exception as e:
                    logger.error(f"Error during GPT training step for item {i}: {e}", exc_info=True);
                    self.optimizer.zero_grad()

            if num_batches > 0:
                avg_loss = total_loss / num_batches;
                logger.info(f"GPT Training Epoch {epoch + 1}/{epochs}, Processed Samples: {processed_samples}/{len(dataset)}, Avg Loss: {avg_loss:.4f}")
            else:
                logger.warning(f"GPT Training Epoch {epoch + 1}/{epochs}: No valid batches processed.")

        logger.info("SimpleGPT training finished."); self.eval()


    def generate(self, context: Optional[str], emotions: Optional[torch.Tensor], max_len: int = Config.NLP.MAX_RESPONSE_LEN) -> str:
        """Generates text based on context and emotions using the BPE tokenizer."""
        self.eval();
        if tokenizer is None or PAD_TOKEN_ID is None or START_TOKEN_ID is None or END_TOKEN_ID is None or UNK_TOKEN_ID is None:
             logger.error("Cannot generate: Tokenizer or special tokens not initialized."); return "Error: Tokenizer missing."

        input_ids: List[int] = [START_TOKEN_ID]

        if context and isinstance(context, str) and context.strip():
            try:
                context_tokens = tokenize(context, max_length=max_len // 2)
                input_ids.extend(context_tokens)
            except Exception as e:
                logger.error(f"GPT Gen: Error tokenizing context '{context}': {e}. Using START only.")

        input_ids = input_ids[:max_len - 1]
        if not input_ids: input_ids = [START_TOKEN_ID]

        try:
            output_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        except Exception as e:
            logger.error(f"GPT Gen: Failed create input tensor from IDs {input_ids}: {e}. Fallback."); return "..."

        emotions_input = torch.zeros(1, 4, device=DEVICE)
        if emotions is not None and isinstance(emotions, torch.Tensor):
            if emotions.device != DEVICE: emotions = emotions.to(DEVICE)
            if is_safe(emotions):
                emo_flat = emotions.flatten()
                num_features = emo_flat.numel()
                if num_features >= 4:
                     emotions_input = emo_flat[:4].unsqueeze(0).float()
                elif num_features > 0:
                     padded_emo = F.pad(emo_flat, (0, 4 - num_features))
                     emotions_input = padded_emo.unsqueeze(0).float()
            else:
                 logger.warning("GPT Gen: Unsafe emotions tensor provided. Using zeros for bias.")

        with torch.no_grad():
            for _ in range(max_len - output_ids.size(1)):
                current_seq_len = output_ids.size(1)
                if current_seq_len == 0: logger.warning("GPT Gen Loop: Sequence empty. Breaking."); break
                if current_seq_len >= max_len: logger.debug("GPT Gen Loop: Max length reached."); break

                try:
                    input_for_forward = output_ids
                    logits = self(input_ids=input_for_forward, emotion_vector=emotions_input)

                    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != input_for_forward.shape[1] or logits.shape[2] != self.vocab_size:
                        logger.warning(f"GPT Gen Loop: Invalid logits shape {logits.shape}. Expected (1, {current_seq_len}, {self.vocab_size}). Breaking."); break;

                    next_token_logits = logits[:, -1, :]
                    next_token_logits[:, PAD_TOKEN_ID] = -float('inf');
                    next_token_logits[:, START_TOKEN_ID] = -float('inf');
                    next_token_logits[:, UNK_TOKEN_ID] = -float('inf');

                    next_token = torch.argmax(next_token_logits, dim=-1)

                    output_ids = torch.cat([output_ids, next_token.unsqueeze(1)], dim=1)

                    if next_token.item() == END_TOKEN_ID: break
                    if output_ids.size(1) >= max_len: break

                except Exception as e:
                    logger.error(f"Error in GPT generation loop step: {e}", exc_info=True); break

        generated_indices_list = output_ids[0].cpu().tolist() if output_ids.numel() > 0 else []

        response = detokenize(generated_indices_list)

        if not response.strip():
            fallback_response = "..."
            logger.debug("GPT generated an empty/whitespace response. Using fallback.")
            if emotions is not None and emotions.numel() >= Config.Agent.EMOTION_DIM and is_safe(emotions):
                try:
                    emo_cpu = emotions.cpu()
                    dominant_emotion_idx = torch.argmax(emo_cpu).item()
                    fallback_responses = ["Hmm.", "Okay.", "I see.", "Really?", "Well...", "Oh!"]
                    if 0 <= dominant_emotion_idx < len(fallback_responses):
                        fallback_response = fallback_responses[dominant_emotion_idx % len(fallback_responses)]
                except Exception as e:
                    logger.warning(f"Error selecting fallback based on emotion: {e}")
            response = fallback_response

        return response.strip()
# --- END OF FILE ai_modules.py ---
