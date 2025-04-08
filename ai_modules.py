# --- START OF FILE ai_modules.py ---

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
from typing import Dict, Optional, Tuple, List, Any
import os
import logging # Ensure logging is imported if not already

# --- transformers imports ---
try:
    # Use datasets library for easier handling (pip install datasets)
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        AddedToken # Keep AddedToken if needed elsewhere, maybe not here
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Keep logger defined globally if possible, or pass it in
    try:
        from config import logger
        logger.critical("Fine-tuning requires 'transformers' and 'datasets' libraries. Please install: pip install transformers datasets accelerate")
    except ImportError:
        print("CRITICAL: Fine-tuning requires 'transformers' and 'datasets' libraries. Please install: pip install transformers datasets accelerate")
    TRANSFORMERS_AVAILABLE = False
# --- END imports ---


# Use MasterConfig
from config import MasterConfig as Config
from config import DEVICE, logger, TRAIN_DATA # Keep TRAIN_DATA for potential fine-tuning
from config import NUM_HEAD_MOVEMENTS # Keep if other modules need it
from utils import is_safe

# --- EmotionalModule (with variable decay - unchanged from last step) ---
class EmotionalModule(nn.Module):
    def __init__(self, input_dim: int = Config.Agent.EMOTION_DIM + 1):
         super().__init__();
         self.input_dim=input_dim;
         self.fc=nn.Sequential(nn.Linear(self.input_dim, 32), nn.ReLU(), nn.Linear(32, Config.Agent.EMOTION_DIM), nn.Sigmoid());
         default_decays = [0.85, 0.75, 0.90, 0.80, 0.95, 0.70]
         num_emotions = Config.Agent.EMOTION_DIM
         if len(default_decays) < num_emotions:
             logger.warning(f"EmotionalModule: Not enough default decays ({len(default_decays)}) for {num_emotions} emotions. Padding with 0.85.")
             default_decays.extend([0.85] * (num_emotions - len(default_decays)))
         elif len(default_decays) > num_emotions:
             logger.warning(f"EmotionalModule: More default decays ({len(default_decays)}) than {num_emotions} emotions. Truncating.")
             default_decays = default_decays[:num_emotions]
         self.decay_rates = torch.tensor(default_decays, device=DEVICE, dtype=torch.float32).unsqueeze(0)
         logger.info(f"EmotionalModule using variable decay rates: {self.decay_rates.cpu().numpy()}")

    def forward(self, state_emo_part_batch: torch.Tensor, reward_batch: torch.Tensor, prev_emotions_batch: torch.Tensor) -> torch.Tensor:
        if not isinstance(state_emo_part_batch, torch.Tensor) or \
           not isinstance(reward_batch, torch.Tensor) or \
           not isinstance(prev_emotions_batch, torch.Tensor):
            logger.error(f"EmoMod Batch Type Err: state={type(state_emo_part_batch)}, rew={type(reward_batch)}, prev={type(prev_emotions_batch)}")
            bs = state_emo_part_batch.shape[0] if isinstance(state_emo_part_batch, torch.Tensor) else \
                 (reward_batch.shape[0] if isinstance(reward_batch, torch.Tensor) and reward_batch.ndim > 0 else 1)
            return torch.zeros(bs, Config.Agent.EMOTION_DIM, device=DEVICE)
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
        if not is_safe(state_emo_part_batch) or not is_safe(reward_batch) or not is_safe(prev_emotions_batch): logger.warning("EmoMod Batch Unsafe."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        try:
            emotion_inputs = torch.cat([state_emo_part_batch, reward_batch.float()], dim=1)
        except Exception as e: logger.error(f"EmoMod Batch Concat Err: {e}."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        expected_input_dim = self.fc[0].in_features
        if emotion_inputs.shape != (batch_size, expected_input_dim): logger.error(f"EmoMod Batch Dim Err after concat. Exp: {(batch_size, expected_input_dim)}, Got: {emotion_inputs.shape}"); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        try: scaled_emotions = self.fc(emotion_inputs); scaled_emotions = torch.clamp(scaled_emotions, 0, 1)
        except Exception as e: logger.error(f"EmoMod Batch FC Err: {e}."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        smoothed_emotions = self.decay_rates * prev_emotions_batch + (1.0 - self.decay_rates) * scaled_emotions;
        final_emotions = torch.clamp(smoothed_emotions, 0, 1)
        if not is_safe(final_emotions): logger.error("EmoMod Batch Unsafe Out."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        return final_emotions

# --- SyntrixKorporator ---
class SyntrixKorporator(nn.Module):
    def __init__(self, input_dim: int = Config.Agent.STATE_DIM, hidden_dim: int = Config.Agent.HIDDEN_DIM, m: int = 8):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim <= 0 or \
           not isinstance(hidden_dim, int) or hidden_dim <= 0: raise ValueError(f"Korporator: invalid dims (in={input_dim}, hidden={hidden_dim}).")
        self.input_dim = input_dim; self.hidden_dim = hidden_dim
        self.metrophor = nn.Parameter(torch.randn(hidden_dim, device=DEVICE) * 0.1)
        if hidden_dim < 2: self.m = 1 if hidden_dim == 1 else 0
        else: self.m = min(m, hidden_dim // 2)
        if self.m <= 0: raise ValueError(f"Korporator: m={self.m} invalid for hidden_dim={hidden_dim}.")
        self.k_input_dim = self.m * 2
        self.Km = nn.Linear(self.k_input_dim, hidden_dim); self.Cm = nn.Linear(self.k_input_dim, hidden_dim)
        self.input_projector = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()).to(DEVICE)
        #logger.debug(f"Korporator initialized: input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, m={self.m}")

    def forward(self, phi_input: torch.Tensor, psi_input: torch.Tensor, level: int = 1) -> torch.Tensor:
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
    def __init__(self, input_dim: int, hidden_dim: int, levels: int = Config.Agent.CASCADE_LEVELS):
        super().__init__()
        self.levels=max(1, int(levels))
        layers=[]
        current_dim=input_dim
        self._output_dim=input_dim
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
            #logger.debug(f"Kaskade initialized: input_dim={self._input_dim}, output_dim={self._output_dim}, levels={self.levels}, actual_layers={self.num_actual_layers}")


    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        if not isinstance(x_input, torch.Tensor): logger.warning("Kaskade NaN input"); return torch.zeros(1, self._output_dim, device=DEVICE)
        if x_input.device != DEVICE: x_input = x_input.to(DEVICE)
        if not is_safe(x_input): logger.warning("Kaskade Unsafe input"); return torch.zeros(1, self._output_dim, device=DEVICE) # Return correct shape on error

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

        if not is_safe(output_batch): logger.error("Kaskade Unsafe output"); return torch.zeros(x_batch.shape[0], self._output_dim, device=DEVICE) # Return correct shape on error

        if output_batch.ndim != 2 or output_batch.shape[1] != self._output_dim:
             logger.error(f"Kaskade Shape Err: Expected output ({self._output_dim},), got {output_batch.shape}");
             try: output_batch = output_batch.view(x_batch.shape[0], self._output_dim)
             except RuntimeError: return torch.zeros(x_batch.shape[0], self._output_dim, device=DEVICE)

        return output_batch.squeeze(0) if was_single_instance else output_batch


# --- TransformerGPT Class (Includes fine-tuning logic and updated generate) ---
class TransformerGPT:
    def __init__(self, model_name: str = Config.NLP.HUGGINGFACE_MODEL):
        if not TRANSFORMERS_AVAILABLE:
            logger.critical("Hugging Face 'transformers' library not found or import failed. TransformerGPT cannot be initialized.")
            raise ImportError("transformers library is required for TransformerGPT")

        logger.info(f"Initializing TransformerGPT with model: {model_name}")
        self.model_name = model_name
        self.device = DEVICE
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.pad_token_id: Optional[int] = None
        self.start_token_id: Optional[int] = None
        self.end_token_id: Optional[int] = None

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add special tokens if they don't exist, mapping to standard names
            special_tokens_to_add = {}
            if self.tokenizer.pad_token is None: special_tokens_to_add['pad_token'] = '<PAD>'
            if self.tokenizer.bos_token is None: special_tokens_to_add['bos_token'] = '<START>'
            if self.tokenizer.eos_token is None: special_tokens_to_add['eos_token'] = '<END>'
            if self.tokenizer.unk_token is None: special_tokens_to_add['unk_token'] = '<UNK>'

            num_added_toks = 0
            if special_tokens_to_add:
                 num_added_toks = self.tokenizer.add_special_tokens(special_tokens_to_add)
                 if num_added_toks > 0:
                     logger.info(f"Added {num_added_toks} special tokens ({list(special_tokens_to_add.keys())}) to tokenizer.")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            # Resize embeddings if new tokens were added
            if num_added_toks > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info("Resized model token embeddings.")

            # Set tokenizer pad token id if it was initially None
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                logger.warning("Tokenizer missing pad token, using EOS token as pad token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Check and assign essential token IDs
            if self.tokenizer.pad_token_id is None: raise ValueError("PAD token ID could not be determined.")
            if self.tokenizer.bos_token_id is None: raise ValueError("BOS/START token ID could not be determined.")
            if self.tokenizer.eos_token_id is None: raise ValueError("EOS/END token ID could not be determined.")

            self.pad_token_id = self.tokenizer.pad_token_id
            self.start_token_id = self.tokenizer.bos_token_id
            self.end_token_id = self.tokenizer.eos_token_id

            logger.info(f"Transformer model '{model_name}' and tokenizer loaded successfully.")
            self.model.eval() # Set to eval mode initially

        except OSError as e:
             logger.critical(f"Could not load Hugging Face model/tokenizer '{model_name}'. Is it a valid model name and are you online? Error: {e}", exc_info=True)
             raise RuntimeError(f"Failed to load HF model: {model_name}") from e
        except Exception as e:
             logger.critical(f"Unexpected error loading TransformerGPT: {e}", exc_info=True)
             raise RuntimeError("Failed to initialize TransformerGPT") from e

    def train_model(self, dataset: List[Dict[str, Any]], epochs: int = Config.NLP.TRAIN_EPOCHS):
         """Fine-tunes the Hugging Face model using the Trainer API."""
         if not TRANSFORMERS_AVAILABLE:
             logger.error("Cannot fine-tune: transformers library not available.")
             return
         if not self.model or not self.tokenizer:
             logger.error("Cannot fine-tune: Model or Tokenizer not initialized.")
             return
         if not dataset:
             logger.warning("Cannot fine-tune: No training data provided.")
             return

         logger.info(f"Starting fine-tuning for '{self.model_name}'...")
         self.model.train()

         # 1. Prepare Dataset using datasets library for better handling
         try:
            formatted_texts = []
            for item in dataset:
                output = item.get("output", "")
                situation = item.get("situation", "")
                # Example formatting: Combine situation and output for Causal LM
                # Add BOS/EOS tokens to frame the sequence
                # Ensure BOS/EOS exist in the tokenizer
                bos = self.tokenizer.bos_token if self.tokenizer.bos_token else ""
                eos = self.tokenizer.eos_token if self.tokenizer.eos_token else ""
                text = f"{bos}{situation} >> {output}{eos}"
                if output: # Only include if there's an output
                    formatted_texts.append({"text": text})

            if not formatted_texts:
                 logger.error("Cannot fine-tune: No text data generated from dataset.")
                 self.model.eval(); return

            hf_dataset = Dataset.from_list(formatted_texts)
            logger.info(f"Created Hugging Face dataset with {len(hf_dataset)} samples.")

         except Exception as e:
             logger.error(f"Error creating Hugging Face dataset: {e}", exc_info=True)
             self.model.eval(); return

         # 2. Tokenize Dataset using map for efficiency
         def tokenize_function(examples):
             # Tokenize texts, truncation is important
             return self.tokenizer(examples["text"], truncation=True, padding=False) # Don't pad here, collator will handle it

         try:
             tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"]) # Use multiple procs
             logger.info("Tokenized dataset using map.")
         except Exception as e:
             logger.error(f"Error tokenizing dataset map: {e}", exc_info=True)
             self.model.eval(); return

         # 3. Block Processing (Group texts into blocks)
         block_size = 128 # Adjust based on model/memory
         def group_texts(examples):
             # Concatenate all texts.
             concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
             total_length = len(concatenated_examples[list(examples.keys())[0]])
             # We drop the small remainder
             if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
             else: # Handle case where total length < block_size
                 total_length = 0
             # Split by chunks of block_size.
             result = {
                 k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                 for k, t in concatenated_examples.items()
             }
             # IMPORTANT: Create labels for Causal LM
             result["labels"] = result["input_ids"].copy()
             return result

         try:
             lm_datasets = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
             logger.info(f"Processed dataset into blocks of size {block_size}. New size: {len(lm_datasets)}")
             if len(lm_datasets) == 0:
                 logger.error("Dataset processing resulted in 0 blocks. Check block_size and data length.")
                 self.model.eval(); return
         except Exception as e:
             logger.error(f"Error grouping texts: {e}", exc_info=True)
             self.model.eval(); return


         # 4. Data Collator
         data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

         # 5. Training Arguments
         output_dir = Config.GPT_SAVE_PATH # Use path from config
         training_args = TrainingArguments(
             output_dir=output_dir,
             overwrite_output_dir=True,
             num_train_epochs=epochs,
             per_device_train_batch_size=4,  # Lower if OOM
             gradient_accumulation_steps=4, # Increase if lowering batch size
             save_strategy="epoch",
             save_total_limit=2,
             logging_steps=50,
             learning_rate=Config.NLP.GPT_LR,
             weight_decay=0.01,
             warmup_ratio=0.1,
             fp16=torch.cuda.is_available(), # Use mixed precision if possible
             report_to="none", # Disable wanbd/etc logging by default
             label_names=["labels"] # Important for Trainer
         )

         # 6. Initialize Trainer
         trainer = Trainer(
             model=self.model,
             args=training_args,
             train_dataset=lm_datasets, # Use the processed dataset
             data_collator=data_collator,
         )

         # 7. Train
         logger.info("Starting Hugging Face model fine-tuning...")
         try:
             train_result = trainer.train()
             logger.info("Fine-tuning finished.")
             trainer.save_model() # Save final model to output_dir
             self.tokenizer.save_pretrained(training_args.output_dir) # Save tokenizer too
             logger.info(f"Fine-tuned model saved to {training_args.output_dir}")

             # Log metrics
             metrics = train_result.metrics
             logger.info(f"TrainOutput: {metrics}")

         except Exception as e:
             logger.error(f"Error during fine-tuning: {e}", exc_info=True)

         self.model.eval() # Set back to eval mode

    def generate(self, context: Optional[str], emotions: Optional[torch.Tensor]=None, # Emotions used for prompt building now
                 max_len: int = Config.NLP.MAX_RESPONSE_LEN,
                 temperature: float = Config.NLP.GPT_TEMPERATURE,
                 top_p: float = Config.NLP.GPT_TOP_P,
                 num_beams: int = 1,
                 repetition_penalty: float = 1.15, # <-- Penalty added
                ) -> str:
        """Generates text using the Hugging Face model's generate method."""
        if not self.model or not self.tokenizer:
            logger.error("Cannot generate: Model or Tokenizer not initialized.")
            return "[Generation Error: Model not ready]"
        self.model.eval()

        # Use provided context directly. Orchestrator should format it.
        prompt = context if context else ""
        # If prompt is empty, start with BOS token
        if not prompt and self.tokenizer.bos_token:
             prompt = self.tokenizer.bos_token

        try:
            # Determine max length for prompt based on model's capacity and desired output length
            model_max_len = getattr(self.model.config, 'n_positions', 512) # Use n_positions or n_ctx
            # Ensure max_len is reasonable
            max_output_len = max(5, min(max_len, model_max_len // 2))
            # Leave some buffer for safety
            max_prompt_len = model_max_len - max_output_len - 10
            if max_prompt_len <= 0: max_prompt_len = model_max_len // 2 # Fallback if max_len is too large

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(self.device)
            input_length = inputs.input_ids.shape[1]

            # Ensure EOS token is used for padding if PAD is not set correctly
            gen_pad_token_id = self.pad_token_id if self.pad_token_id is not None else self.end_token_id

            with torch.no_grad():
                # Use generate method with updated parameters
                output_sequences = self.model.generate(
                    **inputs,
                    max_new_tokens=max_output_len,
                    num_beams=num_beams,
                    do_sample=(num_beams == 1 and temperature > 0), # Sample only if temp > 0 and not using beam search
                    temperature=temperature if num_beams == 1 and temperature > 0 else 1.0,
                    top_p=top_p if num_beams == 1 and top_p < 1.0 else None,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=gen_pad_token_id, # Use determined pad token ID
                    eos_token_id=self.end_token_id, # Use EOS token ID
                    # bos_token_id=self.start_token_id, # Typically not needed in generate args
                )

            # Extract only the generated part, excluding the prompt
            generated_ids = output_sequences[0, input_length:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            logger.error(f"Error during HF model generation: {e}", exc_info=True)
            response = "[Generation Error]"

        if not response.strip():
             response = "..."
             logger.warning("GPT generated an empty string after decoding.")

        return response.strip()

    def save_model(self, path: str): # Wrapper for consistency
         if not self.model or not self.tokenizer:
             logger.error("Cannot save HF model: Not initialized.")
             return
         try:
            logger.info(f"Saving HF model and tokenizer to {path}...")
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info("HF model and tokenizer saved.")
         except Exception as e:
             logger.error(f"Failed to save HF model/tokenizer: {e}", exc_info=True)

    def load_model(self, path: str): # Wrapper for consistency
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Cannot load HF model: transformers library not available.")
            return False

        # Determine the actual path to load from
        # Priority: Provided path (if it's a directory), Config.GPT_SAVE_PATH, Base model name
        load_path = path # Start with the provided path argument
        if not os.path.isdir(load_path):
            logger.debug(f"Provided path '{load_path}' is not a directory. Checking config save path...")
            config_save_path = Config.GPT_SAVE_PATH
            if os.path.isdir(config_save_path):
                logger.info(f"Loading fine-tuned model from config save path: {config_save_path}")
                load_path = config_save_path
            else:
                logger.warning(f"Neither provided path '{path}' nor config save path '{config_save_path}' found. Attempting to load base model '{self.model_name}' from Hub.")
                load_path = self.model_name # Fallback to base model name
                # Check again if path exists now (it shouldn't if it's a model name)
                if os.path.isdir(load_path):
                    logger.error(f"Unexpected directory found for base model name '{load_path}'. Load failed.")
                    return False

        try:
            logger.info(f"Loading HF model and tokenizer from {load_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(load_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)

            # Re-assign special token IDs after loading (crucial!)
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None: raise ValueError("PAD token ID missing after load.")
            if self.tokenizer.bos_token_id is None: raise ValueError("BOS token ID missing after load.")
            if self.tokenizer.eos_token_id is None: raise ValueError("EOS token ID missing after load.")
            self.pad_token_id = self.tokenizer.pad_token_id
            self.start_token_id = self.tokenizer.bos_token_id
            self.end_token_id = self.tokenizer.eos_token_id
            logger.info(f"HF model and tokenizer loaded successfully from {load_path}.")
            self.model.eval()
            return True
        except OSError as e:
            logger.error(f"OSError loading HF model/tokenizer from {load_path}. Model name/path likely incorrect or requires internet. Error: {e}", exc_info=True)
            self.model = None; self.tokenizer = None; return False
        except Exception as e:
             logger.error(f"Failed to load HF model/tokenizer from {load_path}: {e}", exc_info=True)
             self.model = None; self.tokenizer = None; return False


# --- END OF FILE ai_modules.py ---
