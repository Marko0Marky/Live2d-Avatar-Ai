# --- START OF FILE ai_modules.py ---

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
from typing import Dict, Optional, Tuple, List, Any
import os
import logging

# --- transformers imports ---
try:
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        AddedToken
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    try: from config import logger; logger.critical("Install transformers/datasets/accelerate for fine-tuning.")
    except ImportError: print("CRITICAL: Install transformers/datasets/accelerate for fine-tuning."); logging.basicConfig(level=logging.CRITICAL); logger = logging.getLogger(__name__); logger.critical("Transformers/datasets missing AND config logger failed.")
    TRANSFORMERS_AVAILABLE = False

# Use MasterConfig object
from config import MasterConfig as Config
from config import DEVICE, logger, TRAIN_DATA # Keep TRAIN_DATA for potential fine-tuning
# *** Import the PATH constant directly ***
from config import GPT_SAVE_PATH as DEFAULT_GPT_SAVE_PATH
# *** End Import ***
from utils import is_safe

# --- EmotionalModule ---
class EmotionalModule(nn.Module):
    def __init__(self, input_dim: int = Config.Agent.EMOTION_DIM + 1):
         super().__init__();
         self.input_dim=input_dim;
         self.fc=nn.Sequential(
             nn.Linear(self.input_dim, 32), nn.ReLU(),
             nn.Linear(32, Config.Agent.EMOTION_DIM), nn.Sigmoid()
         ).to(DEVICE) # Ensure module is on device
         default_decays = [0.85, 0.75, 0.90, 0.80, 0.95, 0.70][:Config.Agent.EMOTION_DIM]
         num_emotions = Config.Agent.EMOTION_DIM
         if len(default_decays) < num_emotions:
             logger.warning(f"EmotionalModule: Not enough default decays ({len(default_decays)}) for {num_emotions} emotions. Padding with 0.85.")
             default_decays.extend([0.85] * (num_emotions - len(default_decays)))
         self.decay_rates = torch.tensor(default_decays, device=DEVICE, dtype=torch.float32).unsqueeze(0)
         logger.info(f"EmotionalModule initialized input={self.input_dim}, output={Config.Agent.EMOTION_DIM}.")

    def forward(self, state_emo_part_batch: torch.Tensor, reward_batch: torch.Tensor, prev_emotions_batch: torch.Tensor) -> torch.Tensor:
        """ Processes emotion-related state inputs and rewards to update emotions. """
        if not all(isinstance(t, torch.Tensor) for t in [state_emo_part_batch, reward_batch, prev_emotions_batch]):
             bs = state_emo_part_batch.shape[0] if isinstance(state_emo_part_batch, torch.Tensor) else 1; logger.error(f"EmoMod Batch Type Err."); return torch.zeros(bs, Config.Agent.EMOTION_DIM, device=DEVICE)
        batch_size = state_emo_part_batch.shape[0]
        expected_state_shape = (batch_size, Config.Agent.EMOTION_DIM)
        if reward_batch.ndim == 1 and reward_batch.shape[0] == batch_size: reward_batch = reward_batch.unsqueeze(1)
        expected_reward_shape = (batch_size, 1)
        expected_prev_shape = (batch_size, Config.Agent.EMOTION_DIM)
        if state_emo_part_batch.shape != expected_state_shape or \
           reward_batch.shape != expected_reward_shape or \
           prev_emotions_batch.shape != expected_prev_shape:
             logger.error(f"EmoMod Batch Shape Err."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        if not all(map(is_safe, [state_emo_part_batch, reward_batch, prev_emotions_batch])): logger.warning("EmoMod Batch Unsafe."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        try: emotion_inputs = torch.cat([state_emo_part_batch, reward_batch.float()], dim=1)
        except Exception as e: logger.error(f"EmoMod Batch Concat Err: {e}."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        expected_input_shape = (batch_size, self.input_dim)
        if emotion_inputs.shape != expected_input_shape: logger.error(f"EmoMod Batch Dim Err after concat."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        try: scaled_emotions = self.fc(emotion_inputs); scaled_emotions = torch.clamp(scaled_emotions, 0, 1)
        except Exception as e: logger.error(f"EmoMod Batch FC Err: {e}."); return torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)
        smoothed_emotions = self.decay_rates * prev_emotions_batch + (1.0 - self.decay_rates) * scaled_emotions;
        final_emotions = torch.clamp(smoothed_emotions, 0, 1)
        return final_emotions if is_safe(final_emotions) else torch.zeros(batch_size, Config.Agent.EMOTION_DIM, device=DEVICE)


# --- TransformerGPT Class (Includes fine-tuning logic and updated generate) ---
class TransformerGPT:
    def __init__(self, model_name: str = Config.NLP.HUGGINGFACE_MODEL):
        if not TRANSFORMERS_AVAILABLE: raise ImportError("transformers/datasets required for TransformerGPT")
        logger.info(f"Initializing TransformerGPT with model: {model_name}")
        self.model_name = model_name; self.device = DEVICE
        self.model: Optional[AutoModelForCausalLM] = None; self.tokenizer: Optional[AutoTokenizer] = None
        self.pad_token_id: Optional[int] = None; self.start_token_id: Optional[int] = None; self.end_token_id: Optional[int] = None
        if not self._load_hf_model_and_tokenizer(model_name): # Load on init
             raise RuntimeError(f"Failed to load initial HF model/tokenizer: {model_name}")

    def _load_hf_model_and_tokenizer(self, model_name_or_path: str):
        """Loads the Hugging Face model and tokenizer."""
        try:
            logger.info(f"Attempting to load Tokenizer: {model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            special_tokens_to_add = {}
            if self.tokenizer.pad_token is None: special_tokens_to_add['pad_token'] = '<PAD>'
            if self.tokenizer.bos_token is None: special_tokens_to_add['bos_token'] = '<START>'
            if self.tokenizer.eos_token is None: special_tokens_to_add['eos_token'] = '<END>'
            if self.tokenizer.unk_token is None: special_tokens_to_add['unk_token'] = '<UNK>'
            num_added_toks = 0
            if special_tokens_to_add:
                 num_added_toks = self.tokenizer.add_special_tokens(special_tokens_to_add, replace_additional_special_tokens=False)
                 if num_added_toks > 0: logger.info(f"Added {num_added_toks} special tokens.")
            logger.info(f"Attempting to load Model: {model_name_or_path}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
            if num_added_toks > 0: self.model.resize_token_embeddings(len(self.tokenizer)); logger.info("Resized model token embeddings.")
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None: self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None: raise ValueError("PAD token ID unknown.")
            if self.tokenizer.bos_token_id is None: raise ValueError("BOS token ID unknown.")
            if self.tokenizer.eos_token_id is None: raise ValueError("EOS token ID unknown.")
            self.pad_token_id = self.tokenizer.pad_token_id; self.start_token_id = self.tokenizer.bos_token_id; self.end_token_id = self.tokenizer.eos_token_id
            logger.info(f"HF model '{model_name_or_path}' & tokenizer loaded.")
            self.model.eval()
            return True
        except OSError as e: logger.critical(f"Cannot load HF model/tokenizer '{model_name_or_path}'. Valid name/path? Online? Error: {e}"); return False
        except Exception as e: logger.critical(f"Error loading TransformerGPT: {e}", exc_info=True); return False

    # FIX: Change default argument for output_dir
    def train_model(self, dataset: List[Dict[str, Any]], epochs: int = Config.NLP.TRAIN_EPOCHS, output_dir: Optional[str] = None):
        """Fine-tunes the Hugging Face model using the Trainer API."""
        if not TRANSFORMERS_AVAILABLE: logger.error("Cannot fine-tune: libs missing."); return
        if not self.model or not self.tokenizer: logger.error("Cannot fine-tune: Model/Tokenizer missing."); return
        if not dataset: logger.warning("Cannot fine-tune: No training data."); return

        # FIX: Get default path inside the method if not provided
        if output_dir is None:
            output_dir = DEFAULT_GPT_SAVE_PATH # Use the imported constant path

        logger.info(f"Starting fine-tuning for '{self.model_name}' on {len(dataset)} examples -> {output_dir}")
        self.model.train()
        try:
            formatted_texts = []
            for item in dataset:
                output = item.get("output", ""); situation = item.get("situation", "")
                bos = self.tokenizer.bos_token or ""; eos = self.tokenizer.eos_token or ""
                text = f"{bos}{situation} >> {output}{eos}"
                if output: formatted_texts.append({"text": text})
            if not formatted_texts: logger.error("No text data for fine-tuning."); self.model.eval(); return
            hf_dataset = Dataset.from_list(formatted_texts)
        except Exception as e: logger.error(f"Error creating HF dataset: {e}"); self.model.eval(); return
        def tokenize_function(examples):
             model_max_len = getattr(self.model.config, 'n_positions', 512)
             return self.tokenizer(examples["text"], truncation=True, padding=False, max_length=model_max_len - 2)
        try: tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, num_proc=max(1, os.cpu_count() // 2), remove_columns=["text"])
        except Exception as e: logger.error(f"Error tokenizing dataset: {e}"); self.model.eval(); return
        block_size = 128
        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}; total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size: total_length = (total_length // block_size) * block_size
            else: return {k: [] for k in examples.keys()} # Return empty if less than block_size
            result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()}
            result["labels"] = result["input_ids"].copy(); return result
        try:
             lm_datasets = tokenized_dataset.map(group_texts, batched=True, num_proc=max(1, os.cpu_count() // 2))
             if len(lm_datasets) == 0: logger.error("Dataset processing resulted in 0 blocks."); self.model.eval(); return
        except Exception as e: logger.error(f"Error grouping texts: {e}"); self.model.eval(); return
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        os.makedirs(output_dir, exist_ok=True) # Ensure resolved output dir exists
        training_args = TrainingArguments(
             output_dir=output_dir, overwrite_output_dir=True, num_train_epochs=epochs,
             per_device_train_batch_size=4, gradient_accumulation_steps=4, save_strategy="epoch",
             save_total_limit=2, logging_steps=50, learning_rate=Config.NLP.GPT_LR, weight_decay=0.01,
             warmup_ratio=0.1, fp16=torch.cuda.is_available(), report_to="none", gradient_checkpointing=False,
             logging_dir=os.path.join(output_dir, 'logs'), )
        trainer = Trainer(model=self.model, args=training_args, train_dataset=lm_datasets, data_collator=data_collator,)
        logger.info("Starting Trainer.train()...")
        try: train_result = trainer.train(); logger.info(f"Fine-tuning finished. Metrics: {train_result.metrics}"); trainer.save_model(); self.tokenizer.save_pretrained(output_dir); logger.info(f"Model saved to {output_dir}")
        except Exception as e: logger.error(f"Error during trainer.train(): {e}", exc_info=True)
        self.model.eval()

    def generate(self, context: Optional[str], emotions: Optional[torch.Tensor]=None,
                 max_len: int = Config.NLP.MAX_RESPONSE_LEN,
                 temperature: float = Config.NLP.GPT_TEMPERATURE,
                 top_p: float = Config.NLP.GPT_TOP_P,
                 num_beams: int = 1,
                 repetition_penalty: float = 1.15
                ) -> str:
        """Generates text using the Hugging Face model's generate method."""
        if not self.model or not self.tokenizer: logger.error("Cannot generate: Model/Tokenizer missing."); return "[Gen Error: No Model]"
        self.model.eval()
        prompt = context if context else ""
        if not prompt.strip() and self.tokenizer.bos_token: prompt = self.tokenizer.bos_token + " "
        elif prompt.strip() and not prompt.endswith((" ","\n", ":")): prompt += " "
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            input_length = inputs.input_ids.shape[1]
            gen_temp = temperature if num_beams == 1 and temperature > 0.01 else 1.0
            gen_top_p = top_p if num_beams == 1 and 0.0 < top_p < 1.0 else None
            gen_pad_token_id = self.pad_token_id if self.pad_token_id is not None else self.tokenizer.eos_token_id

            with torch.no_grad():
                output_sequences = self.model.generate(
                    **inputs, max_new_tokens=max_len, num_beams=num_beams,
                    do_sample=(num_beams == 1 and gen_temp > 0.01), temperature=gen_temp, top_p=gen_top_p,
                    repetition_penalty=repetition_penalty, pad_token_id=gen_pad_token_id, eos_token_id=self.end_token_id, )
            response = self.tokenizer.decode(output_sequences[0, input_length:], skip_special_tokens=True)
        except Exception as e: logger.error(f"Error during HF generation: {e}", exc_info=True); response = "[Gen Error]"
        response = response.strip(); return response if response else "..."

    def save_model(self, path: str):
         """ Saves the model and tokenizer to the specified directory path. """
         if not self.model or not self.tokenizer: logger.error("Cannot save HF model: Not initialized."); return
         try: logger.info(f"Saving HF model/tokenizer to: {path}..."); os.makedirs(path, exist_ok=True); self.model.save_pretrained(path); self.tokenizer.save_pretrained(path); logger.info(f"HF model/tokenizer saved.")
         except Exception as e: logger.error(f"Failed to save HF model/tokenizer to {path}: {e}", exc_info=True)

    def load_model(self, path: str) -> bool:
        """ Loads model/tokenizer. Returns True on success. """
        logger.info(f"Attempting to load HF model/tokenizer from: {path}...")
        load_path = path; fallback_used = False
        if not os.path.isdir(load_path): logger.warning(f"Path '{load_path}' not found/dir. Fallback to base: {self.model_name}"); load_path = self.model_name; fallback_used = True
        loaded = self._load_hf_model_and_tokenizer(load_path)
        if not loaded and not fallback_used: # Try base model if specific path failed
            logger.warning(f"Specific path '{path}' failed, trying base model '{self.model_name}' again.")
            loaded = self._load_hf_model_and_tokenizer(self.model_name)
        return loaded

# --- END OF FILE ai_modules.py ---
