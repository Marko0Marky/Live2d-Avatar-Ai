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
    from transformers import ( AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    try: from config import logger; logger.critical("Install transformers/datasets/accelerate for fine-tuning.")
    except ImportError: print("CRITICAL: Install transformers/datasets/accelerate for fine-tuning."); logging.basicConfig(level=logging.CRITICAL); logger = logging.getLogger(__name__); logger.critical("Transformers/datasets missing AND config logger failed.")
    TRANSFORMERS_AVAILABLE = False

# Use MasterConfig object
from config import MasterConfig as Config
from config import DEVICE, logger, TRAIN_DATA
# Import the PATH constant directly
from config import GPT_SAVE_PATH as DEFAULT_GPT_SAVE_PATH
from utils import is_safe

# --- EmotionalModule ---
class EmotionalModule(nn.Module):
    def __init__(self, input_dim: int = Config.Agent.EMOTION_DIM + 1):
         super().__init__(); self.input_dim=input_dim;
         self.fc=nn.Sequential( nn.Linear(self.input_dim, 32), nn.ReLU(), nn.Linear(32, Config.Agent.EMOTION_DIM), nn.Sigmoid()).to(DEVICE)
         default_decays = [0.85, 0.75, 0.90, 0.80, 0.95, 0.70][:Config.Agent.EMOTION_DIM]
         if len(default_decays) < Config.Agent.EMOTION_DIM: default_decays.extend([0.85]*(Config.Agent.EMOTION_DIM-len(default_decays)))
         self.decay_rates = torch.tensor(default_decays, device=DEVICE, dtype=torch.float32).unsqueeze(0)
         logger.info(f"EmotionalModule initialized input={self.input_dim}, output={Config.Agent.EMOTION_DIM}.")
    def forward(self, state_emo_part_batch: torch.Tensor, reward_batch: torch.Tensor, prev_emotions_batch: torch.Tensor) -> torch.Tensor:
        bs = state_emo_part_batch.shape[0]
        if not all(isinstance(t,torch.Tensor) for t in [state_emo_part_batch,reward_batch,prev_emotions_batch]): return torch.zeros(bs, Config.Agent.EMOTION_DIM, device=DEVICE)
        if reward_batch.ndim == 1: reward_batch = reward_batch.unsqueeze(1)
        if state_emo_part_batch.shape!=(bs, Config.Agent.EMOTION_DIM) or reward_batch.shape!=(bs,1) or prev_emotions_batch.shape!=(bs, Config.Agent.EMOTION_DIM): logger.error("EmoMod Shape Err."); return torch.zeros(bs, Config.Agent.EMOTION_DIM, device=DEVICE)
        if not all(map(is_safe, [state_emo_part_batch,reward_batch,prev_emotions_batch])): logger.warning("EmoMod Unsafe Input."); return torch.zeros(bs, Config.Agent.EMOTION_DIM, device=DEVICE)
        try: emotion_inputs = torch.cat([state_emo_part_batch, reward_batch.float()], dim=1)
        except Exception as e: logger.error(f"EmoMod Concat Err: {e}."); return torch.zeros(bs, Config.Agent.EMOTION_DIM, device=DEVICE)
        if emotion_inputs.shape[1] != self.input_dim: logger.error(f"EmoMod Dim Err post-concat."); return torch.zeros(bs, Config.Agent.EMOTION_DIM, device=DEVICE)
        try: scaled_emotions = torch.clamp(self.fc(emotion_inputs), 0, 1)
        except Exception as e: logger.error(f"EmoMod FC Err: {e}."); return torch.zeros(bs, Config.Agent.EMOTION_DIM, device=DEVICE)
        final_emotions = torch.clamp(self.decay_rates * prev_emotions_batch + (1.0 - self.decay_rates) * scaled_emotions, 0, 1)
        return final_emotions if is_safe(final_emotions) else torch.zeros_like(final_emotions)

# --- TransformerGPT Class ---
class TransformerGPT:
    def __init__(self, model_name: str = Config.NLP.HUGGINGFACE_MODEL):
        if not TRANSFORMERS_AVAILABLE: raise ImportError("transformers/datasets required for TransformerGPT")
        logger.info(f"Initializing TransformerGPT with model: {model_name}")
        self.model_name = model_name; self.device = DEVICE
        self.model: Optional[AutoModelForCausalLM] = None; self.tokenizer: Optional[AutoTokenizer] = None
        self.pad_token_id: Optional[int]=None; self.start_token_id: Optional[int]=None; self.end_token_id: Optional[int]=None
        if not self._load_hf_model_and_tokenizer(model_name): raise RuntimeError(f"Failed load HF model: {model_name}")

    def _load_hf_model_and_tokenizer(self, model_name_or_path: str):
        try:
            logger.info(f"Attempt load Tokenizer: {model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            special_tokens = {'pad_token': '<PAD>', 'bos_token': '<START>', 'eos_token': '<END>', 'unk_token': '<UNK>'}
            current_tokens = {k: v for k, v in special_tokens.items() if getattr(self.tokenizer, k) is None}
            num_added = self.tokenizer.add_special_tokens(current_tokens, replace_additional_special_tokens=False) if current_tokens else 0
            if num_added > 0: logger.info(f"Added {num_added} special tokens.")
            logger.info(f"Attempt load Model: {model_name_or_path}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)
            if num_added > 0: self.model.resize_token_embeddings(len(self.tokenizer)); logger.info("Resized embeddings.")
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            if any(t is None for t in [self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]): raise ValueError("Essential tokens missing")
            self.pad_token_id=self.tokenizer.pad_token_id; self.start_token_id=self.tokenizer.bos_token_id; self.end_token_id=self.tokenizer.eos_token_id
            self.model.eval(); logger.info(f"HF model '{model_name_or_path}' loaded."); return True
        except Exception as e: logger.critical(f"Failed load TransformerGPT '{model_name_or_path}': {e}", exc_info=True); return False

    def train_model(self, dataset: List[Dict[str, Any]], epochs: int = Config.NLP.TRAIN_EPOCHS, output_dir: Optional[str] = None):
        """Fine-tunes the Hugging Face model using the Trainer API."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Cannot fine-tune: transformers/datasets library not available.")
            return
        if not self.model or not self.tokenizer:
            logger.error("Cannot fine-tune: Model or Tokenizer not initialized.")
            return
        if not dataset:
            logger.warning("Cannot fine-tune: No training data provided.")
            return

        # Resolve output directory using the imported default if None provided
        output_dir = output_dir or DEFAULT_GPT_SAVE_PATH
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

        logger.info(f"Starting fine-tuning for '{self.model_name}' on {len(dataset)} examples -> {output_dir}")
        self.model.train() # Set model to training mode

        # 1. Prepare Dataset
        try:
            formatted_texts = []
            bos = self.tokenizer.bos_token or ""
            eos = self.tokenizer.eos_token or ""
            for item in dataset:
                output = item.get("output", "")
                situation = item.get("situation", "") # Use situation as context
                # Simple formatting: BOS + Situation + Separator + Output + EOS
                text = f"{bos}{situation} >> {output}{eos}"
                if output: # Only add if output is non-empty
                    formatted_texts.append({"text": text})

            if not formatted_texts:
                logger.error("Cannot fine-tune: No valid text data generated from input dataset.")
                self.model.eval(); return # Revert to eval mode

            hf_dataset = Dataset.from_list(formatted_texts)
            logger.info(f"Created Hugging Face dataset with {len(hf_dataset)} samples.")

        except Exception as e:
            logger.error(f"Error creating Hugging Face dataset: {e}", exc_info=True)
            self.model.eval(); return # Revert to eval mode

        # 2. Tokenize Dataset
        # Define tokenizer function (handles potential missing tokenizer attribute)
        def tokenize_function(examples):
            if not self.tokenizer: return {} # Should not happen if check at start passed
            model_max_len = getattr(self.model.config, 'n_positions', 512) if self.model else 512
            # Ensure tokenizer arguments are correct
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False, # Do not pad here, collator will handle it
                max_length=min(model_max_len - 2, 510) # Leave room, ensure reasonable max
            )

        try:
            num_proc = max(1, os.cpu_count() // 2) if os.cpu_count() else 1 # Safer proc count
            tokenized_dataset = hf_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=["text"] # Remove original text column
            )
            logger.info(f"Tokenized dataset using map ({num_proc} processes).")
        except Exception as e:
            logger.error(f"Error tokenizing dataset map: {e}", exc_info=True)
            self.model.eval(); return

        # 3. Block Processing (Group texts for Causal LM)
        block_size = 128 # Adjust based on model VRAM/context window
        def group_texts_function(examples):
            # Concatenate all texts column by column
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])

            # We drop the small remainder, adjust total_length to be multiple of block_size
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            else:
                # If total length is less than block size, skip this batch for grouping
                return {k: [] for k in examples.keys()} # Return empty lists

            # Split by chunks of block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            # Create labels for Causal Language Modeling (input_ids shifted)
            result["labels"] = result["input_ids"].copy()
            return result

        try:
            lm_datasets = tokenized_dataset.map(
                group_texts_function,
                batched=True,
                num_proc=num_proc # Reuse proc count
            )
            logger.info(f"Processed dataset into blocks of size {block_size}. New size: {len(lm_datasets)}")
            if len(lm_datasets) == 0:
                logger.error("Dataset processing resulted in 0 blocks. Check block_size/data length.")
                self.model.eval(); return
        except Exception as e:
            logger.error(f"Error grouping texts: {e}", exc_info=True)
            self.model.eval(); return

        # 4. Data Collator for Causal LM
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # 5. Training Arguments
        # Ensure gradient_checkpointing is bool
        gc_flag = False # Disabled by default for simplicity, enable if memory is an issue
        training_args = TrainingArguments(
             output_dir=output_dir,
             overwrite_output_dir=True,
             num_train_epochs=epochs,
             per_device_train_batch_size=4,       # Lower if OOM issues occur
             gradient_accumulation_steps=4,       # Increase if batch size is lowered
             save_strategy="epoch",               # Save checkpoint every epoch
             save_total_limit=2,                  # Keep only the last 2 checkpoints
             logging_steps=50,                    # Log training progress every 50 steps
             learning_rate=Config.NLP.GPT_LR,     # Learning rate from config
             weight_decay=0.01,                   # Weight decay for regularization
             warmup_ratio=0.1,                    # Ratio of total steps for learning rate warmup
             fp16=torch.cuda.is_available(),      # Enable mixed precision if CUDA is available
             report_to="none",                    # Disable external reporting (like wandb)
             gradient_checkpointing=gc_flag,      # Enable/disable gradient checkpointing
             logging_dir=os.path.join(output_dir, 'logs'), # Store trainer logs within output dir
             dataloader_num_workers = 0, # Set explicitly to 0 for Windows compatibility often
             # label_names=["labels"] # Usually not needed, inferred by Trainer
         )

        # 6. Initialize Trainer
        trainer = Trainer(
             model=self.model,
             args=training_args,
             train_dataset=lm_datasets,           # Use the processed block dataset
             data_collator=data_collator,
         )

        # 7. Train
        logger.info("Starting Trainer.train()...")
        try:
             train_result = trainer.train()
             logger.info(f"Fine-tuning finished. Metrics: {train_result.metrics}")
             # Save the final model and tokenizer
             trainer.save_model() # Saves to training_args.output_dir
             if self.tokenizer: self.tokenizer.save_pretrained(training_args.output_dir)
             logger.info(f"Fine-tuned model and tokenizer saved to {training_args.output_dir}")
        except Exception as e:
             logger.error(f"Error during Trainer.train(): {e}", exc_info=True)
        finally:
            # Ensure model is back in eval mode regardless of training success/failure
            self.model.eval()

    def generate(self, context: Optional[str], emotions: Optional[torch.Tensor]=None, max_len: int = Config.NLP.MAX_RESPONSE_LEN, temperature: float = Config.NLP.GPT_TEMPERATURE, top_p: float = Config.NLP.GPT_TOP_P, num_beams: int = 1, repetition_penalty: float = 1.15) -> str:
        if not self.model or not self.tokenizer: return "[Gen Error: No Model]"
        self.model.eval(); prompt = context or "";
        if not prompt.strip() and self.tokenizer.bos_token: prompt = self.tokenizer.bos_token + " "
        elif prompt.strip() and not prompt.endswith((" ","\n", ":")): prompt += " "
        try:
             model_max_len=getattr(self.model.config, 'n_positions', 512); max_output_len=max(5, min(max_len, model_max_len//2))
             max_prompt_len=model_max_len-max_output_len-10;
             if max_prompt_len<=0: max_prompt_len=model_max_len//2
             inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len).to(self.device)
             input_len = inputs.input_ids.shape[1]; gen_pad = self.pad_token_id if self.pad_token_id is not None else self.end_token_id
             with torch.no_grad():
                  outputs = self.model.generate(**inputs, max_new_tokens=max_output_len, num_beams=num_beams,
                                                 do_sample=(num_beams==1 and temperature > 0.01), temperature=max(temperature, 0.01) if num_beams==1 else 1.0,
                                                 top_p=top_p if num_beams==1 and 0.0 < top_p < 1.0 else None, repetition_penalty=repetition_penalty,
                                                 pad_token_id=gen_pad, eos_token_id=self.end_token_id)
             response = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        except Exception as e: logger.error(f"HF generate error: {e}", exc_info=True); response = "[Gen Error]"
        response = response.strip(); return response if response else "..."

    def save_model(self, path: str):
         if not self.model or not self.tokenizer: logger.error("Cannot save HF model: Not initialized."); return
         try: logger.info(f"Saving HF model/tokenizer to dir: {path}..."); os.makedirs(path, exist_ok=True); self.model.save_pretrained(path); self.tokenizer.save_pretrained(path); logger.info("HF model/tokenizer saved.")
         except Exception as e: logger.error(f"Failed save HF model/tokenizer to {path}: {e}", exc_info=True)

    def load_model(self, path: str) -> bool:
        if not TRANSFORMERS_AVAILABLE: return False
        load_path = path; fallback_used = False
        if not os.path.isdir(load_path): logger.warning(f"Path '{load_path}' not dir. Trying base model '{self.model_name}'."); load_path = self.model_name; fallback_used = True
        # Prevent loading base name if it unexpectedly became a directory
        if os.path.isdir(load_path) and load_path == self.model_name and not fallback_used: logger.error(f"Base model name path conflict '{load_path}'."); return False
        loaded = self._load_hf_model_and_tokenizer(load_path)
        if not loaded and not fallback_used: # Try base model if specific path failed
            logger.warning(f"Specific path '{path}' failed, trying base model '{self.model_name}' again.")
            loaded = self._load_hf_model_and_tokenizer(self.model_name)
        return loaded

# --- END OF FILE ai_modules.py ---
