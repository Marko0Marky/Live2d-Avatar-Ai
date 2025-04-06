# --- START OF FILE config.py ---

import torch
import logging
import sys
import os
import string
from typing import List, Tuple, Optional, Dict # Added typing
from dataclasses import dataclass, field # Added dataclass

# --- Setup logging ---
# ... (logging setup remains the same) ...
log_file = "vr_avatar_ai5_run.log"
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file, mode='w'),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)
# Set console handler level to INFO
console_handler_found = False
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.INFO)
        console_handler_found = True
if not console_handler_found:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.warning("StreamHandler not found, added one manually.")
# Reduce log spam
logging.getLogger('OpenGL').setLevel(logging.WARNING)
logging.getLogger('PyQt5').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING) # Add PIL if it becomes noisy
logging.getLogger('matplotlib').setLevel(logging.WARNING) # Add matplotlib


# --- Determine Compute Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- Configuration Dataclass ---
@dataclass
class GraphicsConfig:
    MODEL_PATH: str = os.path.abspath("./models/as01.model3.json") # Default path
    FPS: int = 60
    PARTICLE_COUNT: int = 25 # Reduced from previous example, adjust as needed
    PARTICLE_MAX_SIZE: float = 8.0
    PARTICLE_MIN_SIZE: float = 2.0
    GLOW_INTENSITY: float = 3.0
    BACKGROUND_COLOR: Tuple[float, float, float, float] = (0.1, 0.1, 0.15, 1.0)
    EYE_PARAM_DEFAULT: float = 1.0
    MOUTH_PARAM_DEFAULT: float = 0.0

@dataclass
class AgentConfig:
    STATE_DIM: int = 12 # Recalculate if EMOTION_DIM changes: EMOTION_DIM + 6
    HIDDEN_DIM: int = 64
    MEMORY_SIZE: int = 10000
    HISTORY_SIZE: int = 10
    CASCADE_LEVELS: int = 3
    EMOTION_DIM: int = 6
    STABILITY_THRESHOLD: float = 0.85
    ACCESSIBILITY_THRESHOLD: float = 0.8
    ATTENTION_THRESHOLD: float = 0.65
    TAU: float = 0.1 # For MetronicLattice

@dataclass
class RLConfig:
    GAMMA: float = 0.99
    LR: float = 0.0005
    AGENT_TRAIN_INTERVAL: int = 4
    AGENT_BATCH_SIZE: int = 64
    GRADIENT_CLIP_AGENT: float = 1.5
    # --- PER ---
    PER_ALPHA: float = 0.6
    PER_BETA_START: float = 0.4
    PER_BETA_FRAMES: int = 100000
    # --- Intrinsic Rewards ---
    INTRINSIC_REWARD_SCALE_CONSISTENCY: float = 0.05 # Scale for rho_score reward
    INTRINSIC_REWARD_SCALE_BOX: float = 0.02       # Scale for box_score reward
    INTRINSIC_REWARD_SCALE_TD: float = 0.0        # Existing TD-error based intrinsic reward (keep disabled for now)


@dataclass
class NLPConfig:
    GPT_LR: float = 0.0005
    TRAIN_EPOCHS: int = 15
    MAX_RESPONSE_LEN: int = 16
    GRADIENT_CLIP_GPT: float = 1.0
    # --- Tokenizer Settings ---
    TOKENIZER_PATH: str = "./tokenizer/bpe_agent_tokenizer.json" # Where to save/load
    VOCAB_SIZE: int = 1000 # Target vocab size for BPE training
    SPECIAL_TOKENS: List[str] = field(default_factory=lambda: ["<PAD>", "<START>", "<END>", "<UNK>"]) # Standard special tokens


@dataclass
class EnvConfig:
    EVENT_FREQ: float = 0.3
    EVENT_DURATION: int = 120
    EVENT_GAP: int = 40

@dataclass
class Config:
    Agent: AgentConfig = field(default_factory=AgentConfig)
    RL: RLConfig = field(default_factory=RLConfig)
    NLP: NLPConfig = field(default_factory=NLPConfig)
    Env: EnvConfig = field(default_factory=EnvConfig)
    Graphics: GraphicsConfig = field(default_factory=GraphicsConfig)

# --- Instantiate Config ---
# This creates the actual config object used by the application
MasterConfig = Config()

# --- Tokenizer Setup ---
# Import necessary libraries (ensure 'tokenizers' is installed: pip install tokenizers)
try:
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
except ImportError:
    logger.critical("Hugging Face 'tokenizers' library not found. Please install: pip install tokenizers")
    sys.exit(1)

# Global tokenizer instance
tokenizer: Optional[Tokenizer] = None
PAD_TOKEN_ID: int = 0
START_TOKEN_ID: int = 1
END_TOKEN_ID: int = 2
UNK_TOKEN_ID: int = 3

def train_or_load_tokenizer(data: List[Dict[str, str]], config: NLPConfig) -> Tokenizer:
    """Trains a BPE tokenizer from data or loads it if it exists."""
    global PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, UNK_TOKEN_ID, tokenizer

    tokenizer_path = config.TOKENIZER_PATH
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

    if os.path.exists(tokenizer_path):
        logger.info(f"Loading existing tokenizer from {tokenizer_path}")
        loaded_tokenizer = Tokenizer.from_file(tokenizer_path)
        config.VOCAB_SIZE = loaded_tokenizer.get_vocab_size() # Update config vocab size
        logger.info(f"Tokenizer loaded. Vocab size: {config.VOCAB_SIZE}")
    else:
        logger.info(f"Tokenizer not found at {tokenizer_path}. Training a new one...")
        if not data:
            logger.error("Cannot train tokenizer: No training data provided.")
            sys.exit(1)

        # Extract text for training
        text_corpus = [item.get("output", "") for item in data if item.get("output")]
        text_corpus.extend([item.get("situation", "") for item in data if item.get("situation")])
        if not text_corpus:
             logger.error("Cannot train tokenizer: No valid text found in training data.")
             sys.exit(1)

        # Initialize tokenizer
        bpe_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        bpe_tokenizer.pre_tokenizer = Whitespace() # Simple split by whitespace first
        bpe_tokenizer.decoder = decoders.BPEDecoder() # Use BPE decoder

        # Trainer
        trainer = BpeTrainer(vocab_size=config.VOCAB_SIZE, special_tokens=config.SPECIAL_TOKENS)

        # Train
        bpe_tokenizer.train_from_iterator(text_corpus, trainer=trainer)
        config.VOCAB_SIZE = bpe_tokenizer.get_vocab_size() # Update config with actual size
        logger.info(f"Tokenizer training complete. Final Vocab size: {config.VOCAB_SIZE}")

        # Save the tokenizer
        bpe_tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        loaded_tokenizer = bpe_tokenizer

    # Set global tokenizer and special IDs
    tokenizer = loaded_tokenizer
    PAD_TOKEN_ID = tokenizer.token_to_id("<PAD>")
    START_TOKEN_ID = tokenizer.token_to_id("<START>")
    END_TOKEN_ID = tokenizer.token_to_id("<END>")
    UNK_TOKEN_ID = tokenizer.token_to_id("<UNK>")

    # Configure post-processing (add START/END tokens automatically if desired)
    # Optional: Template processing can add special tokens during encoding
    # tokenizer.post_processor = TemplateProcessing(
    #     single="<START> $A <END>",
    #     special_tokens=[("<START>", START_TOKEN_ID), ("<END>", END_TOKEN_ID)],
    # )

    if None in [PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, UNK_TOKEN_ID]:
        logger.critical(f"Failed to get IDs for all special tokens from tokenizer: "
                        f"PAD={PAD_TOKEN_ID}, START={START_TOKEN_ID}, END={END_TOKEN_ID}, UNK={UNK_TOKEN_ID}")
        sys.exit(1)
    logger.debug(f"Special Token IDs: PAD={PAD_TOKEN_ID}, START={START_TOKEN_ID}, END={END_TOKEN_ID}, UNK={UNK_TOKEN_ID}")

    return tokenizer

def tokenize(text: str, max_length: int = MasterConfig.NLP.MAX_RESPONSE_LEN - 2) -> List[int]:
    """Tokenizes text using the global tokenizer."""
    if tokenizer is None:
        logger.error("Tokenizer not initialized. Cannot tokenize.")
        return []
    if not isinstance(text, str):
        logger.warning(f"Invalid input to tokenize: type {type(text)}. Returning empty list.")
        return []
    # Encode the text, add_special_tokens=False because we handle START/END manually usually
    encoding = tokenizer.encode(text.lower(), add_special_tokens=False)
    # Truncate if necessary (keeping space for START/END later)
    truncated_ids = encoding.ids[:max_length]
    return truncated_ids

def detokenize(indices: List[int]) -> str:
    """Detokenizes indices using the global tokenizer."""
    if tokenizer is None:
        logger.error("Tokenizer not initialized. Cannot detokenize.")
        return ""
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().tolist()
    if not isinstance(indices, (list, tuple)):
        logger.warning(f"Invalid input to detokenize: type {type(indices)}. Returning empty string.")
        return ""

    # Filter out padding and potentially other special tokens if needed during decoding
    # Note: The decoder itself might handle some special tokens depending on its configuration.
    valid_indices = [int(idx) for idx in indices if idx != PAD_TOKEN_ID] # Remove padding

    # Decode
    decoded_text = tokenizer.decode(valid_indices, skip_special_tokens=True) # Skip special tokens in output
    return decoded_text.strip()

# --- Initial Training Data (Still used for GPT training and tokenizer training) ---
TRAIN_DATA = [
    {"situation": "A cheerful bird sings nearby", "output": "yay i love this wonderful life", "emotion_weights": [0.9, 0.1, 0.4, 0.1]},
    {"situation": "A scary shadow looms nearby", "output": "i am nervous about that", "emotion_weights": [0.1, 0.9, 0.3, 0.4]},
    {"situation": "Something intriguing catches my eye", "output": "what is that tell me more", "emotion_weights": [0.2, 0.3, 0.9, 0.2]},
    {"situation": "An annoying glitch disrupts everything", "output": "ugh why does this happen", "emotion_weights": [0.1, 0.6, 0.2, 0.8]},
    {"situation": "A gentle breeze flows through", "output": "feeling calm and relaxed", "emotion_weights": [0.6, 0.1, 0.2, 0.1]},
    {"situation": "A sudden flash lights up the space", "output": "whoa what was that", "emotion_weights": [0.3, 0.4, 0.7, 0.1]},
    {"situation": "Thinking about complexity", "output": "things feel complicated", "emotion_weights": [0.2, 0.5, 0.6, 0.6]},
    {"situation": "User interaction (click)", "output": "oh hello what was that", "emotion_weights": [0.4, 0.1, 0.7, 0.1]},
    # Add more diverse data if possible
    {"situation": "User: hello there", "output": "oh hello feeling happy", "emotion_weights": [0.7, 0.1, 0.4, 0.1]},
    {"situation": "User: you look sad", "output": "i am not sure why", "emotion_weights": [0.2, 0.4, 0.3, 0.5]},
    {"situation": "User: tell me a joke", "output": "why was the scarecrow happy", "emotion_weights": [0.6, 0.1, 0.5, 0.1]},
]
for i, item in enumerate(TRAIN_DATA): # Validation
    if not isinstance(item, dict) or "output" not in item or "emotion_weights" not in item: raise ValueError(f"Invalid TRAIN_DATA {i}")
    # Adjust emotion weights validation if EMOTION_DIM changes
    expected_emo_len = 4 # GPT bias layer uses 4
    if len(item["emotion_weights"]) != expected_emo_len:
         logger.warning(f"Train data item {i} has {len(item['emotion_weights'])} emo weights, expected {expected_emo_len}. Padding/truncating.")
         item["emotion_weights"] = (item["emotion_weights"] + [0.0] * expected_emo_len)[:expected_emo_len]


# --- Initialize Tokenizer ---
# This should be called early, potentially in main.py before agent init
# For now, let's call it here. It will train only if the file doesn't exist.
train_or_load_tokenizer(TRAIN_DATA, MasterConfig.NLP)

# --- Validate Config After Tokenizer Init ---
if MasterConfig.Agent.STATE_DIM != MasterConfig.Agent.EMOTION_DIM + 6:
     logger.critical(f"FATAL: Config Agent.STATE_DIM ({MasterConfig.Agent.STATE_DIM}) != EMOTION_DIM ({MasterConfig.Agent.EMOTION_DIM}) + 6. Fix config.")
     sys.exit(1)

# --- END OF FILE config.py ---
