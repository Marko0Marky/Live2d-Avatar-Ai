# --- START OF FILE config.py ---

import torch
import logging
import sys
import os
import string
from typing import List, Tuple, Dict, Optional # Added Optional
from dataclasses import dataclass, field # Added dataclass

# --- Setup logging ---
log_file = "vr_avatar_ai5_run.log"
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format, handlers=[
    logging.FileHandler(log_file, mode='w'),
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger(__name__)
console_handler = None
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        console_handler = handler
        break
if console_handler:
    console_handler.setLevel(logging.INFO)
    logger.debug("Console handler found and level set to INFO.")
else:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    ch.setFormatter(formatter)
    logging.root.addHandler(ch)
    logger.warning("StreamHandler not found by basicConfig, added one manually.")
logging.getLogger('OpenGL').setLevel(logging.WARNING)
logging.getLogger('PyQt5').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tokenizers').setLevel(logging.WARNING)


# --- Determine Compute Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- Configuration Dataclasses ---
@dataclass
class GraphicsConfig:
    MODEL_PATH: str = os.path.abspath("./models/as01.model3.json")
    FPS: int = 60
    PARTICLE_COUNT: int = 25
    PARTICLE_MAX_SIZE: float = 8.0
    PARTICLE_MIN_SIZE: float = 2.0
    GLOW_INTENSITY: float = 3.0
    BACKGROUND_COLOR: Tuple[float, float, float, float] = (0.1, 0.1, 0.15, 1.0)
    EYE_PARAM_DEFAULT: float = 1.0
    MOUTH_PARAM_DEFAULT: float = 0.0

@dataclass
class AgentConfig:
    STATE_DIM: int = 12
    HIDDEN_DIM: int = 64
    MEMORY_SIZE: int = 10000
    HISTORY_SIZE: int = 10
    CASCADE_LEVELS: int = 3
    EMOTION_DIM: int = 6
    STABILITY_THRESHOLD: float = 0.85
    ACCESSIBILITY_THRESHOLD: float = 0.8
    ATTENTION_THRESHOLD: float = 0.65
    TAU: float = 0.1

@dataclass
class RLConfig:
    GAMMA: float = 0.99
    LR: float = 0.0005
    AGENT_TRAIN_INTERVAL: int = 4
    AGENT_BATCH_SIZE: int = 64
    GRADIENT_CLIP_AGENT: float = 1.5
    PER_ALPHA: float = 0.6
    PER_BETA_START: float = 0.4
    PER_BETA_FRAMES: int = 100000
    INTRINSIC_REWARD_SCALE_CONSISTENCY: float = 0.05
    INTRINSIC_REWARD_SCALE_BOX: float = 0.02
    INTRINSIC_REWARD_SCALE_TD: float = 0.0
    ADAPTIVE_LR_ENABLED: bool = True
    LR_ADAPTIVE_MIN_FACTOR: float = 0.2
    LR_ADAPTIVE_MAX_FACTOR: float = 1.0
    PRIORITY_ATTENTION_WEIGHT: float = 0.5
    # --- Mood Component --- # <<< SECTION ADDED FOR CLARITY
    MOOD_UPDATE_DECAY: float = 0.995 # <<< THE MISSING LINE


@dataclass
class NLPConfig:
    GPT_LR: float = 0.0005
    TRAIN_EPOCHS: int = 15
    MAX_RESPONSE_LEN: int = 16
    GRADIENT_CLIP_GPT: float = 1.0
    TOKENIZER_PATH: str = "./tokenizer/bpe_agent_tokenizer.json"
    VOCAB_SIZE: int = 1000 # Target size, will be updated after load/train
    SPECIAL_TOKENS: List[str] = field(default_factory=lambda: ["<PAD>", "<START>", "<END>", "<UNK>"])
    GPT_TEMPERATURE: float = 0.7 # Default sampling temperature
    GPT_TOP_P: float = 0.9       # Default nucleus sampling p


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
MasterConfig = Config()

# --- Tokenizer Setup ---
try:
    from tokenizers import Tokenizer, decoders, AddedToken
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from typing import Optional # Import Optional here for the global variable type hint
except ImportError:
    logger.critical("Hugging Face 'tokenizers' library not found. Please install: pip install tokenizers")
    sys.exit(1)

tokenizer: Optional[Tokenizer] = None
PAD_TOKEN_ID: Optional[int] = None
START_TOKEN_ID: Optional[int] = None
END_TOKEN_ID: Optional[int] = None
UNK_TOKEN_ID: Optional[int] = None

def train_or_load_tokenizer(data: List[Dict[str, str]], config: NLPConfig) -> Tokenizer:
    global PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, UNK_TOKEN_ID, tokenizer
    tokenizer_path = config.TOKENIZER_PATH
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    special_token_objects = [AddedToken(token, single_word=True) for token in config.SPECIAL_TOKENS]

    if os.path.exists(tokenizer_path):
        logger.info(f"Loading existing tokenizer from {tokenizer_path}")
        loaded_tokenizer = Tokenizer.from_file(tokenizer_path)
        config.VOCAB_SIZE = loaded_tokenizer.get_vocab_size()
        logger.info(f"Tokenizer loaded. Vocab size: {config.VOCAB_SIZE}")
    else:
        logger.info(f"Tokenizer not found at {tokenizer_path}. Training a new one...")
        if not data: logger.error("Cannot train tokenizer: No training data provided."); sys.exit(1)
        text_corpus = [item.get("output", "") for item in data if item.get("output")]
        text_corpus.extend([item.get("situation", "") for item in data if item.get("situation")])
        if not text_corpus: logger.error("Cannot train tokenizer: No valid text found in training data."); sys.exit(1)

        bpe_tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        bpe_tokenizer.pre_tokenizer = Whitespace()
        bpe_tokenizer.decoder = decoders.BPEDecoder()
        trainer = BpeTrainer(vocab_size=config.VOCAB_SIZE, special_tokens=special_token_objects)

        logger.info(f"Training tokenizer with vocab size {config.VOCAB_SIZE} and {len(special_token_objects)} special tokens...")
        bpe_tokenizer.train_from_iterator(text_corpus, trainer=trainer)
        config.VOCAB_SIZE = bpe_tokenizer.get_vocab_size()
        logger.info(f"Tokenizer training complete. Final Vocab size: {config.VOCAB_SIZE}")

        bpe_tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        loaded_tokenizer = bpe_tokenizer

    tokenizer = loaded_tokenizer
    PAD_TOKEN_ID = tokenizer.token_to_id("<PAD>")
    START_TOKEN_ID = tokenizer.token_to_id("<START>")
    END_TOKEN_ID = tokenizer.token_to_id("<END>")
    UNK_TOKEN_ID = tokenizer.token_to_id("<UNK>")

    if None in [PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, UNK_TOKEN_ID]:
        logger.critical(f"Failed to get IDs for all special tokens from tokenizer: "
                        f"PAD={PAD_TOKEN_ID}, START={START_TOKEN_ID}, END={END_TOKEN_ID}, UNK={UNK_TOKEN_ID}")
        sys.exit(1)
    logger.debug(f"Special Token IDs: PAD={PAD_TOKEN_ID}, START={START_TOKEN_ID}, END={END_TOKEN_ID}, UNK={UNK_TOKEN_ID}")

    return tokenizer

def tokenize(text: str, max_length: int = MasterConfig.NLP.MAX_RESPONSE_LEN - 2) -> List[int]:
    if tokenizer is None: logger.error("Tokenizer not initialized. Cannot tokenize."); return []
    if not isinstance(text, str): logger.warning(f"Invalid input to tokenize: type {type(text)}. Returning empty list."); return []
    encoding = tokenizer.encode(text.lower(), add_special_tokens=False)
    truncated_ids = encoding.ids[:max_length]
    return truncated_ids

def detokenize(indices: List[int]) -> str:
    if tokenizer is None: logger.error("Tokenizer not initialized. Cannot detokenize."); return ""
    if isinstance(indices, torch.Tensor): indices = indices.cpu().tolist()
    if not isinstance(indices, (list, tuple)): logger.warning(f"Invalid input to detokenize: type {type(indices)}. Returning empty string."); return ""
    if PAD_TOKEN_ID is None: logger.error("Cannot detokenize: PAD_TOKEN_ID not set."); return ""
    valid_indices = [int(idx) for idx in indices if idx != PAD_TOKEN_ID]
    decoded_text = tokenizer.decode(valid_indices, skip_special_tokens=True)
    return decoded_text.strip()

# --- Initial Training Data ---
TRAIN_DATA = [
    {"situation": "A cheerful bird sings nearby", "output": "yay i love this wonderful life", "emotion_weights": [0.9, 0.1, 0.4, 0.1]},
    {"situation": "A scary shadow looms nearby", "output": "i am nervous about that", "emotion_weights": [0.1, 0.9, 0.3, 0.4]},
    {"situation": "Something intriguing catches my eye", "output": "what is that tell me more", "emotion_weights": [0.2, 0.3, 0.9, 0.2]},
    {"situation": "An annoying glitch disrupts everything", "output": "ugh why does this happen", "emotion_weights": [0.1, 0.6, 0.2, 0.8]},
    {"situation": "A gentle breeze flows through", "output": "feeling calm and relaxed", "emotion_weights": [0.6, 0.1, 0.2, 0.1]},
    {"situation": "A sudden flash lights up the space", "output": "whoa what was that", "emotion_weights": [0.3, 0.4, 0.7, 0.1]},
    {"situation": "Thinking about complexity", "output": "things feel complicated", "emotion_weights": [0.2, 0.5, 0.6, 0.6]},
    {"situation": "User interaction (click)", "output": "oh hello what was that", "emotion_weights": [0.4, 0.1, 0.7, 0.1]},
    {"situation": "User: hello there", "output": "oh hello feeling happy", "emotion_weights": [0.7, 0.1, 0.4, 0.1]},
    {"situation": "User: you look sad", "output": "i am not sure why", "emotion_weights": [0.2, 0.4, 0.3, 0.5]},
    {"situation": "User: tell me a joke", "output": "why was the scarecrow happy", "emotion_weights": [0.6, 0.1, 0.5, 0.1]},
]
for i, item in enumerate(TRAIN_DATA):
    if not isinstance(item, dict) or "output" not in item or "emotion_weights" not in item: raise ValueError(f"Invalid TRAIN_DATA {i}")
    expected_emo_len = 4
    current_weights = item["emotion_weights"]
    if len(current_weights) != expected_emo_len:
         item["emotion_weights"] = (current_weights + [0.0] * expected_emo_len)[:expected_emo_len]

# --- Initialize Tokenizer ---
try:
    train_or_load_tokenizer(TRAIN_DATA, MasterConfig.NLP)
except Exception as e:
    logger.critical(f"CRITICAL: Tokenizer initialization failed during config setup: {e}", exc_info=True)
    sys.exit(1)

# --- Config Validation ---
if MasterConfig.Agent.STATE_DIM != MasterConfig.Agent.EMOTION_DIM + 6:
     logger.critical(f"FATAL: Config Agent.STATE_DIM ({MasterConfig.Agent.STATE_DIM}) != EMOTION_DIM ({MasterConfig.Agent.EMOTION_DIM}) + 6. Fix config.")
     sys.exit(1)

# --- END OF FILE config.py ---
