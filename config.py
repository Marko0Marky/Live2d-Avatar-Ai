# --- START OF FILE config.py ---

# --- START OF FILE config.py ---

import torch
import logging
import sys
import os
import string
import re
import json
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

# --- Logging Setup & DEVICE ---
log_file = "vr_avatar_ai5_run.log"
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format, handlers=[
    logging.FileHandler(log_file, mode='w'),
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger(__name__)
console_handler = None
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler): console_handler = handler; break
if console_handler: console_handler.setLevel(logging.INFO); logger.debug("Console handler set to INFO.")
else: ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); formatter = logging.Formatter(log_format); ch.setFormatter(formatter); logging.root.addHandler(ch); logger.warning("StreamHandler added manually.")
logging.getLogger('OpenGL').setLevel(logging.WARNING); logging.getLogger('PyQt5').setLevel(logging.WARNING); logging.getLogger('PIL').setLevel(logging.WARNING); logging.getLogger('matplotlib').setLevel(logging.WARNING); logging.getLogger('tokenizers').setLevel(logging.WARNING); logging.getLogger('sentence_transformers').setLevel(logging.WARNING) # Add sentence_transformers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration Dataclasses ---
# --- NEW: Define Embedding Dimension ---
# NOTE: This MUST match the output dimension of the chosen sentence transformer model
# 'all-MiniLM-L6-v2' outputs 384 dimensions.
# If you change the model in NLPConfig, change this value too!
LANGUAGE_EMBEDDING_DIM = 384
BASE_AGENT_STATE_DIM = 12 # Original state dim (EMOTION_DIM + 6 meta)

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
    # --- NEW ---
    USE_LANGUAGE_EMBEDDING: bool = True # Enable/disable language embedding in state

    # --- UPDATED STATE_DIM ---
    # Combines base state + language embedding (if enabled)
    STATE_DIM: int = field(init=False) # Calculate after __post_init__
    BASE_STATE_DIM: int = BASE_AGENT_STATE_DIM # Store base dim for convenience if needed
    LANGUAGE_EMBEDDING_DIM: int = LANGUAGE_EMBEDDING_DIM # Store embedding dim

    HIDDEN_DIM: int = 128 # Note: Increased hidden dim due to larger state
    MEMORY_SIZE: int = 10000
    HISTORY_SIZE: int = 10 # History will now contain the *combined* state
    CASCADE_LEVELS: int = 3
    EMOTION_DIM: int = 6
    STABILITY_THRESHOLD: float = 0.85
    ACCESSIBILITY_THRESHOLD: float = 0.8
    ATTENTION_THRESHOLD: float = 0.65
    TAU: float = 0.1

    def __post_init__(self):
        self.STATE_DIM = self.BASE_STATE_DIM + (self.LANGUAGE_EMBEDDING_DIM if self.USE_LANGUAGE_EMBEDDING else 0)
        logger.info(f"AgentConfig Calculated STATE_DIM: {self.STATE_DIM} (Base: {self.BASE_STATE_DIM}, LangEmbed: {self.LANGUAGE_EMBEDDING_DIM if self.USE_LANGUAGE_EMBEDDING else 0}, Enabled: {self.USE_LANGUAGE_EMBEDDING})")


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
    MOOD_UPDATE_DECAY: float = 0.995
    MEMORY_GATING_ENABLED: bool = True
    MEMORY_GATE_ATTENTION_THRESHOLD: float = 0.3

@dataclass
class NLPConfig:
    # --- NEW ---
    SENTENCE_TRANSFORMER_MODEL: str = 'all-MiniLM-L6-v2' # Model to use for embeddings
    GPT_LR: float = 0.0005
    TRAIN_EPOCHS: int = 15 # Reduced epochs for faster startup if needed
    MAX_RESPONSE_LEN: int = 32
    GRADIENT_CLIP_GPT: float = 1.0
    TOKENIZER_PATH: str = "./tokenizer/bpe_agent_tokenizer.json"
    VOCAB_SIZE: int = 1000 # Target size, will be updated after load/train
    SPECIAL_TOKENS: List[str] = field(default_factory=lambda: ["<PAD>", "<START>", "<END>", "<UNK>"])
    GPT_TEMPERATURE: float = 0.7
    GPT_TOP_P: float = 0.9
    CONVERSATION_HISTORY_LENGTH: int = 4

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

MasterConfig = Config()

# --- Training Data Path & Loading ---
TRAINING_DATA_PATH = "./train_data.json"
# Global variable to hold validated data for Agent __init__
TRAIN_DATA: List[Dict[str, Any]] = []
def load_and_validate_train_data(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): logger.error(f"Training data file not found: {path}"); return []
    try:
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        if not isinstance(data, list): logger.error(f"Training data file {path} does not contain a valid JSON list."); return []
        logger.info(f"Successfully loaded {len(data)} training examples from {path}")

        validated_data = []
        logger.info("Validating and adjusting training data during load...")
        for i, item in enumerate(data):
            if not isinstance(item, dict) or "output" not in item:
                logger.warning(f"Invalid item format in TRAIN_DATA at index {i}. Skipping.")
                continue

            expected_emo_len = 4 # For GPT bias layer
            current_weights = item.get("emotion_weights", []) # Default to empty list if missing

            if not isinstance(current_weights, list):
                logger.warning(f"Invalid emotion_weights type ({type(current_weights)}) in data at index {i}. Using zeros.")
                item["emotion_weights"] = [0.0] * expected_emo_len
            elif len(current_weights) != expected_emo_len:
                logger.debug(f"Adjusting emotion_weights length for item {i} from {len(current_weights)} to {expected_emo_len}.")
                item["emotion_weights"] = (current_weights + [0.0] * expected_emo_len)[:expected_emo_len]

            validated_data.append(item) # Add validated item

        logger.info(f"Validated training data: {len(validated_data)} items.")
        return validated_data
    except json.JSONDecodeError as e: logger.error(f"Error decoding JSON from {path}: {e}"); return []
    except Exception as e: logger.error(f"Error loading training data from {path}: {e}"); return []

# --- Tokenizer Setup ---
try:
    from tokenizers import Tokenizer, decoders, AddedToken
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from typing import Optional as TokenizerOptional # Rename to avoid conflict
except ImportError:
    logger.critical("Hugging Face 'tokenizers' library not found. Please install: pip install tokenizers")
    sys.exit(1)
tokenizer: TokenizerOptional[Tokenizer] = None
PAD_TOKEN_ID: TokenizerOptional[int] = None
START_TOKEN_ID: TokenizerOptional[int] = None
END_TOKEN_ID: TokenizerOptional[int] = None
UNK_TOKEN_ID: TokenizerOptional[int] = None
def train_or_load_tokenizer(data: List[Dict[str, str]], config: NLPConfig) -> Tokenizer:
    global PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, UNK_TOKEN_ID, tokenizer
    tokenizer_path = config.TOKENIZER_PATH
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    special_token_objects = [AddedToken(token, single_word=True) for token in config.SPECIAL_TOKENS]
    if os.path.exists(tokenizer_path):
        logger.info(f"Loading existing tokenizer from {tokenizer_path}")
        loaded_tokenizer = Tokenizer.from_file(tokenizer_path)
        if loaded_tokenizer.decoder is None: logger.warning("Loaded tokenizer missing decoder, setting BPEDecoder."); loaded_tokenizer.decoder = decoders.BPEDecoder()
        config.VOCAB_SIZE = loaded_tokenizer.get_vocab_size()
        logger.info(f"Tokenizer loaded. Vocab size: {config.VOCAB_SIZE}")
    else:
        logger.info(f"Tokenizer not found at {tokenizer_path}. Training a new one...")
        if not data: logger.warning("Cannot train tokenizer: No training data provided to train_or_load_tokenizer."); loaded_tokenizer = Tokenizer(BPE(unk_token="<UNK>")); config.VOCAB_SIZE=len(config.SPECIAL_TOKENS);
        else:
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
        logger.critical(f"Failed to get IDs for all special tokens: PAD={PAD_TOKEN_ID}, START={START_TOKEN_ID}, END={END_TOKEN_ID}, UNK={UNK_TOKEN_ID}"); sys.exit(1)
    logger.debug(f"Special Token IDs: PAD={PAD_TOKEN_ID}, START={START_TOKEN_ID}, END={END_TOKEN_ID}, UNK={UNK_TOKEN_ID}")
    return tokenizer

def tokenize(text: str, max_length: int = MasterConfig.NLP.MAX_RESPONSE_LEN - 2) -> List[int]:
    if tokenizer is None: logger.error("Tokenizer not initialized."); return []
    if not isinstance(text, str): logger.warning(f"Invalid input to tokenize: type {type(text)}."); return []
    encoding = tokenizer.encode(text.lower(), add_special_tokens=False)
    truncated_ids = encoding.ids[:max_length]
    return truncated_ids

def detokenize(indices: List[int]) -> str:
    if tokenizer is None: logger.error("Tokenizer not initialized."); return ""
    if isinstance(indices, torch.Tensor): indices = indices.cpu().tolist()
    if not isinstance(indices, (list, tuple)): logger.warning(f"Invalid input to detokenize: type {type(indices)}."); return ""
    if PAD_TOKEN_ID is None or START_TOKEN_ID is None or END_TOKEN_ID is None: logger.error("Cannot detokenize: Special tokens not set."); return ""
    special_ids_to_filter = {PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID}
    valid_indices = [int(idx) for idx in indices if idx not in special_ids_to_filter]
    decoded_text = tokenizer.decode(valid_indices, skip_special_tokens=True)
    processed_text = re.sub(r'(?<=[a-zA-Z0-9])([.,!?;:])', r' \1', decoded_text)
    processed_text = re.sub(r'\s+', ' ', processed_text)
    return processed_text.strip()

# --- Load data and initialize tokenizer ---
TRAIN_DATA = load_and_validate_train_data(TRAINING_DATA_PATH) # Load and validate data
try:
    train_or_load_tokenizer(TRAIN_DATA, MasterConfig.NLP)
except Exception as e:
    logger.critical(f"CRITICAL: Tokenizer initialization failed during config setup: {e}", exc_info=True)
    sys.exit(1)

# --- Final Config Validation (After post_init calculation) ---
expected_state_dim = MasterConfig.Agent.BASE_STATE_DIM + (MasterConfig.Agent.LANGUAGE_EMBEDDING_DIM if MasterConfig.Agent.USE_LANGUAGE_EMBEDDING else 0)
if MasterConfig.Agent.STATE_DIM != expected_state_dim:
     logger.critical(f"FATAL: Calculated Agent.STATE_DIM ({MasterConfig.Agent.STATE_DIM}) mismatch with expected ({expected_state_dim}). Fix config or __post_init__.")
     sys.exit(1)
if MasterConfig.Agent.EMOTION_DIM > MasterConfig.Agent.BASE_STATE_DIM:
    logger.critical(f"FATAL: Config Agent.EMOTION_DIM ({MasterConfig.Agent.EMOTION_DIM}) cannot be larger than BASE_STATE_DIM ({MasterConfig.Agent.BASE_STATE_DIM}).")
    sys.exit(1)


# --- END OF FILE config.py ---
