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
logging.getLogger('OpenGL').setLevel(logging.WARNING); logging.getLogger('PyQt5').setLevel(logging.WARNING); logging.getLogger('PIL').setLevel(logging.WARNING); logging.getLogger('matplotlib').setLevel(logging.WARNING); logging.getLogger('tokenizers').setLevel(logging.WARNING); logging.getLogger('sentence_transformers').setLevel(logging.WARNING); logging.getLogger('transformers').setLevel(logging.WARNING) # Added transformers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LANGUAGE_EMBEDDING_DIM = 384
BASE_AGENT_STATE_DIM = 12

# --- Head Movement Labels ---
HEAD_MOVEMENT_LABELS = [
    "idle", "slight_tilt", "small_nod", "gentle_nod", "quick_nod", "slow_tilt",
    "ponder_tilt", "concerned_tilt", "sympathetic_tilt", "curious_turn",
    "quick_turn", "negative_tilt", "confused_tilt", "restless_shift",
]
NUM_HEAD_MOVEMENTS = len(HEAD_MOVEMENT_LABELS)
HEAD_MOVEMENT_TO_IDX = {label: i for i, label in enumerate(HEAD_MOVEMENT_LABELS)}
IDX_TO_HEAD_MOVEMENT = {i: label for i, label in enumerate(HEAD_MOVEMENT_LABELS)}
logger.info(f"Defined {NUM_HEAD_MOVEMENTS} head movement labels: {HEAD_MOVEMENT_LABELS}")

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
    USE_LANGUAGE_EMBEDDING: bool = True
    STATE_DIM: int = field(init=False) # Calculated in __post_init__
    BASE_STATE_DIM: int = BASE_AGENT_STATE_DIM
    LANGUAGE_EMBEDDING_DIM: int = LANGUAGE_EMBEDDING_DIM
    HIDDEN_DIM: int = 128
    MEMORY_SIZE: int = 10000
    HISTORY_SIZE: int = 10
    CASCADE_LEVELS: int = 3
    EMOTION_DIM: int = 6
    STABILITY_THRESHOLD: float = 0.85
    ACCESSIBILITY_THRESHOLD: float = 0.8
    ATTENTION_THRESHOLD: float = 0.65
    TAU: float = 0.1

    def __post_init__(self):
        """Recalculates STATE_DIM based on other parameters."""
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
    HEAD_MOVEMENT_LOSS_WEIGHT: float = 0.1
    # --- Target Network Update Params ---
    TARGET_NETWORK_UPDATE_FREQ: int = 1000 # Steps between hard updates (if using hard update mode in agent.learn)
    TARGET_NETWORK_SOFT_UPDATE_TAU: float = 0.005 # Tau for soft updates (if using soft update mode in agent.learn)


@dataclass
class NLPConfig:
    SENTENCE_TRANSFORMER_MODEL: str = 'all-MiniLM-L6-v2'
    HUGGINGFACE_MODEL: str = "distilgpt2" # Or "gpt2", "microsoft/DialoGPT-small", etc.
    GPT_LR: float = 0.0005 # LR for separate fine-tuning if using TransformerGPT Trainer
    TRAIN_EPOCHS: int = 3 # Epochs for fine-tuning or SimpleGPT pre-training
    MAX_RESPONSE_LEN: int = 40 # Max length for generated responses
    GRADIENT_CLIP_GPT: float = 1.0 # Grad clip for GPT training/fine-tuning
    TOKENIZER_PATH: str = "./tokenizer/bpe_agent_tokenizer.json" # Path for custom BPE tokenizer (potentially unused)
    VOCAB_SIZE: int = 1000 # Target vocab size for BPE (potentially unused)
    SPECIAL_TOKENS: List[str] = field(default_factory=lambda: ["<PAD>", "<START>", "<END>", "<UNK>"]) # For BPE tokenizer
    GPT_TEMPERATURE: float = 0.8
    GPT_TOP_P: float = 0.9
    CONVERSATION_HISTORY_LENGTH: int = 4 # Number of User+AI turns to keep

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

MasterConfig = Config() # Instantiate the main config object

# --- Paths for Saving/Loading ---
# Ensure the saved_models directory exists
os.makedirs("./saved_models", exist_ok=True)
AGENT_SAVE_PATH = os.path.abspath("./saved_models/agent_state.pth")
GPT_SAVE_PATH = os.path.abspath(f"./saved_models/{MasterConfig.NLP.HUGGINGFACE_MODEL.replace('/','_')}_finetuned") # Directory for HF models based on name
OPTIMIZER_SAVE_PATH = os.path.abspath("./saved_models/optimizer_state.pth")
TARGET_NET_SAVE_SUFFIX = "_target" # Suffix for target network file
REPLAY_BUFFER_SAVE_PATH = os.path.abspath("./saved_models/replay_buffer.pkl")

# --- Training Data Path & Loading ---
TRAINING_DATA_PATH = "./train_data.json" # Use the original name by default
# Global variable to hold validated data
TRAIN_DATA: List[Dict[str, Any]] = []
def load_and_validate_train_data(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        # Try loading the anime girl data if the default doesn't exist
        anime_path = "./train_data_anime_girl.json"
        logger.warning(f"Training data file not found: {path}. Trying '{anime_path}'...")
        if not os.path.exists(anime_path):
            logger.error(f"Alternative training data '{anime_path}' also not found. No data loaded.")
            return []
        path = anime_path # Switch path to the anime data

    try:
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        if not isinstance(data, list): logger.error(f"Training data file {path} does not contain a valid JSON list."); return []
        logger.info(f"Successfully loaded {len(data)} training examples from {path}")

        validated_data = []
        logger.info("Validating and adjusting training data during load...")
        items_with_hm = 0
        unknown_labels = set() # Track unknown labels found
        for i, item in enumerate(data):
            # Basic format check
            if not isinstance(item, dict) or "output" not in item:
                logger.warning(f"Invalid item format in TRAIN_DATA at index {i} from {path}. Skipping.")
                continue

            # Validate/Adjust emotion_weights (Keep for potential SimpleGPT fallback or other uses)
            expected_emo_len = 4
            current_weights = item.get("emotion_weights", [])
            if not isinstance(current_weights, list):
                logger.warning(f"Invalid emotion_weights type ({type(current_weights)}) in data at index {i}. Using zeros.")
                item["emotion_weights"] = [0.0] * expected_emo_len
            elif len(current_weights) != expected_emo_len:
                item["emotion_weights"] = (current_weights + [0.0] * expected_emo_len)[:expected_emo_len]

            # Validate/Process head_movement labels
            hm_label = item.get("head_movement")
            if hm_label is not None:
                items_with_hm += 1
                if hm_label not in HEAD_MOVEMENT_TO_IDX:
                    if hm_label not in unknown_labels:
                         logger.warning(f"Item {i} in {path} has unknown head_movement label: '{hm_label}'. Setting to 'idle'. Valid: {HEAD_MOVEMENT_LABELS}")
                         unknown_labels.add(hm_label)
                    item["head_movement"] = "idle"
                item["head_movement_idx"] = HEAD_MOVEMENT_TO_IDX.get(item["head_movement"], HEAD_MOVEMENT_TO_IDX["idle"])
            else:
                item["head_movement"] = "idle"
                item["head_movement_idx"] = HEAD_MOVEMENT_TO_IDX["idle"]

            validated_data.append(item)

        logger.info(f"Validated training data: {len(validated_data)} items from {path}. Found head_movement labels in {items_with_hm} items.")
        if unknown_labels: logger.warning(f"Unknown head movement labels encountered: {list(unknown_labels)}")
        return validated_data
    except json.JSONDecodeError as e: logger.error(f"Error decoding JSON from {path}: {e}"); return []
    except FileNotFoundError: logger.error(f"Training data file not found during validation: {path}"); return []
    except TypeError as e: logger.error(f"Type error during data validation: {e}", exc_info=True); return []
    except Exception as e: logger.error(f"Unexpected error loading/validating training data from {path}: {e}", exc_info=True); return []


# --- Load training data ---
TRAIN_DATA = load_and_validate_train_data(TRAINING_DATA_PATH) # Load and validate data


# --- Final Config Validation ---
# Check after __post_init__ has run
MasterConfig.Agent.__post_init__() # Ensure post_init runs after potential changes
expected_state_dim = MasterConfig.Agent.BASE_STATE_DIM + (MasterConfig.Agent.LANGUAGE_EMBEDDING_DIM if MasterConfig.Agent.USE_LANGUAGE_EMBEDDING else 0)
if MasterConfig.Agent.STATE_DIM != expected_state_dim:
     logger.critical(f"FATAL: Calculated Agent.STATE_DIM ({MasterConfig.Agent.STATE_DIM}) mismatch with expected ({expected_state_dim}). Check AgentConfig USE_LANGUAGE_EMBEDDING and AgentConfig.__post_init__.")
     sys.exit(1)
if MasterConfig.Agent.EMOTION_DIM > MasterConfig.Agent.BASE_STATE_DIM:
    logger.critical(f"FATAL: Config Agent.EMOTION_DIM ({MasterConfig.Agent.EMOTION_DIM}) > BASE_STATE_DIM ({MasterConfig.Agent.BASE_STATE_DIM}).")
    sys.exit(1)

# --- END OF FILE config.py ---
