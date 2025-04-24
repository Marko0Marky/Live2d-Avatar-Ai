# --- START OF FILE config.py ---

import torch
import logging
import sys
import os
import string
import re
import json
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field, InitVar

# --- Logging Setup & DEVICE ---
log_file = "vr_avatar_ai5_rrdt.log" # New log file name
log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
log_dir = os.path.dirname(log_file)
if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
# Configure root logger only if it hasn't been configured yet
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format=log_format, handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ])
logger = logging.getLogger(__name__) # Use module-specific logger
# Configure console handler level if needed
console_handler = next((h for h in logging.getLogger().handlers if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout), None)
if console_handler:
    console_handler.setLevel(logging.INFO)
    logger.debug("Console handler level set to INFO.")
else: # Add handler if none exists for stdout
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format); ch.setFormatter(formatter)
    logging.getLogger().addHandler(ch); logger.warning("StreamHandler added manually to root logger.")

# Silence overly verbose libraries
logging.getLogger('OpenGL').setLevel(logging.WARNING); logging.getLogger('PyQt5').setLevel(logging.WARNING);
logging.getLogger('PIL').setLevel(logging.WARNING); logging.getLogger('matplotlib').setLevel(logging.WARNING);
logging.getLogger('tokenizers').setLevel(logging.WARNING); logging.getLogger('sentence_transformers').setLevel(logging.WARNING);
logging.getLogger('transformers').setLevel(logging.WARNING); logging.getLogger('h5py').setLevel(logging.WARNING);
logging.getLogger('numexpr').setLevel(logging.WARNING); logging.getLogger('torch_geometric').setLevel(logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
try: logger.info(f"PyTorch version {torch.__version__} available.")
except Exception: logger.warning("Could not determine PyTorch version.")

# --- Head Movement Labels (Kept for graphics, but agent doesn't learn them) ---
HEAD_MOVEMENT_LABELS = [
    "idle", "slight_tilt", "small_nod", "gentle_nod", "quick_nod", "slow_tilt",
    "ponder_tilt", "concerned_tilt", "sympathetic_tilt", "curious_turn",
    "quick_turn", "negative_tilt", "confused_tilt", "restless_shift",
]
NUM_HEAD_MOVEMENTS = len(HEAD_MOVEMENT_LABELS)
# logger.info(f"Defined {NUM_HEAD_MOVEMENTS} head movement labels.") # Optional

# --- Syntrometric/RIH/HET Inspired Configuration ---

@dataclass
class GNNConfig:
    GNN_TYPE: str = 'GAT' # Options: 'GCN', 'GAT', 'GraphSAGE', 'GIN', 'Linear' (fallback)
    GNN_LAYERS: int = 3
    GNN_HIDDEN_DIM: int = 64 # Should match AgentConfig.HIDDEN_DIM for this architecture
    GAT_HEADS: int = 4 # Specific to GATConv
    GNN_USE_RESIDUAL: bool = True # Use residual connections in GNN
    GRAPH_IS_DYNAMIC: bool = False # Feature flag for future dynamic graphs

@dataclass
class AgentConfig:
    STATE_DIM: int = 12 # Fixed 12D state
    HIDDEN_DIM: int = 64 # General hidden dim (MUST MATCH GNN_HIDDEN_DIM here)
    EMOTION_DIM: int = 6 # First 6 dims of state
    QUALIA_DIM: int = 6 # Last 6 dims of state (STATE_DIM - EMOTION_DIM)
    PHYSICAL_DIMS: Tuple[int, ...] = field(default_factory=lambda: tuple(range(6)))
    INFO_SEMANTIC_DIMS: Tuple[int, ...] = field(default_factory=lambda: tuple(range(6, 12)))
    DIM_WEIGHTS: Optional[List[float]] = None # For weighted integration

    MEMORY_SIZE: int = 20000
    HISTORY_SIZE: int = 8 # Agent's internal state history length

    TAU: float = 1e-5 # Metronic Lattice discretization factor
    ATTENTION_THRESHOLD: float = 0.55 # Threshold for GPT response style/memory gating

    GNN: GNNConfig = field(default_factory=GNNConfig) # Embed GNN config

    def __post_init__(self):
        if self.GNN.GNN_HIDDEN_DIM != self.HIDDEN_DIM:
             logger.critical(f"CRITICAL CONFIG: Agent HIDDEN_DIM ({self.HIDDEN_DIM}) != GNN_HIDDEN_DIM ({self.GNN.GNN_HIDDEN_DIM}). MUST MATCH.")
             self.GNN.GNN_HIDDEN_DIM = self.HIDDEN_DIM # Force match
        if self.QUALIA_DIM != (self.STATE_DIM - self.EMOTION_DIM):
             logger.error(f"Config Dimension mismatch: Qualia({self.QUALIA_DIM}) + Emotion({self.EMOTION_DIM}) != State({self.STATE_DIM})")

@dataclass
class RLConfig:
    GAMMA: float = 0.99
    LR: float = 0.0002 # Learning Rate
    AGENT_TRAIN_INTERVAL: int = 4
    AGENT_BATCH_SIZE: int = 64
    GRADIENT_CLIP_AGENT: float = 1.0
    PER_ALPHA: float = 0.6 # Prioritization exponent
    PER_BETA_START: float = 0.4 # Initial importance sampling exponent
    PER_BETA_FRAMES: int = 150000 # Steps over which beta anneals to 1.0
    TARGET_NETWORK_UPDATE_FREQ: int = 1 # Steps between soft updates (if tau < 1.0)
    TARGET_NETWORK_SOFT_UPDATE_TAU: float = 0.005 # Tau for soft Polyak averaging
    MOOD_UPDATE_DECAY: float = 0.995 # Keep mood mechanism

    # RIH/HET Parameters
    RHO_SIMILARITY_THRESHOLD: float = 0.90 # Target for reflexivity score
    TAU_0_BASE: float = 0.8               # Base integration threshold
    ALPHA_TAU_DYNAMICS: float = 0.08      # Weight for entropy in tau(t) calculation
    BETA_TELEZENTRIK: float = 0.03       # Weight for g_ik stability in tau(t)
    DYNAMICS_THRESHOLD_WEIGHT: float = 0.10 # Weight for belief change norm in tau(t)

    # Loss Function Weights (REBALANCED - NEED TUNING)
    VALUE_LOSS_WEIGHT: float = 0.6            # TD Value Loss (Smooth L1)
    INTEGRATION_WEIGHT: float = 0.3           # RIH integration penalty (I < tau)
    REFLEXIVITY_WEIGHT: float = 0.2           # RIH reflexivity penalty (rho < threshold)
    STABILITY_WEIGHT: float = 0.05          # g_ik instability penalty
    DYNAMICAL_STABILITY_WEIGHT: float = 0.1   # Lyapunov proxy penalty (lambda_max > 0)
    GEOMETRIC_COHERENCE_WEIGHT: float = 0.01  # Penalty for deviation from target metric g (Placeholder)
    TELEZENTRIK_WEIGHT: float = 0.02          # Telezentrik reward (negated in loss)
    CONSISTENCY_WEIGHT: float = 0.01          # Self-consistency reward (negated in loss)
    COMPLEXITY_WEIGHT: float = 0.02           # Zeta penalty
    CURVATURE_WEIGHT: float = 0.001         # R penalty

    MEMORY_GATING_ENABLED: bool = True # Can re-enable gating
    MEMORY_GATE_ATTENTION_THRESHOLD: float = 0.45 # Gating threshold

@dataclass
class NLPConfig:
    HUGGINGFACE_MODEL: str = "distilgpt2"
    GPT_LR: float = 5e-5 # Fine-tuning LR
    TRAIN_EPOCHS: int = 1 # Fine-tuning epochs
    MAX_RESPONSE_LEN: int = 35 # Max new tokens for GPT response
    GRADIENT_CLIP_GPT: float = 1.0
    GPT_TEMPERATURE: float = 0.7 # Generation temperature
    GPT_TOP_P: float = 0.9
    CONVERSATION_HISTORY_LENGTH: int = 6 # Longer history

@dataclass
class EnvConfig:
    EVENT_FREQ: float = 0.05 # Less frequent random events
    EVENT_DURATION: int = 100 # Shorter duration
    EVENT_GAP: int = 80 # Longer gap
    QUALIA_FEEDBACK_STRENGTH: float = 0.05 # How much qualia feedback affects next emotions

@dataclass
class GraphicsConfig:
    MODEL_PATH: str = os.path.abspath("./models/as01.model3.json")
    FPS: int = 60
    PARTICLE_COUNT: int = 30
    PARTICLE_MAX_SIZE: float = 7.0
    PARTICLE_MIN_SIZE: float = 1.5
    GLOW_INTENSITY: float = 2.5
    BACKGROUND_COLOR: Tuple[float, float, float, float] = (0.1, 0.1, 0.15, 1.0)
    EYE_PARAM_DEFAULT: float = 1.0
    MOUTH_PARAM_DEFAULT: float = 0.0

@dataclass
class Config:
    Agent: AgentConfig = field(default_factory=AgentConfig)
    RL: RLConfig = field(default_factory=RLConfig)
    NLP: NLPConfig = field(default_factory=NLPConfig)
    Env: EnvConfig = field(default_factory=EnvConfig)
    Graphics: GraphicsConfig = field(default_factory=GraphicsConfig)
    DT: float = field(init=False)
    LOG_FILE: str = field(init=False)

    def __post_init__(self):
        self.DT = 1.0 / self.Graphics.FPS if self.Graphics.FPS > 0 else (1.0 / 60.0)
        self.LOG_FILE = os.path.abspath(log_file)
        # Call AgentConfig's post_init if it exists
        if hasattr(self.Agent, '__post_init__'): self.Agent.__post_init__()

MasterConfig = Config()

logger.info(f"Calculated DT: {MasterConfig.DT:.6f} (FPS: {MasterConfig.Graphics.FPS})")
logger.info(f"Logging to: {MasterConfig.LOG_FILE}")
logger.info(f"Configuration loaded for RRDT/PyG Agent (GNN: {MasterConfig.Agent.GNN.GNN_TYPE}).")

# --- Paths for Saving/Loading ---
save_dir = "./saved_models_rrdt"
os.makedirs(save_dir, exist_ok=True)
AGENT_SAVE_PATH = os.path.abspath(os.path.join(save_dir, "rrdt_agent_state.pth"))
GPT_SAVE_PATH = os.path.abspath(os.path.join(save_dir, f"{MasterConfig.NLP.HUGGINGFACE_MODEL.replace('/','_')}_finetuned"))
OPTIMIZER_SAVE_PATH = os.path.abspath(os.path.join(save_dir, "rrdt_optimizer_state.pth"))
TARGET_NET_SAVE_SUFFIX = "_target" # Used for saving target network components
REPLAY_BUFFER_SAVE_PATH = os.path.abspath(os.path.join(save_dir, "rrdt_replay_buffer.pkl"))

# --- Training Data Path (Optional) ---
TRAINING_DATA_PATH = "./train_data.json"
TRAIN_DATA: List[Dict[str, Any]] = []
def load_and_validate_train_data(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): logger.warning(f"Training data file not found: {path}."); return []
    try:
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        if not isinstance(data, list): logger.error(f"Training data file {path} invalid format."); return []
        logger.info(f"Loaded {len(data)} potential training examples from {path}")
        validated_data = [item for item in data if isinstance(item, dict) and "situation" in item and "output" in item]
        logger.info(f"Validated training data contains {len(validated_data)} entries.")
        return validated_data
    except Exception as e: logger.error(f"Error loading/validating training data from {path}: {e}"); return []
# TRAIN_DATA = load_and_validate_train_data(TRAINING_DATA_PATH)

# --- END OF FILE config.py ---
