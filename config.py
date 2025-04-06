# --- START OF FILE config.py ---

import torch
import logging
import sys
import os
import string
from collections import namedtuple

# --- Setup logging ---
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

# --- Determine Compute Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- Configuration Class ---
class Config:
    # --- Syntrometrie & Agent Core ---
    SPACE_SIZE = 5
    EMOTION_DIM = 6
    STATE_DIM = EMOTION_DIM + 6 # 12 total
    HIDDEN_DIM = 64
    MEMORY_SIZE = 10000 # Increased memory size for PER effectiveness
    HISTORY_SIZE = 10
    CASCADE_LEVELS = 3
    TAU = 0.1
    RHO_THRESHOLD = 0.9 # Unused?
    STABILITY_THRESHOLD = 0.85
    ACCESSIBILITY_THRESHOLD = 0.8
    ATTENTION_THRESHOLD = 0.65

    # --- Environment & Events ---
    EVENT_FREQ = 0.3
    EVENT_DURATION = 120
    EVENT_GAP = 40

    # --- Reinforcement Learning & Training ---
    GAMMA = 0.99
    LR = 0.0005             # Slightly reduced LR often helps with more complex RL
    GPT_LR = 0.0005
    TRAIN_EPOCHS = 15       # Initial GPT training epochs
    AGENT_TRAIN_INTERVAL = 4 # Train agent every N simulation steps
    AGENT_BATCH_SIZE = 64   # Batch size for agent learning

    # --- Prioritized Experience Replay (PER) ---
    PER_ALPHA = 0.6         # Prioritization exponent (0=uniform, 1=fully prioritized)
    PER_BETA_START = 0.4    # Initial importance sampling exponent
    PER_BETA_FRAMES = 100000 # Steps over which beta anneals to 1.0

    # --- Intrinsic Motivation ---
    INTRINSIC_REWARD_SCALE = 0.0 # Scale factor for TD-error based reward (0 = disabled)

    # --- GPT & Language ---
    VOCAB_SIZE = 0 # Dynamically set below
    MAX_RESPONSE_LEN = 16

    # --- Live2D Avatar & Graphics ---
    # Ensure this path is correct relative to the CWD when main.py runs
    MODEL_PATH = os.path.abspath("./models/as01.model3.json")
    FPS = 60
    PARTICLE_COUNT = 25
    PARTICLE_MAX_SIZE = 8.0
    PARTICLE_MIN_SIZE = 2.0
    GLOW_INTENSITY = 3.0
    BACKGROUND_COLOR = (0.1, 0.1, 0.15, 1.0)
    EYE_PARAM_DEFAULT = 1.0
    MOUTH_PARAM_DEFAULT = 0.0

    # --- Additional Configuration ---
    GRADIENT_CLIP_AGENT = 1.5
    GRADIENT_CLIP_GPT = 1.0

# --- Vocabulary and Tokenizer ---
VOCAB = ["<PAD>", "<START>", "<END>", "i", "am", "so", "happy", "to", "chat", "with", "you", "feeling", "nervous", "about", "this", "what", "is", "that", "tell", "me", "more", "ugh", "why", "does", "everything", "have", "be", "interesting", "how", "works", "yay", "love", "bit", "can", "we", "talk", "calm", "tricky", "complicated", "no", "way", "not", "sure", "do", "feel", "great", "life", "wonderful", "things", "happen", "scary", "confused", "help", "understand", "thinking", "it", "makes", "sense", "oh", "hello", "relaxed", "whoa", "was", "bird", "sings", "shadow", "looms", "intriguing", "object", "glitch", "occurs", "gentle", "breeze", "sudden", "flash", "quiet"]
Config.VOCAB_SIZE = len(VOCAB); logger.info(f"Config.VOCAB_SIZE set to {Config.VOCAB_SIZE}")
if Config.VOCAB_SIZE <= 3: raise ValueError(f"Vocab size too small.")
WORD_TO_IDX = {word: idx for idx, word in enumerate(VOCAB)}; IDX_TO_WORD = {idx: word for idx, word in enumerate(VOCAB)}
required_tokens = ["<PAD>", "<START>", "<END>"];
for token in required_tokens:
    if token not in WORD_TO_IDX: raise ValueError(f"Required token '{token}' missing.")
PAD_TOKEN_ID = WORD_TO_IDX["<PAD>"]; START_TOKEN_ID = WORD_TO_IDX["<START>"]; END_TOKEN_ID = WORD_TO_IDX["<END>"]
def tokenize(text):
    if not isinstance(text, str): return []
    translator = str.maketrans('', '', string.punctuation); clean_text = text.translate(translator)
    tokens = clean_text.lower().split(); valid_tokens = [WORD_TO_IDX.get(token, PAD_TOKEN_ID) for token in tokens]
    return valid_tokens[:max(0, Config.MAX_RESPONSE_LEN - 2)]
def detokenize(indices):
    if isinstance(indices, torch.Tensor): indices = indices.cpu().tolist()
    if not isinstance(indices, (list, tuple)): return ""
    special_ids = {PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID}; valid_indices = []
    for idx in indices: 
        try: int_idx=int(idx); valid_indices.append(int_idx) if int_idx in IDX_TO_WORD else None; 
        except: pass
    return " ".join([IDX_TO_WORD[idx] for idx in valid_indices if idx not in special_ids])

# --- Initial Training Data ---
TRAIN_DATA = [ {"situation": "A cheerful bird sings nearby", "output": "yay i love this wonderful life", "emotion_weights": [0.9, 0.1, 0.4, 0.1]}, {"situation": "A scary shadow looms nearby", "output": "i am nervous about that", "emotion_weights": [0.1, 0.9, 0.3, 0.4]}, {"situation": "Something intriguing catches my eye", "output": "what is that tell me more", "emotion_weights": [0.2, 0.3, 0.9, 0.2]}, {"situation": "An annoying glitch disrupts everything", "output": "ugh why does this happen", "emotion_weights": [0.1, 0.6, 0.2, 0.8]}, {"situation": "A gentle breeze flows through", "output": "feeling calm and relaxed", "emotion_weights": [0.6, 0.1, 0.2, 0.1]}, {"situation": "A sudden flash lights up the space", "output": "whoa what was that", "emotion_weights": [0.3, 0.4, 0.7, 0.1]}, {"situation": "Thinking about complexity", "output": "things feel complicated", "emotion_weights": [0.2, 0.5, 0.6, 0.6]}, {"situation": "User interaction (click)", "output": "oh hello what was that", "emotion_weights": [0.4, 0.1, 0.7, 0.1]},]
for i, item in enumerate(TRAIN_DATA): # Validation
    if not isinstance(item, dict) or "output" not in item or "emotion_weights" not in item: raise ValueError(f"Invalid TRAIN_DATA {i}")
    if len(item["emotion_weights"]) != 4: raise ValueError(f"Invalid emo weights len {i}")

# --- END OF FILE config.py ---