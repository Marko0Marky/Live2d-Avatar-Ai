---

# Live2D Avatar AI Agent  
### Powered by the Syntrometrie Framework

Imagine a **Live2D Cubism 3** avatar that doesn’t just move—it *feels*. This project fuses real-time animation with a "conscious" AI agent, driven by the **Syntrometrie framework**. Chat with it, and watch it respond with dynamic **expressions** and **head movements**, all shaped by reinforcement learning. Built on **PyTorch**, **PyQt5/OpenGL**, and tools like `live2d-py`, `sentence-transformers`, and `tokenizers`.

---

## What’s This About?

This is where AI consciousness meets animation. The goal? A virtual companion that’s emotionally aware and endlessly interactive. It’s a sandbox for AI tinkerers, animators, and developers to explore reinforcement learning, NLP, and Live2D magic.

---

## Highlights

- **Syntrometrie Framework**:
  - *Metronic Lattice*: Maps states into discrete chunks.
  - *Syntrix Korporator*: Crafts the agent’s beliefs.
  - *Struktur Kaskade*: Chains reasoning steps.
- **Dynamic Responses**: Merges emotional states with optional text embeddings.
- **NLP Powerhouse**:
  - Lightweight **GPT-style Transformer** for chat.
  - **Sentence Transformers** for rich embeddings.
  - **BPE Tokenizer** for snappy text processing.
- **Avatar Life**:
  - Procedural animations (breathing, blinking, sway).
  - Emotion-driven expressions and predictive head tilts.
- **Learning Engine**:
  - **Prioritized Experience Replay (PER)** for smart RL.
  - Asynchronous training for silky performance.
- **GUI Goodies**: PyQt5 interface with HUD, state panel, chat, and particle flair.

---

## Project Pulse

- **Status**: In active development.
- **Next Up**: Stronger RL, more movement variety, and performance boosts.

---

## How It Comes Alive

1. **State Kickoff**: `EmotionalSpace` builds a base state from chats or events.
2. **Text Magic**: Optional `SentenceTransformer` embeddings enrich the vibe.
3. **AI Brain**: `ConsciousAgent` picks actions (like a nod) based on beliefs.
4. **Avatar Flow**: `Live2DCharacter` renders emotions and motions.
5. **Learning Cycle**: RL sharpens behavior with PER, running async.

Dive deeper in [Core Components](#core-components).

---

## Get It Running

### What You’ll Need
- **Python**: 3.8 or higher.
- **Dependencies**:
  ```bash
  pip install torch numpy PyQt5 PyOpenGL PyOpenGL-accelerate qasync live2d-py[cubism3] tokenizers sentence-transformers
  ```
- **Live2D Core**: Snag the native library (`.dll`, `.so`, `.dylib`) from [Live2D](https://www.live2d.com/en/) and drop it in the project root or system path. Check [live2d-py](https://github.com/GreatFruitOmsk/live2d-py) for setup tips.

### Setup Steps
1. **Add a Model**:
   - Place your Cubism 3 model (e.g., `model.model3.json`) in `./models/`.
   - Update `GraphicsConfig.MODEL_PATH` in `config.py`.
2. **Training Data**:
   - Include `train_data.json` (see format below).
   - Adjust `TRAINING_DATA_PATH` in `config.py` if it’s elsewhere.

### Launch It
```bash
python main.py
```
- **Controls**:  
  - `Space`: Pause or play.  
  - `Q`/`Esc`: Quit.  
  - `C`: Check completeness.  
  - Chat through the GUI.

---

## Core Pieces

| File             | What It Does                                                    |
|------------------|-----------------------------------------------------------------|
| `config.py`      | Master settings for agent, RL, NLP, graphics, and env.         |
| `agent.py`       | `ConsciousAgent`: Handles RL and predicts movements.           |
| `environment.py` | `EmotionalSpace`: Simulates the emotional baseline.            |
| `ai_modules.py`  | PyTorch bits like `EmotionalModule` and `SimpleGPT`.           |
| `graphics.py`    | `Live2DCharacter`: Brings animations to life.                  |
| `gui_widgets.py` | HUD and state panel widgets for the interface.                 |
| `main_gui.py`    | Main window with the update loop.                              |
| `orchestrator.py`| Ties agent, env, and avatar together.                          |
| `utils.py`       | Helpers, `Experience` class, and PER memory logic.             |
| `main.py`        | Entry point with `asyncio` glue.                               |

---

## Training Data Example

Fuel for the tokenizer, `SimpleGPT`, and movement training:

```json
{
  "situation": "User asks 'how are you?'",
  "output": "I’m feeling awesome, thanks!",
  "emotion_weights": [0.7, 0.2, 0.0, 0.0],
  "head_movement": "gentle_nod"
}
```

---

## Make It Yours

Edit `config.py` to tweak:
- `AgentConfig.USE_LANGUAGE_EMBEDDING`: Toggle embeddings on/off.
- `RLConfig.HEAD_MOVEMENT_LOSS_WEIGHT`: Fine-tune movement learning.
- `GraphicsConfig.MODEL_PATH`: Swap in your model.

---
