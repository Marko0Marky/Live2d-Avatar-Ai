# --- START OF FILE README.md ---

# Live2D Conscious Agent with Language-Augmented State

This project implements a conversational AI agent based on the Syntrometrie framework, visualized through a real-time Live2D avatar. A key feature is the augmentation of the agent's internal state vector with language embeddings derived from its internal monologue and user interactions, aiming for more contextually aware and nuanced behavior.

**Overview:**

The system simulates an emotional environment that influences a conscious agent. The agent processes its state (a combination of base sensory/emotional data and language embeddings), forms beliefs, generates internal monologues, responds to user chat, and learns through reinforcement learning. Its internal state (emotions, mood) and generated text/monologue drive the animations and expressions of a Live2D avatar using `live2d-py` and PyQt5/OpenGL.

**Key Features (After Language Embedding Integration):**

*   **Syntrometrie AI Core:** Implements core concepts like Metronic Lattice, Syntrix Korporator, and Struktur Kaskade for state processing and belief formation.
*   **Emotional Module & Mood:** Simulates rapid emotional reactions and a slower-changing underlying mood based on state and rewards.
*   **Language-Augmented State:**
    *   Utilizes `sentence-transformers` to generate semantic embeddings from the agent's internal monologue and user chat input.
    *   Combines the environment's *base state* (emotions, meta-features) with the latest language embedding to form the agent's full *combined state* vector (`STATE_DIM`).
    *   Agent components (Lattice, Korporator, Attention, etc.) process this combined state.
*   **Live2D Avatar Integration:**
    *   Renders a Live2D Cubism 3 model using `live2d-py`.
    *   Includes procedural animations (breathing, blinking, idle sway).
    *   Maps agent emotions to avatar parameters.
    *   **Monologue-Driven Head Movement:** Analyzes the agent's internal monologue using `TextBlob` (sentiment/keywords) to trigger corresponding head tilt/nod animations.
    *   **Micro-movements:** Adds subtle, procedural movements for increased idle realism.
*   **Natural Language Processing:**
    *   Includes a simple GPT-like model (trained on provided data) for generating chat responses and internal monologues.
    *   Uses a BPE Tokenizer trained on the provided dataset.
    *   Maintains a short conversation history for contextual responses.
*   **Reinforcement Learning:**
    *   Employs Prioritized Experience Replay (PER) via `MetaCognitiveMemory`.
    *   Includes basic intrinsic rewards (consistency, exploration heuristic).
    *   Supports adaptive learning rates.
*   **PyQt5 GUI:** Provides real-time visualization of the avatar, an overlay HUD (emotions, metrics), agent state panel, environment controls, and a chat interface.
*   **Asynchronous Learning:** Agent learning (`agent.learn()`) runs in a separate thread pool to avoid blocking the GUI.

**Core Components:**

*   `config.py`: Central configuration (Agent, RL, NLP, Graphics, Env), logging, **state dimension calculation**, training data validation, tokenizer loading/training.
*   `agent.py`: Defines the `ConsciousAgent` class, integrating AI modules, memory, and learning logic. Processes the **combined state vector**.
*   `environment.py`: Simulates the `EmotionalSpace`, providing events and the **base state vector** (emotions + meta-features).
*   `ai_modules.py`: Contains `EmotionalModule`, `SyntrixKorporator` (updated for combined state), `StrukturKaskade`, and `SimpleGPT`.
*   `orchestrator.py`: The `EnhancedConsciousAgent` class managing the simulation loop, loading the Sentence Transformer model, **generating embeddings**, **combining base state and embeddings**, handling chat, triggering learning, and interfacing with the GUI/avatar.
*   `graphics.py`: Implements the `Live2DCharacter` widget, handling rendering, animations, parameter mapping, particle effects, and the **monologue analysis/micro-movement implementation**.
*   `gui_widgets.py`: Defines `HUDWidget` and `AIStateWidget` for displaying agent info.
*   `main_gui.py`: The main `QMainWindow`, integrating all UI parts and managing the main update timer. Passes monologue to graphics.
*   `utils.py`: Helper functions (`is_safe`), `Experience` tuple, `MetaCognitiveMemory` (stores combined states on CPU), `MetronicLattice` (operates on combined state dim).
*   `main.py`: Application entry point, asyncio/QApplication setup, orchestrator/GUI initialization, main event loop execution.

**How it Works (State Flow):**

1.  **Environment (`environment.py`)**: Generates a `base_state` (e.g., 12 dimensions: 6 emotions + 6 meta) based on internal events.
2.  **Orchestrator (`orchestrator.py`)**:
    *   Receives the `base_state`.
    *   Retrieves the `last_text_embedding` (generated from the previous step's monologue or user chat).
    *   Calls `_get_combined_state` to concatenate `base_state` and `last_text_embedding` into the `current_state` (e.g., 12 + 384 = 396 dimensions).
3.  **Agent (`agent.py`)**:
    *   Receives the full `current_state` (combined).
    *   Processes it through its internal modules (Lattice, Korporator, Attention, etc.) operating on the combined dimension.
    *   Outputs `belief`, `value`, internal `emotions`, etc.
4.  **Orchestrator (`orchestrator.py`)**:
    *   Uses the agent's output to generate an `internal_monologue`.
    *   *Embeds* this monologue using the Sentence Transformer, updating `last_text_embedding` for the *next* cycle.
    *   Stores the `state_before_step_combined` and `next_combined_state` (created using the new embedding) in the `Experience` replay buffer.
5.  **Graphics (`graphics.py`)**: Receives internal `monologue` from orchestrator (via GUI), analyzes it (TextBlob), and applies corresponding head movements. Also receives `emotions` to drive expressions.
6.  **Loop**: The process repeats. User chat input also generates an embedding that updates `last_text_embedding`.

**Setup & Installation:**

1.  **Clone Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Install Python Dependencies:**
    ```bash
    pip install torch numpy PyQt5 PyOpenGL PyOpenGL-accelerate qasync live2d-py[cubism3] tokenizers sentence-transformers textblob
    ```
    *   **Important:** `live2d-py` requires the native Live2D Cubism Core library. Please follow the specific installation instructions on the [live2d-py repository](https://github.com/Live2D/live2d-cubism-python) for your operating system *before* running the pip install command.
    *   **Important:** After installing `textblob`, download its required corpora:
        ```bash
        python -m textblob.download_corpora
        ```
3.  **Place Live2D Model:**
    *   Create a `models` directory (or use an existing one).
    *   Place your Live2D Cubism 3 model files (e.g., `your_model.model3.json`, textures folder, `.moc3` file, etc.) inside.
    *   Update `GraphicsConfig.MODEL_PATH` in `config.py` to point to your `.model3.json` file (default is `./models/as01.model3.json`).
4.  **Prepare Training Data:**
    *   Place your training data file (e.g., `train_data.json`) in the project's root directory.
    *   Update `TRAINING_DATA_PATH` in `config.py` if your file has a different name or location.
    *   The expected format is a JSON list of dictionaries:
        ```json
        [
          {
            "output": "This is the desired AI response text.",
            "emotion_weights": [0.8, 0.1, 0.1, 0.0], // 4 floats for GPT bias
            "situation": "Optional situation description.",
            "head_movement": "gentle_nod" // Optional: See HEAD_MOVEMENT_LABELS in config.py
          },
          // ... more entries
        ]
        ```
        *   `emotion_weights` should ideally be 4 floats. The loader will attempt to pad/truncate if needed.
        *   `head_movement` is optional; if present, it should be one of the labels defined in `config.py`.

**Configuration:**

*   Modify parameters within the dataclasses in `config.py` to tune agent behavior, learning rates, model paths, graphics settings, etc.
*   Key parameters:
    *   `AgentConfig.USE_LANGUAGE_EMBEDDING`: `True` or `False`.
    *   `NLPConfig.SENTENCE_TRANSFORMER_MODEL`: Name of the Hugging Face Sentence Transformer model to use.
    *   `AgentConfig.LANGUAGE_EMBEDDING_DIM`: **Must match** the output dimension of the chosen `SENTENCE_TRANSFORMER_MODEL`.
    *   `GraphicsConfig.MODEL_PATH`: Path to your Live2D model JSON.
    *   `TRAINING_DATA_PATH`: Path to your GPT training data.

**Running the Project:**

```bash
python main.py
