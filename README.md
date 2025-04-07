# --- START OF FILE README.md ---

# Live2D Conscious Agent with Language-Augmented State and Predicted Gestures

This project implements a conversational AI agent based on the Syntrometrie framework, visualized through a real-time Live2D avatar. The agent integrates language understanding via sentence embeddings into its state and learns to interact with a simulated environment and user chat. A key feature is the agent's direct prediction of discrete head movements based on its internal state.

**Overview:**

The system simulates an emotional environment influencing a conscious agent. The agent processes its state (a combination of base sensory/emotional data and language embeddings), forms beliefs using Syntrometrie components, generates internal monologues and chat responses, predicts appropriate head gestures, and learns through reinforcement learning. Its internal state (emotions, mood) and predicted gestures drive the animations and expressions of a Live2D avatar using `live2d-py` and PyQt5/OpenGL.

**Key Features:**

*   **Syntrometrie AI Core:** Implements core concepts like Metronic Lattice, Syntrix Korporator, and Struktur Kaskade for state processing and belief formation.
*   **Emotional Module & Mood:** Simulates rapid emotional reactions and a slower-changing underlying mood based on state and rewards.
*   **Language-Augmented State:**
    *   Utilizes `sentence-transformers` to generate semantic embeddings from the agent's internal monologue and user chat input.
    *   Combines the environment's *base state* (emotions, meta-features) with the latest language embedding to form the agent's full *combined state* vector (`STATE_DIM`).
    *   Agent components (Lattice, Korporator, Attention, etc.) process this combined state.
*   **Agent-Predicted Head Movement:**
    *   The agent model includes a dedicated output head (`head_movement_head`) that predicts a discrete head movement label (e.g., "small_nod", "ponder_tilt") based on its internal `belief` state.
    *   The list of possible movements is defined in `config.py` (`HEAD_MOVEMENT_LABELS`).
    *   **Note:** Currently, this prediction head is *not* explicitly trained with supervised loss (as `RLConfig.HEAD_MOVEMENT_LOSS_WEIGHT` is 0.0). The predicted movements arise from the initialized weights and correlations learned implicitly through the value function training. Future work could involve training this head using labels from `train_data.json`.
*   **Live2D Avatar Integration:**
    *   Renders a Live2D Cubism 3 model using `live2d-py`.
    *   Includes procedural animations (breathing, blinking, idle sway).
    *   Maps agent emotions to avatar parameters.
    *   Applies the specific head movement animation corresponding to the label predicted by the agent.
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

*   `config.py`: Central configuration (Agent, RL, NLP, Graphics, Env), logging, **state dimension calculation**, **head movement labels**, training data validation (incl. head labels), tokenizer loading/training.
*   `agent.py`: Defines the `ConsciousAgent` class, integrating AI modules, memory, and learning logic. Processes the **combined state vector** and includes the **`head_movement_head` for prediction**.
*   `environment.py`: Simulates the `EmotionalSpace`, providing events and the **base state vector** (emotions + meta-features).
*   `ai_modules.py`: Contains `EmotionalModule`, `SyntrixKorporator` (updated for combined state), `StrukturKaskade`, and `SimpleGPT`.
*   `orchestrator.py`: The `EnhancedConsciousAgent` class managing the simulation loop, loading the Sentence Transformer model, **generating embeddings**, **combining states**, handling chat, triggering learning, **receiving the predicted head movement label from the agent**, and interfacing with the GUI/avatar.
*   `graphics.py`: Implements the `Live2DCharacter` widget, handling rendering, animations, parameter mapping, particle effects. **Receives the predicted head movement label** and applies the corresponding animation via `_apply_head_movement`. Implements micro-movements. (TextBlob analysis removed).
*   `gui_widgets.py`: Defines `HUDWidget` and `AIStateWidget` for displaying agent info.
*   `main_gui.py`: The main `QMainWindow`, integrating all UI parts and managing the main update timer. **Passes the predicted head movement label** to the graphics widget.
*   `utils.py`: Helper functions (`is_safe`), `Experience` tuple, `MetaCognitiveMemory` (stores combined states on CPU), `MetronicLattice` (operates on combined state dim).
*   `main.py`: Application entry point, asyncio/QApplication setup, orchestrator/GUI initialization, main event loop execution.

**How it Works (State & Movement Flow):**

1.  **Environment (`environment.py`)**: Generates a `base_state` (e.g., 12 dimensions).
2.  **Orchestrator (`orchestrator.py`)**:
    *   Receives `base_state`.
    *   Retrieves `last_text_embedding`.
    *   Combines them into `current_state` (e.g., 396 dimensions).
3.  **Agent (`agent.py`)**:
    *   Receives `current_state`.
    *   Processes it through internal modules (`forward` pass).
    *   Outputs `belief`, `value`, internal `emotions`, and `head_movement_logits`.
    *   `step` method determines the `predicted_hm_label` from logits.
4.  **Orchestrator (`orchestrator.py`)**:
    *   Receives agent outputs including `predicted_hm_label`.
    *   Generates an `internal_monologue`.
    *   Embeds the monologue, updating `last_text_embedding` for the *next* cycle.
    *   Stores combined states in the replay buffer.
    *   Returns results including `predicted_hm_label` to the GUI.
5.  **GUI (`main_gui.py`)**: Receives `predicted_hm_label` from the orchestrator.
6.  **Graphics (`graphics.py`)**: Receives `predicted_hm_label` via `update_predicted_movement` method and applies the corresponding animation adjustments in `_apply_head_movement`. Also receives `emotions` for expression mapping.
7.  **Loop**: The process repeats. User chat input also generates an embedding that updates `last_text_embedding` and triggers a chat-specific head movement prediction.

**Setup & Installation:**

1.  **Clone Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Install Python Dependencies:**
    ```bash
    pip install torch numpy PyQt5 PyOpenGL PyOpenGL-accelerate qasync live2d-py[cubism3] tokenizers sentence-transformers
    ```
    *   **Important:** `live2d-py` requires the native Live2D Cubism Core library. Follow instructions on the [live2d-py repository](https://github.com/Live2D/live2d-cubism-python).
    *   **Note:** `textblob` is no longer required for head movement.
3.  **Place Live2D Model:**
    *   Ensure your Live2D Cubism 3 model files are correctly placed (default: `./models/`).
    *   Update `GraphicsConfig.MODEL_PATH` in `config.py` if necessary.
4.  **Prepare Training Data:**
    *   Place your training data file (e.g., `train_data.json`) in the project root.
    *   Update `TRAINING_DATA_PATH` in `config.py` if needed.
    *   Format: JSON list of dictionaries:
        ```json
        [
          {
            "output": "AI response text.",
            "emotion_weights": [0.8, 0.1, 0.1, 0.0], // 4 floats for GPT bias
            "situation": "Optional context.",
            "head_movement": "gentle_nod" // Optional: Must match a label in config.HEAD_MOVEMENT_LABELS
          },
          // ... more entries
        ]
        ```
        *   The loader validates `emotion_weights` (padding/truncating to 4) and `head_movement` labels (defaulting to "idle" if missing/invalid). An integer index (`head_movement_idx`) is added for potential future supervised training.

**Configuration:**

*   Modify parameters within the dataclasses in `config.py`.
*   Key parameters:
    *   `AgentConfig.USE_LANGUAGE_EMBEDDING`: `True` or `False`.
    *   `NLPConfig.SENTENCE_TRANSFORMER_MODEL`: Sentence Transformer model name.
    *   `AgentConfig.LANGUAGE_EMBEDDING_DIM`: Must match the chosen ST model output dim.
    *   `GraphicsConfig.MODEL_PATH`: Path to Live2D model JSON.
    *   `TRAINING_DATA_PATH`: Path to training data.
    *   `RLConfig.HEAD_MOVEMENT_LOSS_WEIGHT`: Set > 0 to enable supervised training of the head movement predictor (requires modifications to memory/learning).

**Running the Project:**

```bash
python main.py
