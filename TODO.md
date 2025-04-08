# Master TODO List & Improvement Roadmap

**🎯 Overall Goal:**  
Create a more robust, performant, and engaging AI-driven Live2D avatar with richer conversational abilities and more natural non-verbal communication.

---

## 🚧 Phase 1: Core Functionality & Stability  
*(Critical / High Priority)*  
*These are essential for making the current system usable and reliable enough for further development.*

### 🤖 NLP/GPT - Fine-tuning *(CRITICAL)*  
- **Acquire/Prepare More Training Data:**  
  - Generate/find/filter significantly more data (target: 1000+ examples) specific to the desired persona and conversational style.  
  - Ensure correct formatting: `<START>User: ... AI: ...<END>`.  
- **Run Fine-tuning Script:**  
  - Execute `fine_tune_gpt.py` on a GPU using the prepared dataset.  
  - Monitor loss and results.  
- **Verify Model Loading:**  
  - Ensure the fine-tuned model is correctly saved to and loaded from `GPT_SAVE_PATH` when running `main.py`.  
- **Evaluate Conversational Quality:**  
  - Test chat extensively after fine-tuning.

### 🔧 Code Quality - Testing Framework  
- **Setup pytest:**  
  - Initialize a `tests/` directory.  
- **Write Unit Tests (Core Utils):**  
  - Create tests for `utils.py` (e.g., `is_safe`, `MetronicLattice.discretize`, `Experience` structure).  
- **Write Unit Tests (AI Modules):**  
  - Create basic tests for `ai_modules.py` (check input/output shapes, basic functionality of `EmotionalModule`, `Korporator`, `Kaskade`, `TransformerGPT` init/load).

### 🔧 Code Quality - Dependency Management  
- **Generate `requirements.txt`:**  
  - Run `pip freeze > requirements.txt` (ideally from a clean virtual environment) and commit it.

### 🐛 Bug Fixes & Refinements  
- **HTML Entity Handling:**  
  - Verify the fix in `orchestrator.py` correctly prevents `'`, `&`, etc., from appearing in chat history and display.  
  - Ensure escaping/unescaping logic is sound.  
- **Save/Load Robustness (Buffer):**  
  - Consider replacing `pickle` for saving/loading the replay buffer (`MetaCognitiveMemory`) with `torch.save`/`np.savez` for better long-term compatibility. *(Medium priority compared to fine-tuning)*

---

## 🚀 Phase 2: Improving Agent Learning & Behavior  
*(High / Medium Priority)*  
*Focus on making the agent learn better and act more intelligently once conversations are coherent.*

### 🤖 AI/Agent - Head Movement Prediction  
- **Analyze Belief State:**  
  - Add logging/debugging to see if features input to `head_movement_head` are sufficiently varied.  
- **Tune Loss Weight:**  
  - Experiment with increasing `RLConfig.HEAD_MOVEMENT_LOSS_WEIGHT` in `config.py`.  
- **Review Training Labels:**  
  - Ensure `head_movement` labels in `train_data.json` are accurate and diverse.  
- **(Optional) Improve Network:**  
  - If linear layer underperforms, try a small MLP for `head_movement_head` in `agent.py`.

### 🤖 AI/Agent - Intrinsic Motivation  
- **Implement RND (or ICM):**  
  - Define Target and Predictor networks in `ai_modules.py`.  
  - Integrate into `agent.py` (`__init__`, `learn`).  
  - Add scaling factor(s) to `RLConfig`.  
  - Tune scaling factor(s).

### 🤖 AI/Agent - Advanced RL Algorithms  
- **Explore Dueling DQN:**  
  - Modify `value_head` in `agent.py` to separate value and advantage streams.  
- **Explore Rainbow:**  
  - Integrate multiple DQN improvements (DDQN, Dueling, PER, Noisy Nets, Distributional RL) - complex.  
- **Explore Policy Gradients (PPO/SAC):**  
  - Significant refactor needed if moving away from value-based methods (would likely require redefining discrete/continuous action spaces).

### 🤖 AI/Agent - Richer Emotions/Mood  
- **Non-linear Interactions:**  
  - Modify `EmotionalModule` so emotions influence each other (e.g., high fear reduces joy).  
- **Belief/State Influence:**  
  - Allow metrics like consistency (`rho_score`) or specific belief patterns to directly affect emotion calculation.  
- **Distinct Mood:**  
  - Implement a slower-changing `long_term_mood` vector that biases the faster `current_emotions`.

### 🤖 NLP/GPT - Advanced Context Management  
- **Summarization:**  
  - Implement history summarization for longer conversations.  
- **Memory Injection:**  
  - Experiment with adding structured summaries of recent experiences or belief states into the GPT prompt.

### 🤖 NLP/GPT - Decoding Strategy  
- **Beam Search:**  
  - Implement or enable beam search (`num_beams > 1`) in `TransformerGPT.generate` and compare results.  
- **Other Strategies:**  
  - Explore diverse beam search, top-k sampling, etc.

---

## ✨ Phase 3: Enhancing Interaction & Performance  
*(Medium / Low Priority)*  
*Focus on making the interaction more immersive and optimizing performance.*

### 🎨 Non-Verbal - Lip Sync  
- **Choose Phonemizer:**  
  - Select and install a library (`phonemizer`, `espeak-ng` binding).  
- **Generate Visemes:**  
  - Modify `orchestrator.py` to generate visemes from GPT output.  
- **Create Mapping:**  
  - Define viseme -> mouth parameter map in `graphics.py`.  
- **Implement Timing:**  
  - Synchronize mouth parameter changes with text display/audio duration in `graphics.py`.

### 🎨 Non-Verbal - Eye Gaze  
- **Implement Logic:**  
  - Add logic in `graphics.py` to control `PARAM_EYE_BALL_X/Y` based on:  
    - Cursor position (partially done).  
    - Conversation state (e.g., looking away when thinking).  
    - Random saccades.

### 🎨 Non-Verbal - More Parameter Control  
- **Identify Parameters:**  
  - Analyze your specific Live2D model for controllable parameters (arms, body rotation, accessories).  
- **Map Parameters:**  
  - Add mappings in `graphics.py` linked to emotions, predicted actions, or specific chat triggers.

### ⚡ Performance - Profiling  
- **Profile `update_game` Loop:**  
  - Use `cProfile` or `py-spy`.  
- **Profile PyTorch Code:**  
  - Use `torch.profiler`.  
- **Profile Rendering:**  
  - Use GPU profiling tools if necessary.

### ⚡ Performance - Optimization  
- **Apply Findings:**  
  - Address bottlenecks identified during profiling (e.g., JIT compilation, AMP, shader optimization, ST batching).

### 🗣️ Features - Voice Interaction  
- **STT Integration:**  
  - Add library (e.g., `SpeechRecognition`, `faster-whisper`) to capture user voice input.  
- **TTS Integration:**  
  - Add library (e.g., `pyttsx3`, `gTTS`, cloud APIs, `Coqui TTS`) to speak the agent's response.  
  - Synchronize with lip sync.

### 📊 Features - GUI Enhancements  
- **Add Plots:**  
  - Use `matplotlib` or `pyqtgraph` to display reward/loss history in the GUI.  
- **Config UI:**  
  - Allow loading/saving different `config.py` settings via the GUI.  
- **Visual Feedback:**  
  - Add visual cues for internal agent states or actions.

---

## 🧹 Phase 4: Code Quality & Long-Term Maintenance  
*(Ongoing)*  
*These should be addressed throughout the development process.*

### 🔧 Code Quality - Testing  
- **Expand Unit Tests:**  
  - Cover more functions and edge cases in `utils.py`, `ai_modules.py`, `agent.py`, `environment.py`.  
- **Implement Integration Tests:**  
  - Test interactions between `Orchestrator` and other components.

### 🔧 Code Quality - Error Handling  
- **Specific Exceptions:**  
  - Replace generic `except Exception`.  
- **Graceful Fallbacks:**  
  - Define behavior when key components fail.  
- **Boundary Validation:**  
  - Add more `assert` or checks for tensor shapes/types between modules.

### 🔧 Code Quality - Documentation  
- **Add Docstrings:**  
  - Document all classes and non-trivial functions (purpose, args, returns).  
- **Add Comments:**  
  - Explain complex or non-obvious code sections.  
- **Update README:**  
  - Keep the README accurate as features are added/changed.

### 🔧 Code Quality - Refactoring  
- **Review `agent.py`/`orchestrator.py`:**  
  - Identify logic that could be moved to helper classes/functions for better organization.  
- **Clean up BPE Code:**  
  - If the BPE tokenizer is confirmed unused, remove related code from `utils.py` and `config.py`.

### 🔧 Code Quality - Configuration  
- **Consider Hydra/YAML:**  
  - Evaluate if configuration complexity warrants moving away from dataclasses in `config.py`.

---

### Final Notes  
