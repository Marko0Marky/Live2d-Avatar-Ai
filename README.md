---

# Live2D Avatar AI Agent  
### Powered by the Syntrometrie Framework

Meet a **Live2D Cubism 3** avatar that doesn’t just animate—it *thinks* and *feels*. This project blends real-time animation with a "conscious" AI agent, driven by the innovative **Syntrometrie framework**. Chat with it, and watch it respond with dynamic **expressions** and **head movements**, all refined through reinforcement learning. Built on a stack of **PyTorch**, **PyQt5/OpenGL**, `live2d-py`, `sentence-transformers`, and `tokenizers`.

---

## 🌟 What’s It All About?

This is where AI meets emotional intelligence in a virtual companion. It’s a playground for AI enthusiasts, animators, and developers to experiment with reinforcement learning, natural language processing (NLP), and Live2D animation. The result? A responsive, emotionally aware avatar that evolves with every interaction.

---

## 🚀 Key Features

- **Syntrometrie Framework**:  
  - *Metronic Lattice*: Discretizes states for efficient mapping.  
  - *Syntrix Korporator*: Shapes the agent’s belief system.  
  - *Struktur Kaskade*: Links reasoning steps seamlessly.  
- **Dynamic Responses**: Combines emotional states with optional text embeddings for depth.  
- **NLP Backbone**:  
  - Lightweight **GPT-style Transformer** for natural chat.  
  - **Sentence Transformers** for rich semantic embeddings.  
  - **BPE Tokenizer** for fast, efficient text processing.  
- **Avatar Animation**:  
  - Procedural effects (breathing, blinking, idle sway).  
  - Emotion-driven expressions and predictive head tilts.  
- **Learning Core**:  
  - **Prioritized Experience Replay (PER)** for smarter reinforcement learning.  
  - Asynchronous training for smooth, lag-free performance.  

---

## 📊 Project Status

- **Current State**: Actively evolving.  
- **Next Steps**:  
  - Enhanced RL algorithms for faster learning.  
  - Expanded movement library for richer animations.  
  - Performance optimizations for real-time responsiveness.  

---

## 🛠️ How It Works

1. **State Initialization**: `EmotionalSpace` sets the emotional tone from chats or events.  
2. **Text Processing**: Optional `SentenceTransformer` embeddings add nuance.  
3. **AI Decision**: `ConsciousAgent` selects actions (e.g., a nod) based on its beliefs.  
4. **Animation**: `Live2DCharacter` renders emotions and motions in real time.  
5. **Learning Loop**: RL refines behavior with PER, running asynchronously for efficiency.  

For a deeper dive into the architecture, check out the [Mermaid Diagrams](#architecture-diagrams).

---

## 🚀 Get Started

### Prerequisites
- **Python**: 3.8+.  
- **Dependencies**:  
  ```bash
  pip install torch numpy PyQt5 PyOpenGL PyOpenGL-accelerate qasync live2d-py[cubism3] tokenizers sentence-transformers
  ```  
- **Live2D Core**: Grab the native library (`.dll`, `.so`, `.dylib`) from [Live2D](https://www.live2d.com/en/). Place it in the project root or system path. See [live2d-py](https://github.com/Arkueid/live2d-py/blob/main/README.en.md) for setup help.

### Setup
1. **Add a Model**:  
   - Drop your Cubism 3 model (e.g., `model.model3.json`) into `./models/`.  
   - Update `GraphicsConfig.MODEL_PATH` in `config.py`.  
2. **Training Data**:  
   - Add `train_data.json` (format below).  
   - Adjust `TRAINING_DATA_PATH` in `config.py` if needed.  

### Run It
```bash
python main.py
```  
- **Controls**:  
  - `Space`: Toggle pause/play.  
  - `Q`/`Esc`: Exit.  
  - `C`: Check completeness.  
  - Chat via the GUI.  

---

## 🧩 Core Components

| File             | Purpose                                                      |
|------------------|-------------------------------------------------------------|
| `config.py`      | Central hub for agent, RL, NLP, graphics, and env settings. |
| `agent.py`       | `ConsciousAgent`: Drives RL and movement predictions.       |
| `environment.py` | `EmotionalSpace`: Simulates emotional context.              |
| `ai_modules.py`  | PyTorch modules: `EmotionalModule`, `SimpleGPT`, and more.  |
| `graphics.py`    | `Live2DCharacter`: Powers real-time animations.             |
| `gui_widgets.py` | HUD and state panel widgets for the GUI.                    |
| `main_gui.py`    | Main window with the update loop.                           |
| `orchestrator.py`| Syncs agent, environment, and avatar.                       |
| `utils.py`       | Helpers, `Experience` class, and PER memory logic.          |
| `main.py`        | Entry point with `asyncio` integration.                     |

---

## 📚 Training Data Format

Fuel for the tokenizer, `SimpleGPT`, and movement training:

```json
{
  "situation": "User asks 'how are you?'",
  "output": "I’m feeling awesome, thanks!",
  "emotion_weights": [0.7, 0.2, 0.0, 0.0],  // e.g., [happy, calm, sad, angry]
  "head_movement": "gentle_nod"
}
```

---

## 🎨 Customize It

Tweak `config.py` to:  
- Enable/disable embeddings: `AgentConfig.USE_LANGUAGE_EMBEDDING`.  
- Adjust movement learning: `RLConfig.HEAD_MOVEMENT_LOSS_WEIGHT`.  
- Switch models: `GraphicsConfig.MODEL_PATH`.  

---

## 🖥️ Architecture Diagrams

Below are the key architectural components of the project visualized using **Mermaid diagrams**. These diagrams provide a high-level overview of the system's structure and interactions.

### **1. ConsciousAgent Architecture**

```mermaid
graph TD
    Agent_Model["ConsciousAgent (DDQN)"] --> EmoModule["Emotional Module"]
    Agent_Model -->|"Reward (r)"| EmoModule
    Agent_Model -->|"Prev Emotions"| EmoModule
    EmoModule -->|"Updated Emotions"| StateProcMerge["State w/ Updated Emotions"]
    Agent_Model -->|"Other State Components"| StateProcMerge
    StateProcMerge --> Lattice["MetronicLattice"]
    Lattice --> Korporator["SyntrixKorporator"]
    Korporator --> Kaskade["StrukturKaskade"]
    Kaskade --> ValueHead["Value Head (V(s))"]
    Kaskade --> HMHead["Head Movement Head (Supervised)"]
    ValueHead --> Agent_Learn["DDQN Value Loss"]
    HMHead --> Agent_Step_Out["Select HM Label (ArgMax)"]
    Agent_Model -->|"State History (H)"| Accessibility["Compute Accessibility\n(R_acc)"]
    Accessibility --> BoxScore["Compute Box Score"]
    BoxScore --> Agent_Learn
    Kaskade --> Consistency["Compute Consistency\n(rho_score)"]
    Consistency --> Agent_Learn["Intrinsic Reward Calc"]
```

### **2. Syntrometrie Framework**

```mermaid
graph TD
    %% Foundational Layer
    subgraph "A: Foundational Logic"
        A1["Primordial Exp.\n(ästhetische Empirie)"] --> A2["Reflection Synthesis\n(Endo/Exo)"]
        A2 --> A3["Subjective Aspect (S)\n(Mental State)"]
        A3 --> A4["Predicates P_n = [f_q]_n"]
        A3 --> A5["Dialectics D_n = [d_q]_n"]
        A3 --> A6["Coordination K_n = E_n F(ζ_n, z_n)"]
        A4 --> A6
        A5 --> A6
        A7["Antagonismen\n(Logical Tensions)"] --> A5
        A3 --> A8["Aspect Systems A = α(S)"]
        A9["Categories γ\n(Invariant Grounding)"] --> A3
    end

    %% Recursive Structure Layer
    subgraph "B: Recursive Hierarchy"
        B1["Metrophor a ≡ (a_i)_n\n(Base Qualia)"] --> B2["Synkolator Functor F\nGenerates L_{k+1} from L_k"]
        B2 --> B3["Syntrix Levels L_k = F^k(L0)\n(Hierarchical Constructs)"]
        B3 --> B4["Syntrix\n(Union L_k)\n⟨{, a, m⟩"]
        B5["Recursive Def.\na = ⟨{, a, m⟩"] --> B2
        B6["Normalization\n(Stabilizes Recursion)"] --> B2
        B7["Hierarchical Coord.\nK_Syntrix = ∏ K_n"] --> B4
        A9 --> B1
    end

    %% Geometric Layer
    subgraph "C: Geometric Structure"
        C1["12D Hyperspace (H12)\n(Underlying Reality)"] <-->|"Maps Onto"| B4
        C2["N=6 Stability\n(Physical Constraint)"] --> C1
        C1 --> C3["Metric Tensor\ng_ik^γ(x) = sum f_q^i(x) f_q^k(x)"]
        C3 --> C4["Connection\nΓ^i_kl"]
        C4 --> C5["Curvature\nR^i_klm = sum (...)"]
        C6["Quantized Change\nδφ = φ(n) - φ(n-1)"] --> C1
        B3 --> C3
        C2 --> C3
        C7["Mass Formula\n(Link to Physics)"] <-- "Relates to" --> C3
    end

    %% Emergence Layer
    subgraph "D: Reflexive Integration"
        D1["RIH\n(Reflexive Integration Hypothesis)"]
        C3 --> D2["Integration Measure\nI(S) = sum MI_d(S) > τ(t)"]
        C5 --> D2
        B4 --> D3["Reflexivity Cond.\nρ: Id_S → F^n"]
        B5 --> D3
        C2 --> D4["Threshold\nτ = τ_0(N=6) + Δτ(t)"]
        C3 --> D4
        D2 --> D1
        D3 --> D1
        D4 --> D1
        D1 --> D5["Emergent Properties\n(e.g., Consciousness)"]
        A7 --> D5
    end
```

---

## 📈 Future Work

- **Enhanced RL Algorithms**: Experiment with advanced techniques like Deep Deterministic Policy Gradient (DDPG) or Soft Actor-Critic (SAC).  
- **Expanded Animations**: Add more complex movements and gestures for richer interactions.  
- **Performance Optimizations**: Optimize rendering and training pipelines for real-time performance.  
- **Interdisciplinary Research**: Explore connections to philosophy, neuroscience, and AI ethics.  

---

This README now includes embedded Mermaid diagrams that will render directly on GitHub. Let me know if you'd like further refinements! 🚀
