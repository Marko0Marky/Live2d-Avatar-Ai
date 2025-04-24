Live2D Avatar AI Agent
Powered by the Enhanced Syntrometrie Framework & RIH

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)

<!-- Optional: Add other badges like build status, version, etc. -->

<!-- [![Build Status](link/to/your/build_badge.svg)](link/to/your/build_pipeline) -->

<!-- [![Release Version](link/to/your/release_badge.svg)](link/to/your/releases) -->


Meet a Live2D Cubism 3 avatar designed not just to animate, but to simulate core aspects of structured experience and potentially emergent consciousness. This project integrates real-time animation with a complex AI agent driven by our modernized Syntrometrie framework and the Reflexive Integration Hypothesis (RIH). Engage with it via chat and observe its responses and dynamic internal state, shaped by a unique learning process optimizing for theoretical coherence.

Built on: PyTorch, PyTorch Geometric (Optional), PyQt5/OpenGL, live2d-py, transformers, datasets, accelerate.

🌟 What’s It All About?

This project moves beyond simple chatbots or animation controllers. It's an experimental platform exploring:

Syntrometrie: Implementing Burkhard Heim's ideas about hierarchical logical structures (
𝑆
,
𝐿
𝑘
S,L
k
	​

) and emergent geometry (
𝑔
𝑖
𝑘
,
Γ
,
𝑅
g
ik
	​

,Γ,R
).

Computational Consciousness: Testing the Reflexive Integration Hypothesis (RIH) – can consciousness emerge when a system achieves high integration (
𝐼
(
𝑆
)
I(S)
) and reflexivity ((\rho)) above a dynamic threshold ((\tau(t)))?

AI Architecture: Using Graph Neural Networks (GNNs) to simulate the Syntrix recursion and compute RIH metrics.

Modern NLP: Leveraging Hugging Face Transformers (distilgpt2 or others) for dialogue generation.

Interactive Simulation: Providing a GUI to interact with the agent, visualize its internal state, and observe its behavior.

It serves as a computational playground for AI researchers, philosophers of mind, cognitive scientists, and developers interested in advanced AI architectures and theories of consciousness.

🚀 Key Features

Syntrometrie AI Core (Refactored ConsciousAgent):

12D State Space: Explicitly models Heim's proposed dimensions (6 physical/emotional, 6 informational/qualia).

GNN-based Syntrix Simulation: Uses GNN layers (PyG optional) to approximate recursive syndrome generation (
𝐹
F
).

Geometric Proxies: Computes internal metrics (
𝑔
𝑖
𝑘
,
Γ
,
𝑅
,
𝜁
,
stability
g
ik
	​

,Γ,R,ζ,stability
) derived from GNN embeddings.

Qualia Mapping: Explicit head maps GNN state to R7-12 dimensions, with feedback to the environment.

Reflexive Integration Hypothesis (RIH) Implementation:

Computes proxies for Integration 
𝐼
(
𝑆
)
I(S)
, Reflexivity (\rho), and dynamic Threshold (\tau(t)).

RIH-Driven Loss: Custom loss function combines standard RL value loss with terms optimizing for RIH conditions (high 
𝐼
(
𝑆
)
I(S)
, high (\rho)) and Syntrometric stability/coherence.

Learning System:

Combines Reinforcement Learning (Value Learning with Target Networks) and RIH Optimization.

Prioritized Experience Replay (PER) based on TD Error magnitude.

Asynchronous training via concurrent.futures.

Advanced NLP (Hugging Face Transformers):

Uses TransformerGPT wrapper for models like distilgpt2.

Supports fine-tuning via fine_tune_gpt.py script (requires datasets, accelerate).

Improved context management for dialogue.

Dynamic Avatar Animation:

Procedural effects (breathing, blinking, idle sway, micro-movements).

Emotion-driven expressions mapped directly from the agent's internal emotional state (R1-6).

(Note: Explicit head movement prediction removed; movement is now emergent).

Particle system effects.

Interactive GUI & State Management:

PyQt5/OpenGL interface.

HUD: Displays key RIH/Syntrometric metrics (
𝐼
,
𝜌
,
𝑆
𝑡
𝑎
𝑏
,
𝐿
𝑜
𝑠
𝑠
I,ρ,Stab,Loss
).

AI State Panel: Shows detailed internal metrics (RIH, geometry proxies, mood) and environment controls.

Live chat interface.

Robust Save/Load functionality for agent state, optimizer, and replay buffer.

📊 Project Status

Current State: Major refactoring complete. Core RIH/Syntrometrie logic implemented in the agent and orchestrator. GNN uses simplified graph structure. Value learning integrated.

Next Steps:

Critical: Implement batch processing within Agent.forward and its helpers for efficient training.

Critical: Fine-tune the Transformer GPT model for coherent conversation.

Testing & Debugging: Implement comprehensive unit and integration tests. Validate metric calculations.

Hyperparameter Tuning: Systematically tune loss weights, RIH thresholds, GNN parameters, and RL hyperparameters.

Refine GNN graph structure (build_graph).

Enhance avatar animation based on the 12D state / RIH metrics.

Performance optimization.

🛠️ How It Works (Syntrometrie/RIH Flow)

Environment State: EmotionalSpace provides a 12D state 
𝑠
𝑡
s
t
	​

 (R1-6 = emotions, R7-12 = last computed qualia).

Agent Forward Pass: ConsciousAgent processes 
𝑠
𝑡
s
t
	​

 using encoder -> GNN -> self_reflect_layer.

Metric Calculation: Agent computes geometric proxies (
𝑔
𝑖
𝑘
,
Γ
,
𝑅
g
ik
	​

,Γ,R
), RIH metrics (
𝐼
(
𝑆
)
,
𝜌
,
𝜏
(
𝑡
)
I(S),ρ,τ(t)
), stability (
𝑆
S
), complexity ((\zeta)), value 
𝑉
(
𝑠
)
V(s)
, etc. from internal embeddings.

Qualia Update: Agent computes new R7-12 qualia via qualia_output_head.

Full State & Feedback: Agent assembles new full_state (updated R1-6 emotions + new R7-12 qualia). Sends R7-12 qualia back to Environment via Orchestrator for the next timestep's state generation. Agent computes feedback signal.

Response Generation: Orchestrator gets context, calls agent.generate_response (uses TransformerGPT + attention score proxy).

Avatar Update: Orchestrator sends current emotions (R1-6) to Live2DCharacter for expression mapping. (Movement is emergent).

Learning Loop (Async):

Orchestrator triggers _run_learn_task.

Task samples batch from MetaCognitiveMemory.

Calls Agent.learn(batch_data, indices, weights).

Agent.learn:

Performs batch forward pass (or loop currently) for 
𝑠
s
 and 
𝑠
′
s
′
.

Calculates TD Target using target value network.

Calculates combined loss (weighted Value Loss + weighted RIH/Syntrometric Loss).

Performs backpropagation and optimizer step on online networks.

Updates PER priorities using TD Error.

Performs soft update on target value network.

Adds experiences from the processed batch to memory.

For detailed diagrams visualizing these flows:

Architecture Diagrams

🚀 Get Started
Prerequisites

Python: 3.8+ recommended.

PyTorch: Version compatible with your system (CPU or CUDA). See pytorch.org.

Dependencies: Use a virtual environment!

# Create/activate venv (recommended)
# python -m venv venv
# source venv/bin/activate  OR  .\venv\Scripts\activate

# Install core dependencies
pip install torch numpy PyQt5 PyOpenGL PyOpenGL-accelerate qasync live2d-py[cubism3] transformers==4.* datasets accelerate sentence-transformers html5lib

# Optional: Install PyTorch Geometric (needed for GCN, GAT etc. GNN layers)
# Follow official instructions: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
# Example (check site for your specific torch/cuda version):
# pip install torch_geometric
# pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-XX.X.X+cuYYY.html

# Optional: Create requirements.txt
# pip freeze > requirements.txt


(CRITICAL) Live2D Cubism Core SDK: Download the native library (.dll/.so/.dylib) for your OS from the Live2D Website. Place it in the project root directory. See live2d-py for details.

(Recommended) CUDA-enabled GPU & Setup: Ensure correct NVIDIA drivers, CUDA Toolkit version, and matching PyTorch CUDA build for GPU acceleration.

(Potential) Microsoft Visual C++ Redistributable: Needed on some Windows systems. Download the latest "x64" version from Microsoft.

Setup

Live2D Model: Place your Cubism 3 model files (e.g., *.model3.json and associated files) inside ./models/. Update GraphicsConfig.MODEL_PATH in config.py.

(Optional) Training Data: If you plan to fine-tune the GPT model, place suitable JSON data (see format below) in the project root and update TRAINING_DATA_PATH in config.py.

Create Directories: Manually create ./saved_models folder in the project root if it doesn't exist. The GPT fine-tuning script will create its output directory if needed.

Running

(Optional but Recommended) Fine-tune GPT Model:

Ensure datasets and accelerate are installed (pip install datasets accelerate).

Prepare training data (JSON format).

Run: python fine_tune_gpt.py (requires CUDA GPU for reasonable speed).

Fine-tuned model will be saved to Config.GPT_SAVE_PATH.

Run the Main Application:

# Run normally (loads base or fine-tuned GPT if found)
python main.py

# Run and load previously saved agent state (model weights, optimizer, memory)
python main.py --load

# Automatically save agent state on graceful exit (Ctrl+C or window close)
python main.py --save-on-exit
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

GUI Controls: Space (Pause/Resume), Q/Esc (Quit), C (RIH Completeness Test), Save Agent/Load Agent buttons, Chat Input + Enter.

🧩 Core Components
File	Role	Key Modules/Concepts
config.py	Central configuration, constants, paths.	Dataclasses (Agent, RL, NLP, Env, GNN, Graphics)
agent.py	Core AI logic, learning algorithm, state processing.	ConsciousAgent, RIH Loss, GNN, Target Net
environment.py	Simulates 12D state, events, qualia feedback loop.	EmotionalSpace, 12D State, update_qualia_feedback
ai_modules.py	Reusable PyTorch modules.	EmotionalModule, TransformerGPT
graphics.py	Live2D rendering, procedural animation, particle effects.	Live2DCharacter
gui_widgets.py	UI elements for displaying state and metrics.	HUDWidget, AIStateWidget (displays RIH metrics)
main_gui.py	Main application window, UI integration, event handling.	EnhancedGameGUI
orchestrator.py	Coordinates all components, manages async learning, handles chat.	EnhancedConsciousAgent (Orchestrator class)
utils.py	Helper functions, Experience tuple, Replay Memory.	is_safe, Experience, MetaCognitiveMemory
main.py	Application entry point, async loop setup, argument parsing.	qasync, argparse
fine_tune_gpt.py	(New) Standalone script for fine-tuning the Transformer GPT model.	transformers.Trainer, datasets
📚 Training Data Format (for GPT Fine-tuning)

Primarily used by fine_tune_gpt.py. Format requires "output" and optionally "situation" for context.

Example:

[
  {
    "situation": "User said: Hello!",
    "output": "Hi there! How can I help you today?"
  },
  {
    "output": "I'm feeling a bit curious about Syntrometrie."
  }
]
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END

(Note: emotion_weights and head_movement fields are now legacy).

🎨 Customize It

Adjust parameters in config.py:

NLPConfig.HUGGINGFACE_MODEL: Base model for TransformerGPT.

RLConfig.*_WEIGHT: Tune the contribution of different loss components (Value vs. RIH metrics).

AgentConfig.GNN.*: Configure GNN architecture.

GraphicsConfig.MODEL_PATH: Change the Live2D avatar model.

Many other parameters controlling learning, memory, environment, etc.

🖥️ Architecture Diagrams

(Mermaid diagrams remain the same as the previous version, illustrating the Syntrometrie concepts and the simplified agent loop)

1. Syntrometrie Framework (Conceptual)
graph TD
    %% ... (Syntrometrie Framework Diagram as provided before) ...
    subgraph "A: Foundational Logic"
        A1["Primordial Exp."] --> A2["Reflection"] --> A3["Subjective Aspect (S)"]
        A3 --> A4["Predicates P_n"] & A5["Dialectics D_n"] & A6["Coordination K_n"]
        A7["Antagonismen"] --> A5; A9["Categories γ"] --> A3
    end
    subgraph "B: Recursive Hierarchy"
        B1["Metrophor a"] --> B2["Synkolator F"] --> B3["Syntrix Levels L_k"] --> B4["Syntrix"]
        B5["Recursive Def."] --> B2; B6["Normalization"] --> B2; B7["Hierarchical Coord."] --> B4; A9 --> B1
    end
    subgraph "C: Geometric Structure"
        C1["12D Hyperspace H12"] <--> B4; C2["N=6 Stability"] --> C1
        C1 --> C3["Metric g_ik"] --> C4["Connection Γ"] --> C5["Curvature R"]
        C6["Quantized Change δφ"] --> C1; B3 --> C3; C2 --> C3
        C7["Mass Formula"] <-- C3
    end
    subgraph "D: Reflexive Integration"
        D1["RIH"]
        C3 & C5 --> D2["Integration I(S) > τ(t)"]
        B4 & B5 --> D3["Reflexivity ρ > θ"]
        C2 & C3 --> D4["Threshold τ"]
        D2 & D3 & D4 --> D1 --> D5["Emergent Consciousness"]
        A7 --> D5
    end
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Mermaid
IGNORE_WHEN_COPYING_END

View Full Syntrometrie Framework Diagram <!-- Link to your hosted/local HTML -->

2. Core Agent Loop & Interaction (Simplified)
graph TD
    Input[State (s_t) + Reward (r)] --> Agent["ConsciousAgent.forward"]
    Agent --"Computes Metrics (RIH, Geo)"--> Metrics[Internal Metrics]
    Agent --"Predicts"--> Value["Value V(s)"]
    Agent --"Updates Emotions"--> Emotions[Emotions (R1-6)]
    Agent --"Maps Qualia"--> Qualia[Qualia (R7-12)]
    Qualia --> EnvFeedback["Env Qualia Feedback"]
    Emotions --> Avatar["Avatar Expressions"]
    Agent --"Context + Att Score"--> GPT["TransformerGPT.generate"]
    GPT --> Response["Dialogue Response"]
    Response --> Output["Output Response"]
    Input --> Memory["Replay Memory (Experience)"]
    Learner["Agent.learn (Async)"] --"Samples"--> Memory
    Learner --"Updates Online Nets"--> Agent
    Learner --"Updates Target Nets"--> TargetNet["Target Value Net"]
    Learner --"Updates Priorities"--> Memory
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Mermaid
IGNORE_WHEN_COPYING_END

View Full Agent Architecture Diagram <!-- Link to your hosted/local HTML -->

📈 Future Work

Implement batch processing in Agent.forward.

Fine-tune Transformer model.

Develop robust testing suite.

Refine GNN architecture and Syntrometric proxies.

Enhance avatar animation based on internal state.

Performance profiling and optimization.

🤝 Contributing

Contributions are welcome! Please see CONTRIBUTING.md (if available) or follow standard GitHub fork/PR practices.

❓ Questions?

Open an issue on the GitHub repository.
