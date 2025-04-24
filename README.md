\documentclass[12pt]{article}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

% Define colors for badges
\definecolor{mit-yellow}{RGB}{255, 203, 5}
\definecolor{badge-bg}{RGB}{230, 230, 230}

% Custom command for badges
\newcommand{\badge}[2]{%
    \colorbox{badge-bg}{\textcolor{#1}{\textbf{#2}}}%
}

\title{Live2D Avatar AI Agent: Powered by the Enhanced Syntrometrie Framework \& RIH}
\author{}
\date{}

\begin{document}

\maketitle

\begin{center}
    \badge{mit-yellow}{License: MIT}
    % Optional badges can be added here
\end{center}

\section*{Overview}

This project introduces a Live2D Cubism 3 avatar designed not only to animate but also to simulate core aspects of structured experience and potentially emergent consciousness. It integrates real-time animation with a complex AI agent driven by the modernized Syntrometrie framework and the Reflexive Integration Hypothesis (RIH). Engage with it via chat and observe its responses and dynamic internal state, shaped by a unique learning process optimizing for theoretical coherence.

The project is built on:
[noitemsep]
    
- PyTorch
    
- PyTorch Geometric (Optional)
    
- PyQt5/OpenGL
    
- live2d-py
    
- Transformers
    
- Datasets
    
- Accelerate


\section*{What’s It All About?}

This experimental platform explores:
[noitemsep]
    
- \textbf{Syntrometrie}: Implementing Burkhard Heim's ideas about hierarchical logical structures ($S$, $L_k$) and emergent geometry ($g_{ik}$, $\Gamma$, $R$).
    
- \textbf{Computational Consciousness}: Testing the Reflexive Integration Hypothesis (RIH) – can consciousness emerge when a system achieves high integration ($I(S)$) and reflexivity ($\rho$) above a dynamic threshold ($\tau(t)$)?
    
- \textbf{AI Architecture}: Using Graph Neural Networks (GNNs) to simulate the Syntrix recursion and compute RIH metrics.
    
- \textbf{Modern NLP}: Leveraging Hugging Face Transformers (e.g., distilgpt2) for dialogue generation.
    
- \textbf{Interactive Simulation}: Providing a GUI to interact with the agent, visualize its internal state, and observe its behavior.


It serves as a computational playground for AI researchers, philosophers of mind, cognitive scientists, and developers interested in advanced AI architectures and theories of consciousness.

\section*{Key Features}

\subsection*{Syntrometrie AI Core (Refactored ConsciousAgent)}
[noitemsep]
    
- \textbf{12D State Space}: Explicitly models Heim's proposed dimensions (6 physical/emotional, 6 informational/qualia).
    
- \textbf{GNN-based Syntrix Simulation}: Uses GNN layers (PyG optional) to approximate recursive syndrome generation ($F$).
    
- \textbf{Geometric Proxies}: Computes internal metrics ($g_{ik}, \Gamma, R, \zeta, \text{stability}$) derived from GNN embeddings.
    
- \textbf{Qualia Mapping}: Explicit head maps GNN state to R7-12 dimensions, with feedback to the environment.


\subsection*{Reflexive Integration Hypothesis (RIH) Implementation}
[noitemsep]
    
- Computes proxies for Integration $I(S)$, Reflexivity ($\rho$), and dynamic Threshold ($\tau(t)$).
    
- \textbf{RIH-Driven Loss}: Custom loss function combines standard RL value loss with terms optimizing for RIH conditions (high $I(S)$, high $\rho$) and Syntrometric stability/coherence.


\subsection*{Learning System}
[noitemsep]
    
- Combines Reinforcement Learning (Value Learning with Target Networks) and RIH Optimization.
    
- Prioritized Experience Replay (PER) based on TD Error magnitude.
    
- Asynchronous training via \texttt{concurrent.futures}.


\subsection*{Advanced NLP (Hugging Face Transformers)}
[noitemsep]
    
- Uses TransformerGPT wrapper for models like distilgpt2.
    
- Supports fine-tuning via \texttt{fine\_tune\_gpt.py} script.
    
- Improved context management for dialogue.


\subsection*{Dynamic Avatar Animation}
[noitemsep]
    
- Procedural effects (breathing, blinking, idle sway, micro-movements).
    
- Emotion-driven expressions mapped directly from the agent's internal emotional state (R1-6).
    
- Particle system effects.


\subsection*{Interactive GUI \& State Management}
[noitemsep]
    
- PyQt5/OpenGL interface.
    
- HUD: Displays key RIH/Syntrometric metrics ($I, \rho, \text{Stab}, \text{Loss}$).
    
- AI State Panel: Shows detailed internal metrics (RIH, geometry proxies, mood) and environment controls.
    
- Live chat interface.
    
- Robust Save/Load functionality for agent state, optimizer, and replay buffer.


\section*{Project Status}

\textbf{Current State}: Major refactoring complete. Core RIH/Syntrometrie logic implemented in the agent and orchestrator. GNN uses simplified graph structure. Value learning integrated.

\textbf{Next Steps}:
[noitemsep]
    
- Implement batch processing within \texttt{Agent.forward} and its helpers for efficient training.
    
- Fine-tune the Transformer GPT model for coherent conversation.
    
- Implement comprehensive unit and integration tests. Validate metric calculations.
    
- Systematically tune hyperparameters (loss weights, RIH thresholds, GNN parameters, RL hyperparameters).
    
- Refine GNN graph structure (\texttt{build\_graph}).
    
- Enhance avatar animation based on the 12D state / RIH metrics.
    
- Performance optimization.


\section*{How It Works (Syntrometrie/RIH Flow)}

\begin{enumerate}[noitemsep]
    \item \textbf{Environment State}: EmotionalSpace provides a 12D state $s_t$ (R1-6 = emotions, R7-12 = last computed qualia).
    \item \textbf{Agent Forward Pass}: ConsciousAgent processes $s_t$ using encoder $\rightarrow$ GNN $\rightarrow$ self\_reflect\_layer.
    \item \textbf{Metric Calculation}: Agent computes geometric proxies ($g_{ik}, \Gamma, R$), RIH metrics ($I(S), \rho, \tau(t)$), stability ($S$), complexity ($\zeta$), value $V(s)$, etc., from internal embeddings.
    \item \textbf{Qualia Update}: Agent computes new R7-12 qualia via \texttt{qualia\_output\_head}.
    \item \textbf{Full State \& Feedback}: Agent assembles new full state (updated R1-6 emotions + new R7-12 qualia). Sends R7-12 qualia back to Environment via Orchestrator for the next timestep's state generation.
    \item \textbf{Response Generation}: Orchestrator gets context, calls \texttt{agent.generate\_response} (uses TransformerGPT + attention score proxy).
    \item \textbf{Avatar Update}: Orchestrator sends current emotions (R1-6) to Live2DCharacter for expression mapping.
    \item \textbf{Learning Loop (Async)}: Orchestrator triggers \texttt{_run\_learn\_task}. Task samples batch from MetaCognitiveMemory, calls \texttt{Agent.learn(batch\_data, indices, weights)}.
\end{enumerate}

\section*{Get Started}

\subsection*{Prerequisites}
[noitemsep]
    
- Python: 3.8+ recommended.
    
- PyTorch: Version compatible with your system (CPU or CUDA). See \url{https://pytorch.org}.
    
- Dependencies: Use a virtual environment!


\subsection*{Setup}
[noitemsep]
    
- Place your Cubism 3 model files inside \texttt{./models/}. Update \texttt{GraphicsConfig.MODEL\_PATH} in \texttt{config.py}.
    
- Manually create \texttt{./saved\_models} folder in the project root if it doesn't exist.


\subsection*{Running}
[noitemsep]
    
- Run normally: \texttt{python main.py}.
    
- Load saved agent state: \texttt{python main.py --load}.
    
- Automatically save agent state on graceful exit: \texttt{python main.py --save-on-exit}.


\section*{Core Components}

\begin{tabular}{|l|l|}
    \hline
    \textbf{File} & \textbf{Role} \\
    \hline
    \texttt{config.py} & Central configuration, constants, paths. \\
    \texttt{agent.py} & Core AI logic, learning algorithm, state processing. \\
    \texttt{environment.py} & Simulates 12D state, events, qualia feedback loop. \\
    \texttt{ai\_modules.py} & Reusable PyTorch modules. \\
    \texttt{graphics.py} & Live2D rendering, procedural animation, particle effects. \\
    \texttt{gui\_widgets.py} & UI elements for displaying state and metrics. \\
    \texttt{main\_gui.py} & Main application window, UI integration, event handling. \\
    \texttt{orchestrator.py} & Coordinates all components, manages async learning, handles chat. \\
    \texttt{utils.py} & Helper functions, Experience tuple, Replay Memory. \\
    \texttt{main.py} & Application entry point, async loop setup, argument parsing. \\
    \texttt{fine\_tune\_gpt.py} & Standalone script for fine-tuning the Transformer GPT model. \\
    \hline
\end{tabular}

\section*{Future Work}
[noitemsep]
    
- Implement batch processing in \texttt{Agent.forward}.
    
- Fine-tune Transformer model.
    
- Develop robust testing suite.
    
- Refine GNN architecture and Syntrometric proxies.
    
- Enhance avatar animation based on internal state.
    
- Performance profiling and optimization.


\section*{Contributing}

Contributions are welcome! Please see \texttt{CONTRIBUTING.md} (if available) or follow standard GitHub fork/PR practices.

\section*{Questions?}

Open an issue on the GitHub repository.

\end{document}
