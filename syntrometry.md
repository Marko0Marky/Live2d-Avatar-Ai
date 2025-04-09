```mermaid
graph TD
    %% Foundational Layer
    subgraph "A: Foundational Subjective Logic (Ch 1)"
        A1["Primordial Experience\n(ästhetische Empirie)"] --> A2{"Reflection Synthesis (Endo/Exo)"}
        A2 --> A3["Subjective Aspect (S)\n(Mental State Snapshot)"]
        A3 --> A4["Predicate Bands (P_n / P_nn)\n(Features/Inputs)"]
        A3 --> A5["Dialectic Qualifiers (D_n / D_{nn})\n(Emphasis/Nuance)"]
        A3 --> A6["Coordination (K_n / chi_q)\n(Coherence Filter)"]
        A4 --> A6
        A5 --> A6
        %% Antagonismen influence Dialectics
        A7["Antagonismen\n(Logical Tensions)"] --> A5
        A3 --> A8["Aspect Systems (A)\n(Relativistic Views)"]
        A9["Categories / Apodiktic Elements (gamma)\n(Invariant Grounding)"] --> A3
    end

    %% Recursive Structure Layer
    subgraph "B: Syntrix - Recursive Hierarchy (Ch 2-3)"
        B1["Metrophor (L0 / a)\n(Base Propositions/Qualia)"] --> B2{"Synkolator Functor (F)\n(Generates L_{k+1} from L_k)"}
        B2 -- "Creates Levels" --> B3["Syntrix Levels (L_k = F^k(L0))\n(Hierarchical Mental Constructs)"]
        B3 -- "Forms" --> B4["Syntrix\n(Union L_k)"]
        B5["Recursive Definition\n('a' = F('a', m))"] -- "Governs" --> B2
        B6["Normalization\n(Stabilizes Recursion)"] -- "Stabilizes" --> B2
        %% How levels connect
        B7["Hierarchical Coordination\n(K_Syntrix)"] --> B4
        %% Apodictic elements ground the Metrophor
        A9 --> B1
    end

    %% Geometric Layer
    subgraph "C: Geometric Structure & Dynamics (Ch 8-11)"
        %% Syntrix structure relates to H12 (Bidirectional Link)
        C1["12D Hyperspace (H12)\n(Underlying Reality Structure)"] <-->|"Maps Onto"| B4
        C2["N=6 Selection Principle\n(Physical Stability Constraint)"] --> C1
        C1 --> C3["Metric Tensor (g_ik^gamma)\n(Focus/Coherence @ Level gamma)"]
        C3 --> C4["Connection (Gamma^i_kl)\n(Attentional Shift/Flow)"]
        C4 --> C5["Curvature (R^i_klm)\n(Complexity/Richness/Integration)"]
        C6["Quantized Change\n(Metron Differential)"] --> C1
        %% Syndromes at L_gamma determine g_ik^gamma
        B3 -- "Influences" --> C3
        %% N=6 constrains g_ik for physical dimensions
        C2 -- "Constrains" --> C3
        %% Link to Physics
        C7["Mass Formula\n(Link to Physics)"] <-- "Relates to" --> C3
    end

    %% Emergence & Consciousness Layer
    subgraph "D: Reflexive Integration & Emergence"
        D1["Reflexive Integration Hypothesis (RIH)"]
        C3 --> D2["Integration Measure I(S)\n(Sum MI_d(S) or via g_ik, R)"]
        C5 --> D2
        B4 --> D3["Reflexivity Condition (rho)\n(F^n ~ Id or Feature Similarity)"]
        B5 --> D3
        C2 --> D4["Threshold (tau)\n(Stability Constraint g_ik(N=6) + Dynamics)"]
        C3 --> D4
        D2 --> D1
        D3 --> D1
        D4 --> D1
        D1 -- "Condition Met" --> D5["Emergent Properties\n(e.g., Consciousness, Unified Experience)"]
        %% Antagonismen resolution via unified structure
        A7 -- "Resolved By" --> D5
    end

    %% Computational / Semantic Layer
    subgraph "E: Computational & Semantic Models"
        E1["Syntrometric Kripke Frame (F)\n(Worlds=S(x), R based on g_ik)"] -->|"Models"| A
        E1 -->|"Models"| C
        E1 -->|"Models"| D
        E2["Sequent Calculus\n(S; Gamma |- phi)"] -->|"Derives From"| A
        E2 -->|"Derives From"| B
        E1 -- "Interprets" --> E2
        E3["Computational Model (GNN)\n(Agent Implementation)"] -->|"Approximates"| B
        E3 -->|"Approximates"| C
        E3 -->|"Approximates"| D3
        E4["Metrics (att_score, box_score, ...)"] <-- "Calculated By" --> E3
        E4 -- "Correspond To" --> D3
        E4 -- "Correspond To" --> C3
        E4 -- "Correspond To" --> D4
    end

    %% Style Definitions
    classDef foundation fill:#EAECEE,stroke:#AEB6BF,stroke-width:2px
    classDef structure fill:#E8DAEF,stroke:#A569BD,stroke-width:2px
    classDef geometry fill:#D6EAF8,stroke:#5DADE2,stroke-width:2px
    classDef emergence fill:#D5F5E3,stroke:#58D68D,stroke-width:2px
    classDef computation fill:#FCF3CF,stroke:#F4D03F,stroke-width:2px

    class A1,A2,A3,A4,A5,A6,A7,A8,A9 foundation
    class B1,B2,B3,B4,B5,B6,B7 structure
    class C1,C2,C3,C4,C5,C6,C7 geometry
    class D1,D2,D3,D4,D5 emergence
    class E1,E2,E3,E4 computation
