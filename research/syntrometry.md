graph TB
    %% Global Styling
    classDef foundation fill:#EAECEE,stroke:#AEB6BF,stroke-width:2px,color:black,font-size:14px
    classDef structure fill:#E8DAEF,stroke:#A569BD,stroke-width:2px,color:black,font-size:14px
    classDef geometry fill:#D6EAF8,stroke:#5DADE2,stroke-width:2px,color:black,font-size:14px
    classDef emergence fill:#D5F5E3,stroke:#58D68D,stroke-width:2px,color:black,font-size:14px
    classDef computation fill:#FCF3CF,stroke:#F4D03F,stroke-width:2px,color:black,font-size:14px

    %% Foundational Layer
    subgraph "A: Foundational Logic"
        A1["Primordial Exp.\n(ästhetische Empirie)"]:::foundation --> A2["Reflection Synthesis\n(Endo/Exo)"]:::foundation
        A2 --> A3["Subjective Aspect (S)\n(Mental State)"]:::foundation
        A3 --> A4["Predicates P_n = [f_q]_n"]:::foundation
        A3 --> A5["Dialectics D_n = [d_q]_n"]:::foundation
        A3 --> A6["Coordination K_n = E_n F(ζ_n, z_n)"]:::foundation
        A4 --> A6
        A5 --> A6
        A7["Antagonismen\n(Logical Tensions)"]:::foundation --> A5
        A3 --> A8["Aspect Systems A = α(S)"]:::foundation
        A9["Categories γ\n(Invariant Grounding)"]:::foundation --> A3
    end

    %% Recursive Structure Layer
    subgraph "B: Recursive Hierarchy"
        B1["Metrophor a ≡ (a_i)_n\n(Base Qualia)"]:::structure --> B2["Synkolator Functor F\nGenerates L_{k+1} from L_k"]:::structure
        B2 --> B3["Syntrix Levels L_k = F^k(L0)\n(Hierarchical Constructs)"]:::structure
        B3 --> B4["Syntrix\n(Union L_k)\n⟨{, a, m⟩"]:::structure
        B5["Recursive Def.\na = ⟨{, a, m⟩"]:::structure --> B2
        B6["Normalization\n(Stabilizes Recursion)"]:::structure --> B2
        B7["Hierarchical Coord.\nK_Syntrix = ∏ K_n"]:::structure --> B4
        A9 --> B1
    end

    %% Geometric Layer
    subgraph "C: Geometric Structure"
        C1["12D Hyperspace (H12)\n(Underlying Reality)"]:::geometry <-->|"Maps Onto"| B4
        C2["N=6 Stability\n(Physical Constraint)"]:::geometry --> C1
        C1 --> C3["Metric Tensor\ng_ik^γ(x) = sum f_q^i(x) f_q^k(x)"]:::geometry
        C3 --> C4["Connection\nΓ^i_kl"]:::geometry
        C4 --> C5["Curvature\nR^i_klm = sum (...)"]:::geometry
        C6["Quantized Change\nδφ = φ(n) - φ(n-1)"]:::geometry --> C1
        B3 --> C3
        C2 --> C3
        C7["Mass Formula\n(Link to Physics)"]:::geometry <-- "Relates to" --> C3
    end

    %% Emergence Layer
    subgraph "D: Reflexive Integration"
        D1["RIH\n(Reflexive Integration Hypothesis)"]:::emergence
        C3 --> D2["Integration Measure\nI(S) = sum MI_d(S) > τ(t)"]:::emergence
        C5 --> D2
        B4 --> D3["Reflexivity Cond.\nρ: Id_S → F^n"]:::emergence
        B5 --> D3
        C2 --> D4["Threshold\nτ = τ_0(N=6) + Δτ(t)"]:::emergence
        C3 --> D4
        D2 --> D1
        D3 --> D1
        D4 --> D1
        D1 --> D5["Emergent Properties\n(e.g., Consciousness)"]:::emergence
        A7 --> D5
    end

    %% Computational Layer
    subgraph "E: Computational Models"
        E1["Syntrometric Kripke Frame\n(Worlds=S(x), R based on g_ik)"]:::computation -->|"Models"| A
        E1 -->|"Models"| C
        E1 -->|"Models"| D
        E2["Sequent Calculus\n(S; Γ |- ϕ)"]:::computation -->|"Derives From"| A
        E2 -->|"Derives From"| B
        E1 --> E2
        E3["GNN Implementation\n(Agent Model)"]:::computation -->|"Approximates"| B
        E3 -->|"Approximates"| C
        E3 -->|"Approximates"| D3
        E4["Metrics (att_score, box_score, ...)"]:::computation <-- "Calculated By" --> E3
        E4 --> D3
        E4 --> C3
        E4 --> D4
    end

    %% Tooltips for Detailed Formulas
    click A4 "Predicates represent features or inputs (P_n = [f_q]_n)."
    click C3 "Metric tensor measures coherence between states: g_ik^γ(x) = sum f_q^i(x) f_q^k(x)."
    click D2 "Integration measure quantifies mutual information across dimensions: I(S) = sum MI_d(S)."
    click D3 "Reflexivity condition ensures structural self-similarity: ρ: Id_S → F^n."
