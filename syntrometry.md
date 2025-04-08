```markdown
# Syntrometrie Agent Architecture (DDQN Version) - Conceptual Flow

```mermaid
graph TD
    %% --- Inputs ---
    In_BaseState[("Environment Base State (Emotions + Meta)")] --> Orch
    In_LangEmbed[("Language Embedding (Monologue/Chat)")] --> Orch
    In_Reward[("Environment Reward (r)")] --> Agent_Learn

    %% --- Orchestrator ---
    subgraph Orchestrator
        Orch(Combine State) --> Agent_Step[Agent Interaction Step]
        Orch --> Agent_Learn[Agent Learning Step]
        Agent_Step -- Combined State (s_t) --> Agent_Model["ConsciousAgent (DDQN)"]
        Agent_Learn -- Batch (s, a, r, s', d, hm_idx) --> Agent_Model %% hm_idx was target label
    end

    %% --- Conscious Agent (DDQN Version) ---
    subgraph "ConsciousAgent (DDQN Version)"
        %% Input State Processing
        Agent_Model -- State (s_t) --> EmoModule
        Agent_Model -- Reward (r) --> EmoModule
        Agent_Model -- Prev Emotions --> EmoModule
        EmoModule[Emotional Module] -- Updated Emotions --> StateProcMerge
        Agent_Model -- Other State Components --> StateProcMerge
        StateProcMerge -- State w/ Updated Emotions --> Lattice

        %% Syntrometrie Core
        Lattice[MetronicLattice] -- Discretized State (phi_t) --> Korporator
        Agent_Model -- State History (H) --> Attention
        Attention[Self-Attention] -- Context (psi_t) --> Korporator
        Korporator[SyntrixKorporator] -- Raw Belief (belief_raw) --> Kaskade
        Korporator -- Raw Belief (belief_raw) --> Consistency["Compute Consistency (rho_score)"]
        Kaskade[StrukturKaskade] -- Structured Belief (belief) --> ValueHead
        Kaskade -- Structured Belief (belief) --> HMHead
        Kaskade -- Structured Belief (belief) --> FeedbackNet
        Kaskade -- Structured Belief (belief) --> Consistency

        %% Output Heads & Learning Signals
        ValueHead[Value Head (V(s))] -- Estimated Value --> Agent_Learn["DDQN Value Loss"]
        HMHead[Head Movement Head (Supervised)] -- HM Logits --> Agent_Step_Out["Select HM Label (ArgMax)"]
        HMHead -- HM Logits & Target Index --> Agent_Learn["HM Supervised Loss"]
        FeedbackNet[Feedback Network] -- Feedback Signal (unused?) --- Agent_Model

        %% Intrinsic Metrics
        Agent_Model -- State History (H) --> Accessibility["Compute Accessibility (R_acc)"]
        Accessibility -- R_acc --> BoxScore["Compute Box Score"]
        Consistency -- rho_score --> Agent_Learn["Intrinsic Reward Calc"]
        BoxScore -- box_score --> Agent_Learn

        %% Dialogue Generation
        Agent_Model -- EmoModule Output --> GPT["TransformerGPT (Dialogue)"] %% Uses updated emotions
        GPT -- Dialogue Response --> Agent_Step_Out

        %% Agent Outputs
        Agent_Step_Out --> Output_HM["Output: Head Movement Label"]
        Agent_Step_Out --> Output_Dialogue["Output: Dialogue Response"]
    end

    %% --- External Systems ---
    Output_HM --> Avatar["Live2D Avatar Animation"]
    Output_Dialogue --> GUI["GUI Chat Display"]

    %% --- Styles ---
    classDef input fill:#D6EAF8,stroke:#5DADE2,stroke-width:2px;
    classDef module fill:#D5F5E3,stroke:#58D68D,stroke-width:2px;
    classDef metric fill:#FCF3CF,stroke:#F4D03F,stroke-width:2px;
    classDef output fill:#FADBD8,stroke:#EC7063,stroke-width:2px;
    classDef agent fill:#E8DAEF,stroke:#A569BD,stroke-width:2px;
    classDef external fill:#E5E7E9,stroke:#85929E,stroke-width:2px;
    classDef stateproc fill:#D6DBDF,stroke:#839192,stroke-width:1px;

    class In_BaseState,In_LangEmbed,In_Reward input;
    class Orch,Agent_Step,Agent_Learn,Agent_Model agent;
    class StateProcMerge stateproc
    class Lattice,Attention,Korporator,Kaskade,ValueHead,HMHead,FeedbackNet,EmoModule,GPT module;
    class Accessibility,Consistency,BoxScore metric;
    class Agent_Step_Out agent;
    class Output_HM,Output_Dialogue output;
    class Avatar,GUI external;

```

# Syntrometrie Theoretical Framework

```mermaid
graph TD
    %% --- Foundational Layer ---
    subgraph "A: Foundational Subjective Logic (Ch 1)"
        A1[("Primordial Experience\n(ästhetische Empirie)")] --> A2{"Reflection Synthesis (Endo/Exo)"};
        A2 --> A3["Subjective Aspect (S)\n(Mental State Snapshot)"];
        A3 --> A4["Predicate Bands (P_n / P_nn)\n(Features/Inputs)"];
        A3 --> A5["Dialectic Qualifiers (D_n / D_{nn})\n(Emphasis/Nuance)"];
        A3 --> A6["Coordination (K_n / chi_q)\n(Coherence Filter)"];
        A4 & A5 --> A6;
        A7[("Antagonismen\n(Logical Tensions)")] --> A5; %% Antagonismen influence Dialectics
        A3 --> A8["Aspect Systems (A)\n(Relativistic Views)"];
        A9[("Categories / Apodiktic Elements (gamma)\n(Invariant Grounding)")] --> A3;
    end

    %% --- Recursive Structure Layer ---
    subgraph "B: Syntrix - Recursive Hierarchy (Ch 2-3)"
        B1[("Metrophor (L0 / a)\n(Base Propositions/Qualia)")] --> B2{"Synkolator Functor (F)\n(Generates L_{k+1} from L_k)"};
        B2 -- Creates Levels --> B3["Syntrix Levels (L_k = F^k(L0))\n(Hierarchical Mental Constructs)"];
        B3 -- Forms --> B4["Syntrix\n(Union L_k)"];
        B5["Recursive Definition\n(~a = <F, ~a, m>)"] -- Governs --> B2;
        B6["Normalization\n(Stabilizes Recursion)"] --> B2;
        B7["Hierarchical Coordination\n(K_Syntrix)"] --> B4; %% How levels connect
        A9 --> B1; %% Apodictic elements ground the Metrophor
    end

    %% --- Geometric Layer ---
    subgraph "C: Geometric Structure & Dynamics (Ch 8-11)"
        C1[("12D Hyperspace (H12)\n(Underlying Reality Structure)")] <== Maps Onto ==> B4; %% Syntrix structure relates to H12
        C2["N=6 Selection Principle\n(Physical Stability Constraint)"] --> C1;
        C1 --> C3["Metric Tensor (g_ik^gamma)\n(Focus/Coherence @ Level gamma)"];
        C3 --> C4["Connection (Gamma^i_kl)\n(Attentional Shift/Flow)"];
        C4 --> C5["Curvature (R^i_klm)\n(Complexity/Richness/Integration)"];
        C6[("Metron Differential (∂̃_k)\n(Quantized Change)")] --> C1;
        B3 -- Influences --> C3; %% Syndromes at L_gamma determine g_ik^gamma
        C2 -- Constrains --> C3; %% N=6 constrains g_ik for physical dimensions
        C7[("Mass Formula\n(Link to Physics)")] <-- Relates to --> C3;
    end

    %% --- Emergence & Consciousness Layer ---
    subgraph "D: Reflexive Integration & Emergence"
        D1["Reflexive Integration Hypothesis (RIH)"]
        C3 & C5 --> D2["Integration Measure I(S)\n(Sum MI_d(S) or via g_ik, R)"];
        B4 & B5 --> D3["Reflexivity Condition (rho)\n(F^n ~ Id or Feature Similarity)"];
        C2 & C3 --> D4["Threshold (tau)\n(Stability Constraint g_ik(N=6) + Dynamics)"];
        D2 & D3 & D4 --> D1;
        D1 -- Condition Met --> D5["Emergent Properties\n(e.g., Consciousness, Unified Experience)"];
        A7 -- Resolved By --> D5; %% Antagonismen resolution via unified structure
    end

    %% --- Computational / Semantic Layer ---
    subgraph "E: Computational & Semantic Models"
        E1["Syntrometric Kripke Frame (F)\n(Worlds=S(x), R based on g_ik)"] <== Models ==> A & C & D;
        E2["Sequent Calculus\n(S; Gamma |- phi)"] <== Derives From ==> A & B;
        E1 -- Interprets --> E2;
        E3["Computational Model (GNN)\n(Agent Implementation)"] -- Approximates --> B & C & D3;
        E4["Metrics (att_score, box_score, ...)"] <-- Calculated By --- E3;
        E4 -- Correspond To --> D3 & C3 & D4; %% Map computational metrics back to semantic concepts
    end

    %% --- Style Definitions ---
    classDef foundation fill:#EAECEE,stroke:#AEB6BF,stroke-width:2px;
    classDef structure fill:#E8DAEF,stroke:#A569BD,stroke-width:2px;
    classDef geometry fill:#D6EAF8,stroke:#5DADE2,stroke-width:2px;
    classDef emergence fill:#D5F5E3,stroke:#58D68D,stroke-width:2px;
    classDef computation fill:#FCF3CF,stroke:#F4D03F,stroke-width:2px;

    class A1,A2,A3,A4,A5,A6,A7,A8,A9 foundation;
    class B1,B2,B3,B4,B5,B6,B7 structure;
    class C1,C2,C3,C4,C5,C6,C7 geometry;
    class D1,D2,D3,D4,D5 emergence;
    class E1,E2,E3,E4 computation;
```
```

---
