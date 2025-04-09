```mermaid

graph TD
    %% ConsciousAgent (DDQN Version)
    subgraph "ConsciousAgent (DDQN Version)"
        %% Input State Processing
        Agent_Model["ConsciousAgent (DDQN)"] -- "State (s_t)" --> EmoModule["Emotional Module"]
        Agent_Model -- "Reward (r)" --> EmoModule
        %% Immediate reward potentially influences emotion
        Agent_Model -- "Prev Emotions" --> EmoModule
        EmoModule -- "Updated Emotions" --> StateProcMerge["State w/ Updated Emotions"]
        Agent_Model -- "Other State Components" --> StateProcMerge
        StateProcMerge --> Lattice["MetronicLattice"]

        %% Syntrometrie Core
        Lattice -- "Discretized State (phi_t)" --> Korporator["SyntrixKorporator"]
        Agent_Model -- "State History (H)" --> Attention["Self-Attention"]
        Attention -- "Context (psi_t)" --> Korporator
        Korporator -- "Raw Belief (belief_raw)" --> Kaskade["StrukturKaskade"]
        Korporator -- "Raw Belief (belief_raw)" --> Consistency["Compute Consistency (rho_score)"]
        Kaskade -- "Structured Belief (belief)" --> ValueHead["Value Head (V(s))"]
        Kaskade -- "Structured Belief (belief)" --> HMHead["Head Movement Head (Supervised)"]
        Kaskade -- "Structured Belief (belief)" --> FeedbackNet["Feedback Network"]
        Kaskade -- "Structured Belief (belief)" --> Consistency

        %% Output Heads & Learning Signals
        ValueHead -- "Estimated Value" --> Agent_Learn["DDQN Value Loss"]
        HMHead -- "HM Logits" --> Agent_Step_Out["Select HM Label (ArgMax)"]
        HMHead -- "HM Logits & Target Index" --> Agent_Learn["HM Supervised Loss"]
        FeedbackNet -- "Feedback Signal (unused?)" --> Agent_Model

        %% Intrinsic Metrics
        Agent_Model -- "State History (H)" --> Accessibility["Compute Accessibility (R_acc)"]
        Accessibility -- "R_acc" --> BoxScore["Compute Box Score"]
        Consistency -- "rho_score" --> Agent_Learn["Intrinsic Reward Calc"]
        BoxScore -- "box_score" --> Agent_Learn

        %% Dialogue Generation
        %% Uses updated emotions
        Agent_Model -- "EmoModule Output" --> GPT["TransformerGPT (Dialogue)"]
        GPT -- "Dialogue Response" --> Agent_Step_Out

        %% Agent Outputs
        Agent_Step_Out["Agent Step Output"] --> Output_HM["Output: Head Movement Label"]
        Agent_Step_Out --> Output_Dialogue["Output: Dialogue Response"]
    end

    %% Styles
    classDef agent fill:#E8DAEF,stroke:#A569BD,stroke-width:2px
    classDef module fill:#D5F5E3,stroke:#58D68D,stroke-width:2px
    classDef metric fill:#FCF3CF,stroke:#F4D03F,stroke-width:2px
    classDef output fill:#FADBD8,stroke:#EC7063,stroke-width:2px
    classDef stateproc fill:#D6DBDF,stroke:#839192,stroke-width:1px
    class Agent_Model,Agent_Step_Out agent
    class EmoModule,Lattice,Attention,Korporator,Kaskade,ValueHead,HMHead,FeedbackNet,GPT module
    class Accessibility,Consistency,BoxScore metric
    class Output_HM,Output_Dialogue output
    class StateProcMerge stateproc
