```mermaid
graph TD
    %% Global Styling
    classDef agent fill:#E8DAEF,stroke:#A569BD,stroke-width:2px,color:black,font-size:14px
    classDef module fill:#D5F5E3,stroke:#58D68D,stroke-width:2px,color:black,font-size:14px
    classDef metric fill:#FCF3CF,stroke:#F4D03F,stroke-width:2px,color:black,font-size:14px
    classDef output fill:#FADBD8,stroke:#EC7063,stroke-width:2px,color:black,font-size:14px
    classDef stateproc fill:#D6DBDF,stroke:#839192,stroke-width:1px,color:black,font-size:14px

    %% ConsciousAgent (DDQN Version)
    subgraph "ConsciousAgent (DDQN Version)"
        %% Input State Processing
        Agent_Model["ConsciousAgent (DDQN)"]:::agent -- "State (s_t)" --> EmoModule["Emotional Module"]:::module
        Agent_Model -- "Reward (r)" --> EmoModule
        Agent_Model -- "Prev Emotions" --> EmoModule
        EmoModule -- "Updated Emotions" --> StateProcMerge["State w/ Updated Emotions"]:::stateproc
        Agent_Model -- "Other State Components" --> StateProcMerge
        StateProcMerge --> Lattice["MetronicLattice"]:::module

        %% Syntrometrie Core
        subgraph "Syntrometrie Core"
            Lattice -- "Discretized State (phi_t)" --> Korporator["SyntrixKorporator"]:::module
            Agent_Model -- "State History (H)" --> Attention["Self-Attention"]:::module
            Attention -- "Context (psi_t)" --> Korporator
            Korporator -- "Raw Belief (belief_raw)" --> Kaskade["StrukturKaskade"]:::module
            Korporator -- "Raw Belief (belief_raw)" --> Consistency["Compute Consistency\n(rho_score)"]:::metric
            Kaskade -- "Structured Belief (belief)" --> ValueHead["Value Head (V(s))"]:::module
            Kaskade -- "Structured Belief (belief)" --> HMHead["Head Movement Head\n(Supervised)"]:::module
            Kaskade -- "Structured Belief (belief)" --> FeedbackNet["Feedback Network"]:::module
            Kaskade -- "Structured Belief (belief)" --> Consistency
        end

        %% Output Heads & Learning Signals
        subgraph "Learning Signals"
            ValueHead -- "Estimated Value" --> Agent_Learn["DDQN Value Loss"]:::metric
            HMHead -- "HM Logits" --> Agent_Step_Out["Select HM Label (ArgMax)"]:::agent
            HMHead -- "HM Logits & Target Index" --> Agent_Learn["HM Supervised Loss"]
            FeedbackNet -- "Feedback Signal (unused?)" --> Agent_Model
        end

        %% Intrinsic Metrics
        subgraph "Intrinsic Metrics"
            Agent_Model -- "State History (H)" --> Accessibility["Compute Accessibility\n(R_acc)"]:::metric
            Accessibility -- "R_acc" --> BoxScore["Compute Box Score"]:::metric
            Consistency -- "rho_score" --> Agent_Learn["Intrinsic Reward Calc"]
            BoxScore -- "box_score" --> Agent_Learn
        end

        %% Dialogue Generation
        subgraph "Dialogue Generation"
            Agent_Model -- "EmoModule Output" --> GPT["TransformerGPT (Dialogue)"]:::module
            GPT -- "Dialogue Response" --> Agent_Step_Out
        end

        %% Agent Outputs
        subgraph "Outputs"
            Agent_Step_Out["Agent Step Output"]:::agent --> Output_HM["Output: Head Movement Label"]:::output
            Agent_Step_Out --> Output_Dialogue["Output: Dialogue Response"]:::output
        end
    end

    %% Tooltips for Detailed Explanations
    click EmoModule "Processes emotions based on state and reward."
    click Lattice "Discretizes the state into metronic units."
    click Korporator "Aggregates raw beliefs using Syntrix logic."
    click Kaskade "Structures beliefs for downstream tasks."
    click Consistency "Computes reflexivity score (rho_score)."
    click Accessibility "Calculates accessibility metric (R_acc)."
    click BoxScore "Computes box score for intrinsic rewards."
    click GPT "Generates dialogue responses using TransformerGPT."
