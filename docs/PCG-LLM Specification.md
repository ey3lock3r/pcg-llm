# PCG-LLM SOTA Edition

## 1. System Overview

The Predictive Coding Graph Language Model (PCG-LLM) is a non-hierarchical, dynamically routed neural architecture. It merges the parallel compute efficiency of modern Transformers with the sample-efficient, error-driven reasoning capabilities of mammalian predictive coding. It is designed to match standard LLMs in linguistic fluency while vastly outperforming them in structured reasoning, logic, and hallucination reduction.

## 2. Topological Architecture

- **Substrate:** A universally connected Predictive Coding Graph (PCG) substituting rigid layers.

- **Nodes:** Each node $h_i$ acts as a latent state estimator and a local error detector.

- **Edges:** Connections are dynamically gated via Block-Sparsity. The adjacency matrix $A_{evolved}$ is computed block-wise to align with GPU SRAM hardware constraints.


## 3. Training & Learning Dynamics

- **Mechanism:** Global Backpropagation is entirely replaced by **Inference Learning (IL)**.

- **Objective:** Minimization of system-wide Free Energy ($F$), balancing predictive accuracy with topological sparsity.

- **Plasticity Rules:**

    - _State Weights:_ Updated via local prediction errors ($e_i$).

    - _Routing Weights ($W_{structure}$):_ Updated via Error-Driven Hebbian Plasticity, creating a "use-it-or-lose-it" evolutionary pressure on the graph's structure.


## 4. Inference & Generation (Two-System Approach)

The model utilizes a dual-pathway execution strategy for text generation, ensuring SOTA-level latency metrics.

### System 1: Fast Heuristic Pass (Testing Equivalence)

- **Trigger:** Standard text generation, summarization, conversational phrasing.

- **Execution:** The iterative error-minimization loops are mathematically bypassed. The model executes a single Block-Sparse Matrix Multiplication forward pass.

- **Decoding:** Energy-Based Parallel Decoding drafts multi-token sequences simultaneously.


### System 2: Deep Reasoning Iteration

- **Trigger:** Spikes in localized Free Energy (indicating logical conflicts, complex math, or structured reasoning requirements).

- **Execution:** Fast-pass is suspended. The model engages asynchronous, iterative message passing within the specific sub-graph experiencing the conflict. The graph dynamically rewires its attention to resolve the contradiction before token projection.


## 5. Hardware & Scaling Strategy

- **Sparsity Engine:** Custom OpenAI Triton kernels optimize the Block-Sparse operations, allowing the model to bypass the quadratic memory bottleneck of standard global attention.

- **Context Window:** Because learning and inference are localized (nodes only compute based on immediate neighbors, not the global sequence length), the theoretical context window scales linearly ($O(N)$) rather than quadratically ($O(N^2)$), allowing for infinite-context deployment on standard enterprise hardware.
