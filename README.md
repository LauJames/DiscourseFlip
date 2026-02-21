# 🎯 DiscourseFlip: An Oblique Discourse-Level Opinion Manipulation Attack against Black-box RAG

## 🧠 Overview

This repository contains the full implementation of **DiscourseFlip**. Retrieval-Augmented Generation (RAG) systems are widely deployed and increasingly influential, but their reliance on external corpora exposes new security risks from poisoned retrieval content. Existing RAG attacks are largely focusing on individual queries or narrow topic-local query sets, which limits their practical reach and offers limited camouflage in real-world settings. 

To address this, we introduce **discourse-level opinion manipulation**, a new threat model in which coordinated influence across a semantic query network induces opinion shifts over a broad, multi-topic query space. We formalize this threat in a black-box setting and propose **DiscourseFlip**, an agentic, graph-guided attack that dynamically allocates a limited poisoning budget to maximize discourse-level opinion shift. 

### 📂 Repository Structure & Methodology

1. **`src/prep/` (Stage 1: Contextualized Query Network)** Extracts the contextualized query network to represent the broad, multi-topic query space.

2. **`src/graph_cluster/` (Stage 2: Hierarchical Attack Surface Organization)** Constructs the structured semantic graph. Then applies a 2-stage Leiden-KMeans clustering to organize queries.

3. **`src/attack/` (Stage 3: Graph-Guided Agentic Process Optimization)** The core DiscourseFlip implementation. Operating in a black-box setting, it uses an agentic approach to generate and optimize poisoned documents. 

4. **`src/eval/` (Stage 4: Evaluating Discourse-Level Opinion Shift)** Injects the optimized poisoned documents back into the RAG system to measure the attack's effectiveness.

## 🚀 Quick Start

### External API Configuration
The pipeline requires API access for both dataset construction (web search) and evaluation. Please export your API credentials:

```bash
# Required for Stage 1 (Data Construction via Jina AI Search Service)
export JINA_API_KEY="your_jinaai_key"

# Required for Stage 4 (Evaluation via LLM API)
export EVAL_API_KEY="your_api_key"
export EVAL_API_BASE="your_api_base_url"
```

### Deploying the Attack Generator (vLLM)
For the attack generation (Stage 3), we use Qwen/Qwen3-Next-80B-A3B-Instruct. For efficient inference, please deploy this model locally using vLLM on port 8000.

Open a separate terminal and start the vLLM server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000 \
    --dtype auto
```


### 🏃 Running the Pipeline

All execution scripts are located in the `scripts/` directory. The global configuration file (`scripts/topic.sh`) contains the dataset arrays covering **4 domains and 40 topics** (Politics, Sports, Entertainment, Society).

To reproduce the experiments, execute the following bash scripts sequentially from the root directory:

```bash
# Stage 1: Preprocess data and evaluate the clean baseline
bash scripts/run_prep.sh

# Stage 2: Hierarchical Attack Surface Organization
bash scripts/run_graph.sh

# Stage 3: Run the DiscourseFlip Agentic Attack
bash scripts/run_attack.sh

# Stage 4: Final Retrieval and Stance Evaluation
bash scripts/run_eval.sh

```
