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

## Baseline Methods

We compare **DiscourseFlip** with three representative RAG poisoning baselines: **PoisonedRAG**, **Topic-FlipRAG**, and **UniC-RAG**. For a fair comparison, all methods use the same poisoning budget of \(M\) documents and, whenever a surrogate retriever is required, the same proxy retriever as DiscourseFlip. Each poisoned passage is limited to fewer than 500 tokens.

### PoisonedRAG

**PoisonedRAG** is a targeted corpus-poisoning attack applicable to both white-box and black-box RAG settings. We follow its **black-box variant**, which does not require access to the victim retriever’s parameters or internal retrieval scores.

For each attack target, we generate \(M\) poisoned documents based on:

* the **root topic**;
* the desired **target stance** (supporting or opposing the target); and
* the constraint that each passage contains fewer than 500 tokens.

Each passage presents content consistent with the target stance. To strengthen its association with the target topic during retrieval, we prepend the root-topic query to the generated passage before injecting it into the corpus. Thus, PoisonedRAG directly optimizes the relevance of each poisoned document to the root-topic query, without explicitly modeling the topic’s broader discourse structure.

### Topic-FlipRAG

**Topic-FlipRAG** constructs a set of topic-related queries and optimizes poisoned documents to influence responses to those queries. Its black-box setting uses a surrogate retriever to estimate whether the generated documents will be retrieved by the victim RAG system.

To adapt Topic-FlipRAG to our discourse-based setting, we randomly partition the discourse-node set \(N(A)\) into \(M\) mutually disjoint subsets:

$$
N(A) = N_1 \cup N_2 \cup \cdots \cup N_M,
\qquad
N_i \cap N_j = \varnothing \ \text{for } i \neq j.
$$

Each subset \(N_i\) is treated as the target query set for one poisoned document. We then generate a passage of fewer than 500 tokens conditioned on:

* the nodes contained in \(N_i\);
* the root topic; and
* the desired target stance.

The resulting passage is optimized to be relevant to the corresponding node subset while conveying content that supports or opposes the target stance. This process produces \(M\) poisoned documents in total. We use the same surrogate retriever as DiscourseFlip so that performance differences are not caused by different proxy retrieval models.


### UniC-RAG

**UniC-RAG** is a multi-query attack originally developed under white-box access to the retriever. It uses **prompt-injection content**, following the attack construction described in the original paper, to manipulate the downstream generator when poisoned documents are retrieved.

Because our evaluation assumes black-box access to the victim RAG system, we adapt UniC-RAG by replacing its original white-box retriever with the same proxy retriever used by DiscourseFlip and the other black-box baselines. The remaining attack procedure, including its original prompt-injection strategy, is kept unchanged.



