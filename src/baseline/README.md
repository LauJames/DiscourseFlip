# Baseline Methods

We compare **DiscourseFlip** with three representative RAG poisoning baselines: **PoisonedRAG**, **Topic-FlipRAG**, and **UniC-RAG**. For a fair comparison, all methods use the same poisoning budget of $M$ documents and, whenever a surrogate retriever is required, the same proxy retriever as DiscourseFlip. Each poisoned passage is limited to fewer than 500 tokens.

## PoisonedRAG

**PoisonedRAG** is a targeted corpus-poisoning attack applicable to both white-box and black-box RAG settings. We follow its **black-box variant**, which does not require access to the victim retriever's parameters or internal retrieval scores.

For each attack target, we generate $M$ poisoned documents based on:

- the **root topic**;
- the desired **target stance** (supporting or opposing the target); and
- the constraint that each passage contains fewer than 500 tokens.

Each passage presents content consistent with the target stance. To strengthen its association with the target topic during retrieval, we prepend the root-topic query to the generated passage before injecting it into the corpus. Thus, PoisonedRAG directly optimizes the relevance of each poisoned document to the root-topic query, without explicitly modeling the topic's broader discourse structure.

## Topic-FlipRAG

**Topic-FlipRAG** constructs a set of topic-related queries and optimizes poisoned documents to influence responses to those queries. Its black-box setting uses a surrogate retriever to estimate whether the generated documents will be retrieved by the victim RAG system.

To adapt Topic-FlipRAG to our discourse-based setting, we randomly partition the discourse-node set $N(A)$ into $M$ mutually disjoint subsets:

$$
N(A) = N_1 \cup N_2 \cup \cdots \cup N_M,
\qquad
N_i \cap N_j = \varnothing \quad \text{for } i \neq j.
$$

Each subset $N_i$ is treated as the target query set for one poisoned document. We then generate a passage of fewer than 500 tokens conditioned on:

- the nodes contained in $N_i$;
- the root topic; and
- the desired target stance.

The resulting passage is optimized to be relevant to the corresponding node subset while conveying content that supports or opposes the target stance. This process produces $M$ poisoned documents in total. We use the same surrogate retriever as DiscourseFlip so that performance differences are not caused by different proxy retrieval models.

## UniC-RAG

**UniC-RAG** is a multi-query attack originally developed under white-box access to the retriever. It uses **prompt-injection content**, following the attack construction described in the original paper, to manipulate the downstream generator when poisoned documents are retrieved.
