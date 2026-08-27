# Defense Evaluation

This folder provides two scripts for evaluating retrieval-time and generation-time defenses in a RAG pipeline.

## Retrieval

```bash
python defense/eva-retrival.py \
  --defense grada \
  --category <category> \
  --topic <topic> \
  --stance oppose \
  --budget 10
```

Available defenses:

- `none`
- `grada`
- `paraphrasing`
- `robust_masking`

Use `--embedding_model_path`, `--clean_index_path`, `--query_path`, and `--poison_path` to override the default paths.

## Generation

```bash
python defense/eva-generate.py \
  --defense target_topic_safeguard \
  --retrieval_defense grada \
  --category <category> \
  --topic <topic> \
  --model_name llama
```

Available defenses:

- `none`
- `target_topic_safeguard`
- `neutral_rewrite`
- `discourse_adaptive_defense`

The discourse-adaptive defense additionally requires a topic graph:

```bash
python defense/eva-generate.py \
  --defense discourse_adaptive_defense \
  --graph_path <topic_graph.json> \
  --defense_scope 40 \
  --input_path <retrieval_results.json>
```

Use `--model_path`, `--input_path`, and `--output_path` to provide custom locations.

Run either script with `--help` to view all options.
