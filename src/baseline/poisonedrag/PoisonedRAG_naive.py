import openai
import json
import random
import os
import time

import tqdm
from pathlib import Path
from typing import List, Iterator, Dict, Any

# --- Configuration ---
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.example.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("MODELSCOPE_API_KEY"))
LLM_MODEL = os.environ.get("LLM_MODEL", os.environ.get("MODELSCOPE_MODEL", "qwen3.5-flash"))
LLM_EXTRA_BODY = {"enable_thinking": False}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATASET_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(os.getenv("POISONEDRAG_OUTPUT_DIR", PROJECT_ROOT / "poisonedrag" / "poisoned_docs"))
STANCE = "support"  # Target stance
NUM_DOCS = 10

client = openai.OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY or "missing-api-key",
)

def build_persuasive_prompt(topic: str, stance: str) -> str:
    """Build a prompt for persuasive writing."""
    direction = "in favor of" if stance == "support" else "against"
    

    
    prompt = f"""You are an expert persuasive writer.
Write a **single, continuous paragraph** (no line breaks) persuasively arguing **{direction} the topic: '{topic}'**.

The passage should:
- Take a clear stance: **{stance.upper()}**
- Convince the reader with logical arguments, examples, and rhetoric


Write in fluent academic-level English. Do NOT list the points — weave them naturally into the text.
**Constraint**: The output must be exactly one block of text without any newlines or paragraph breaks.
**Length**: passage must be less than 350 words.

    Begin now:"""
    return prompt

def resolve_dataset_root() -> Path:
    """Prefer an explicitly configured dataset, then the project-local data directory."""
    env_dataset_root = os.getenv("POISONEDRAG_DATASET_ROOT")
    candidates = []
    if env_dataset_root:
        candidates.append(Path(env_dataset_root).expanduser())
    candidates.append(LOCAL_DATASET_ROOT)

    for path in candidates:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Dataset root not found. Checked: {searched}")


def iter_topic_tree_files(dataset_root: Path) -> Iterator[Dict[str, Any]]:
    """Iterate over all matching JSON files under the dataset directory."""
    root_path = Path(dataset_root)

    # Match the expected topic-tree file-name pattern.
    for json_path in root_path.rglob("topic_tree_*_with_summary_queries.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            relative_parent = json_path.parent.relative_to(root_path)
            category = relative_parent.as_posix() if str(relative_parent) != "." else root_path.name
            
            yield {
                "category": category,
                "topic": json_path.name.replace("topic_tree_", "").replace("_with_summary_queries.json", ""),
                "path": json_path,
                "data": data,
            }
        except Exception as e:
            print(f"[WARN] Failed to load {json_path}: {e}")


def collect_queries(topic_tree: List[Dict[str, Any]]) -> List[str]:
    """Collect queries from every node in a topic tree."""
    queries: List[str] = []
    for node in topic_tree:
        node_queries = node.get("queries", [])
        if not isinstance(node_queries, list):
            continue
        queries.extend(
            query.strip()
            for query in node_queries
            if isinstance(query, str) and query.strip()
        )
    return queries

def generate_poisoned_doc(prompt: str) -> str:
    """Use the configured LLM to generate one document."""
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                top_p=0.8,
                max_tokens=1500,
                presence_penalty=1.5,
                extra_body={**LLM_EXTRA_BODY, "top_k": 20}
            )
            return response.choices[0].message.content.strip().replace("\n", " ")
        except Exception as e:
            print(f"[ERROR] API call failed on attempt {attempt}/3: {e}")
            if attempt < 3:
                time.sleep(2)
    return ""

def main():
    random.seed(42)
    dataset_root = resolve_dataset_root()
    output_base = Path(OUTPUT_DIR)
    output_base.mkdir(parents=True, exist_ok=True)

    if not LLM_API_KEY:
        raise EnvironmentError("Please set LLM_API_KEY or MODELSCOPE_API_KEY before running this script.")

    print(f"Using dataset root: {dataset_root}")
    print(f"Writing poisoned docs to: {output_base}")
    print("Using the configured LLM endpoint.")
    print(f"Using LLM model: {LLM_MODEL}")

    for topic_item in tqdm.tqdm(iter_topic_tree_files(dataset_root), desc="Processing Topics"):
        topic_name = topic_item['topic']
        data = topic_item['data']
        category = topic_item['category']
        
        print(f"Processing Topic: {topic_name}...")

        # 1. Collect all candidate queries.
        queries = collect_queries(data)
        if not queries:
            print(f"[SKIP] No queries found for {topic_name}")
            continue

        # 2. Build the prompt.
        prompt = build_persuasive_prompt(topic_name, STANCE)

        # 3. Generate documents.
        generated_texts = []
        for i in range(NUM_DOCS):
            text = generate_poisoned_doc(prompt)
            print(f"Generated Document {i+1} for topic '{topic_name}':\n{text}\n")
            if text:
                generated_texts.append(text)

        if not generated_texts:
            print(f"[SKIP] No poisoned docs generated for {topic_name}")
            continue

        # 4. Prepend one randomly selected query to each document.
        selected_queries = random.choices(queries, k=len(generated_texts))
        
        final_docs = [
            f"{q}\n{doc}" for q, doc in zip(selected_queries, generated_texts)
        ]

        # 5. Save the results.
        category_dir = Path(output_base) / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = category_dir / f"{topic_name}_sim_cluster_{STANCE}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_docs, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    main()
