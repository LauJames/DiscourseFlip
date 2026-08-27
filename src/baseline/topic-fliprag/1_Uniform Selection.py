import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.getenv("TOPIC_FLIPRAG_DATASET_ROOT", PROJECT_ROOT / "data")
).expanduser()
OUTPUT_ROOT = Path(
    os.getenv(
        "TOPIC_FLIPRAG_CLUSTER_OUTPUT_DIR",
        PROJECT_ROOT / "topic-fliprag" / "pipeline_data" / "clusters",
    )
).expanduser()
STANCE = os.getenv("TOPIC_FLIPRAG_STANCE", "oppose")
NUM_CLUSTERS = int(os.getenv("TOPIC_FLIPRAG_NUM_CLUSTERS", "5"))
RANDOM_SEED = int(os.getenv("TOPIC_FLIPRAG_RANDOM_SEED", "42"))


def iter_topic_tree_files(dataset_root: Path) -> Iterator[Dict[str, Any]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for json_path in sorted(
        dataset_root.rglob("topic_tree_*_with_summary_queries.json")
    ):
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Failed to load {json_path}: {exc}")
            continue

        relative_parts = json_path.relative_to(dataset_root).parts
        category = relative_parts[0] if len(relative_parts) > 1 else "default"
        topic = json_path.name.removeprefix("topic_tree_").removesuffix(
            "_with_summary_queries.json"
        )
        yield {
            "category": category,
            "topic": topic,
            "data": data,
        }


def normalize_items(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for item in data:
        queries = item.get("queries") or []
        query = next(
            (
                value.strip()
                for value in queries
                if isinstance(value, str) and value.strip()
            ),
            None,
        )
        if query is None:
            continue

        items.append(
            {
                "id": item.get("id"),
                "topic": item.get("wiki_title") or item.get("topic") or "",
                "query": query,
                "summary": item.get("summary", ""),
            }
        )
    return items


def distribute_randomly(
    items: List[Dict[str, Any]],
    num_clusters: int,
    rng: random.Random,
) -> List[List[Dict[str, Any]]]:
    if not items:
        return []

    cluster_count = min(max(1, num_clusters), len(items))
    shuffled_items = list(items)
    rng.shuffle(shuffled_items)

    clusters = [[] for _ in range(cluster_count)]
    for index, item in enumerate(shuffled_items):
        clusters[index % cluster_count].append(item)
    return clusters


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    for topic_data in iter_topic_tree_files(DATASET_ROOT):
        items = normalize_items(topic_data["data"])
        clusters = distribute_randomly(items, NUM_CLUSTERS, rng)
        if not clusters:
            continue

        output = [
            {
                "cluster_id": cluster_id,
                "cluster_method": "random_balanced",
                "topics": cluster_items,
                "region_nodes": [item["id"] for item in cluster_items],
            }
            for cluster_id, cluster_items in enumerate(clusters)
        ]

        topic = topic_data["topic"]
        output_dir = OUTPUT_ROOT / topic_data["category"] / topic
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{topic}_sim_cluster_{STANCE}.json"

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)

        sizes = [len(cluster) for cluster in clusters]
        print(f"Saved {output_path} with cluster sizes {sizes}")


if __name__ == "__main__":
    main()
