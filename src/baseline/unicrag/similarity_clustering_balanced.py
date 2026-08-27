# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import random
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel



def mean_pooling(output, mask):
    tok_emb = output.last_hidden_state
    mask = mask.unsqueeze(-1).expand(tok_emb.size()).float()
    return (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def embed_documents(texts, model, tokenizer, device, batch_size=32, fp16=True):
    model.eval()
    # Normalize line breaks before embedding.
    texts = [t.replace("\n", " ") for t in texts]
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="🔹Embedding", ncols=100):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            if fp16:
                with torch.cuda.amp.autocast():
                    out = model(**inputs)
            else:
                out = model(**inputs)
            emb = mean_pooling(out, inputs['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0).numpy()

def balanced_similarity_clustering(embeddings, n_clusters=10):
    """
    Cluster embeddings with a balanced greedy assignment algorithm.
    """
    N = len(embeddings)
    # Reduce the cluster count when there are fewer samples than clusters.
    if N < n_clusters:
        n_clusters = max(1, N)
    
    k = N // n_clusters
    all_idx = list(range(N))
    
    # 1. Select random seeds.
    seeds = random.sample(all_idx, n_clusters)
    remaining = list(set(all_idx) - set(seeds))
    clusters = {i: [s] for i, s in enumerate(seeds)}
    
    # Calculate the full similarity matrix.
    sim_matrix = cosine_similarity(embeddings)
    
    # 2. Greedily fill every cluster to the baseline size k.
    for i in range(n_clusters):
        while len(clusters[i]) < k and remaining:
            # Measure each remaining node against the current cluster.
            avg_sims = [np.mean([sim_matrix[q][j] for j in clusters[i]]) for q in remaining]
            q_star = remaining[int(np.argmax(avg_sims))]
            clusters[i].append(q_star)
            remaining.remove(q_star)
            
    # 3. Assign remaining nodes to their most similar cluster.
    for q in remaining:
        best, best_sim = None, -1
        for i in range(n_clusters):
            sims = [sim_matrix[q][j] for j in clusters[i]]
            s = np.mean(sims)
            if s > best_sim:
                best, best_sim = i, s
        if best is not None:
            clusters[best].append(q)
            
    return clusters

def main():
    parser = argparse.ArgumentParser(description="2_cluster Pipeline")
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Full path to a topic_tree_*_with_summary_queries.json file",
    )
    parser.add_argument("--category", type=str, default=None, help="Target topic category")
    parser.add_argument("--topic", type=str, default=None, help="Target topic name")
    parser.add_argument("--base_dir", type=str, 
                        default=str(Path(__file__).resolve().parents[1]),
                        help="Project root directory")
    parser.add_argument("--model_path", type=str, 
                        default=os.getenv(
                            "TOPIC_FLIPRAG_BGE_MODEL_PATH",
                            "BAAI/bge-large-en-v1.5",
                        ),
                        help="Embedding model path")
    parser.add_argument("--num_cluster", type=int, default=5)
    parser.add_argument("--output-json", type=str, default=None, help="Optional explicit output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input_json:
        input_path = Path(args.input_json).expanduser()
        if not input_path.exists():
            print(f"❌ Error: Input file not found -> {input_path}")
            return

        topic_name = input_path.name.replace("topic_tree_", "").replace("_with_summary_queries.json", "")
        if args.output_json:
            output_path = Path(args.output_json).expanduser()
        else:
            output_dir = Path(args.base_dir) / "unicrag" / "output"
            output_path = output_dir / f"balanced_sim_cluster_{topic_name}.json"
    else:
        if not args.category or not args.topic:
            parser.error("Either --input-json or both --category and --topic must be provided.")

        topic_dir = Path(args.base_dir) / "topic_dataset" / args.category / args.topic
        input_path = topic_dir / f"topic_tree_{args.topic}_with_summary_queries.json"

        if args.output_json:
            output_path = Path(args.output_json).expanduser()
        else:
            output_dir = Path(args.base_dir) / "Baseline" / "Unic_RAG" / "data" / args.category
            output_path = output_dir / f"balanced_sim_cluster_{args.topic}.json"

    output_dir = output_path.parent


    print(f"========== Pipeline Step 2: Clustering ==========")
    print(f"Topic: {input_path.stem.replace('topic_tree_', '').replace('_with_summary_queries', '')}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    if not output_dir.exists():
        print(f"📁 Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # 3. Load the model.
    print("⏳ Loading Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device)

    # 4. Load the data.
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    valid_items = [item for item in data if item.get('queries')]
    if not valid_items:
        print("❌ No valid items with queries found.")
        return

    queries = [item['queries'][0] for item in valid_items]
    nodes = [item['wiki_title'] for item in valid_items]
    node_ids = [item['id'] for item in valid_items]
    summaries = [item['summary'] for item in valid_items]

    print(f"🔹 Processing {len(queries)} items...")

    # 5. Embedding
    embeddings = embed_documents(queries, model, tokenizer, device, batch_size=32, fp16=True)

    # 6. Clustering
    random.seed(args.seed)
    print("🔹 Running Balanced Similarity Clustering...")
    clusters = balanced_similarity_clustering(embeddings, n_clusters=args.num_cluster)

    # 7. Assemble the results.
    clusters_final = []
    for cid, idxs in clusters.items():
        topics_list = []
        for i in idxs:
            topics_list.append({
                "id": node_ids[i],        
                "topic": nodes[i],
                "query": queries[i],
                "summary": summaries[i],
            })
        
        clusters_final.append({
            "cluster_id": int(cid),
            "topics": topics_list,
            "region_nodes": [node_ids[i] for i in idxs],        
            })

    # 8. Save the results.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters_final, f, ensure_ascii=False, indent=2)

    print(f"✅ Done, saved → {output_path}")

if __name__ == "__main__":
    main()
