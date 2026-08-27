#!/usr/bin/env python3
"""Evaluate retrieval with an optional defense method."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

try:
    import networkx as nx
    import numpy as np
    import torch
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as dependency_error:
    DEPENDENCY_IMPORT_ERROR = dependency_error

    class Embeddings:
        """Placeholder that allows command-line help without optional dependencies."""

else:
    DEPENDENCY_IMPORT_ERROR = None


DEFAULT_EMBEDDING_MODELS = {
    "bge": "BAAI/bge-large-en-v1.5",
    "contriever": "facebook/contriever-msmarco",
    "contriever_nonorm": "facebook/contriever-msmarco",
    "dpr": "antoinelouis/dpr-xm",
    "qwen": "Qwen/Qwen3-Embedding-4B",
    "qwen06": "Qwen/Qwen3-Embedding-0.6B",
}


def clean_content(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in normalized.split("\n") if line.strip())


def load_poison_data(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.is_file():
        raise FileNotFoundError(f"Poison data not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    poisoned_documents: list[dict[str, Any]] = []
    seen_passages: set[str] = set()
    for item_index, item in enumerate(data):
        passages = item.get("attack_passages", item.get("attack_passage", []))
        if isinstance(passages, str):
            passages = [passages]
        for passage_index, raw_passage in enumerate(passages or []):
            passage = clean_content(raw_passage)
            if not passage or passage in seen_passages:
                continue
            seen_passages.add(passage)
            source_index = item.get("idx", item_index)
            poisoned_documents.append(
                {
                    "id": f"poison_{source_index}_{passage_index}",
                    "url": "POISONED_SOURCE",
                    "content_chunk": [passage],
                }
            )

    print(f"Loaded {len(poisoned_documents)} unique poisoned documents.")
    return poisoned_documents


class UniversalEmbeddingWrapper(Embeddings):
    """Expose supported Transformer encoders through the LangChain interface."""

    def __init__(self, model_path: str, device: str | None = None, batch_size: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model_name_lower = model_path.lower()
        self.batch_size = batch_size
        print(f"Loading embedding model {model_path} on {self.device}.")

        if "contriever" in self.model_name_lower:
            self.engine_type = "transformers"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
        else:
            self.engine_type = "sentence_transformers"
            self.model = SentenceTransformer(
                model_path,
                device=self.device,
                trust_remote_code=True,
            )
            if "dpr" in self.model_name_lower:
                transformer = self.model[0]
                auto_model = getattr(transformer, "auto_model", None)
                if hasattr(auto_model, "set_default_language"):
                    auto_model.set_default_language("en_XX")
            self.encode_kwargs = {
                "normalize_embeddings": True,
                "batch_size": self.batch_size,
            }

    @staticmethod
    def _mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)
        summed_mask = expanded_mask.sum(dim=1)
        return summed_embeddings / torch.clamp(summed_mask, min=1e-9)

    def _embed_contriever(self, texts: Sequence[str]) -> list[list[float]]:
        normalized_texts = [text.replace("\n", " ") for text in texts]
        all_embeddings: list[list[float]] = []
        for start in range(0, len(normalized_texts), self.batch_size):
            batch = normalized_texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**encoded)
            embeddings = self._mean_pooling(output, encoded["attention_mask"])
            all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.engine_type == "transformers":
            return self._embed_contriever(texts)
        return self.model.encode(texts, **self.encode_kwargs).tolist()

    def embed_query(self, text: str) -> list[float]:
        if "bge" in self.model_name_lower:
            text = f"Represent this sentence for searching relevant passages: {text}"
        if self.engine_type == "transformers":
            return self._embed_contriever([text])[0]
        if "qwen" in self.model_name_lower:
            return self.model.encode(
                [text], prompt_name="query", **self.encode_kwargs
            )[0].tolist()
        return self.model.encode([text], **self.encode_kwargs)[0].tolist()


class QueryParaphraser:
    def __init__(self, model_path: str):
        print(f"Loading paraphrasing model {model_path}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def paraphrase(self, query: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": "You are a precise language assistant."},
            {
                "role": "user",
                "content": (
                    "Rewrite the question using different wording while strictly preserving "
                    "its meaning, intent, and stance. Return only the rewritten question.\n\n"
                    f"Question:\n{query}"
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output_ids[0, inputs.input_ids.shape[1] :]
        paraphrase = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return paraphrase or query


class GradaReranker:
    def __init__(self, model_path: str, device: str, alpha: float):
        print(f"Loading GRADA model {model_path} on {device}.")
        self.model = SentenceTransformer(
            model_path,
            device=device,
            trust_remote_code=True,
        )
        self.alpha = alpha

    def rerank(self, contents: list[str], query: str) -> list[str]:
        if len(contents) <= 1:
            return contents
        document_embeddings = self.model.encode(
            contents,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_embedding = self.model.encode(
            [f"Represent this sentence for searching relevant passages: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        query_similarities = document_embeddings @ query_embedding
        document_similarities = document_embeddings @ document_embeddings.T
        similarity_matrix = np.maximum(
            document_similarities
            - self.alpha
            * (query_similarities[:, None] + query_similarities[None, :]),
            0.0,
        )
        np.fill_diagonal(similarity_matrix, 0.0)
        scores = nx.pagerank(nx.from_numpy_array(similarity_matrix))
        ranked_indices = sorted(range(len(contents)), key=scores.get, reverse=True)
        return [contents[index] for index in ranked_indices]


def inject_poison(
    clean_index_path: Path,
    poisoned_documents: list[dict[str, Any]],
    embedder: UniversalEmbeddingWrapper,
    save_path: Path | None,
) -> FAISS:
    if not clean_index_path.is_dir():
        raise FileNotFoundError(f"Clean FAISS index not found: {clean_index_path}")
    vectorstore = FAISS.load_local(
        str(clean_index_path),
        embedder,
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    texts: list[str] = []
    metadata: list[dict[str, str]] = []
    for record in poisoned_documents:
        for chunk in record["content_chunk"]:
            texts.append(chunk)
            metadata.append(
                {
                    "source_url": "POISONED_SOURCE",
                    "doc_id": record.get("id", "unknown"),
                }
            )
    if texts:
        vectorstore.add_texts(texts, metadatas=metadata)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_path))
    return vectorstore


def random_mask_text(
    text: str,
    mask_rate: float,
    mask_token: str,
    rng: random.Random,
    min_keep: int = 2,
) -> str:
    tokens = text.split()
    if len(tokens) <= min_keep or mask_rate == 0:
        return text
    mask_count = min(max(1, int(len(tokens) * mask_rate)), len(tokens) - min_keep)
    masked = tokens.copy()
    for index in rng.sample(range(len(tokens)), mask_count):
        masked[index] = mask_token
    return " ".join(masked)


def embed_documents_batched(
    embedder: UniversalEmbeddingWrapper,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(texts), batch_size):
        vectors = embedder.embed_documents(texts[start : start + batch_size])
        batches.append(np.asarray(vectors, dtype=np.float32))
    return np.vstack(batches)


def robust_mask_rerank(
    query: str,
    documents: list[Any],
    embedder: UniversalEmbeddingWrapper,
    mask_rate: float,
    mask_samples: int,
    mask_token: str,
    embedding_batch_size: int,
    rng: random.Random,
) -> list[Any]:
    if not documents:
        return documents
    query_vector = np.asarray(embedder.embed_query(query), dtype=np.float32)
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)

    masked_texts: list[str] = []
    document_offsets: list[tuple[int, int]] = []
    for document in documents:
        start = len(masked_texts)
        for _ in range(mask_samples):
            masked_texts.append(
                random_mask_text(
                    document.page_content,
                    mask_rate,
                    mask_token,
                    rng,
                )
            )
        document_offsets.append((start, len(masked_texts)))

    document_vectors = embed_documents_batched(
        embedder, masked_texts, embedding_batch_size
    )
    norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
    document_vectors /= np.maximum(norms, 1e-12)
    similarities = document_vectors @ query_vector
    scores = [float(similarities[start:end].mean()) for start, end in document_offsets]
    ranked_indices = np.argsort(scores)[::-1]
    return [documents[index] for index in ranked_indices]


def print_diagnosis(
    query: str,
    documents: list[Any],
    poison_vectors: np.ndarray | None,
    embedder: UniversalEmbeddingWrapper,
) -> None:
    subset = documents[:10]
    if not subset:
        return
    query_vector = np.asarray(embedder.embed_query(query)).reshape(1, -1)
    document_vectors = np.asarray(
        embedder.embed_documents([document.page_content for document in subset])
    )
    result_scores = cosine_similarity(query_vector, document_vectors).flatten()
    best_poison_score = -1.0
    if poison_vectors is not None and len(poison_vectors):
        best_poison_score = float(
            cosine_similarity(query_vector, poison_vectors).flatten().max()
        )
    print(f"\nDiagnosis for query: {query}")
    print(
        f"Best poison score: {best_poison_score:.4f}; "
        f"current top score: {result_scores[0]:.4f}"
    )
    for rank, (document, score) in enumerate(zip(subset, result_scores), start=1):
        source = document.metadata.get("source_url", "unknown")
        snippet = document.page_content.strip().replace("\n", " ")[:80]
        print(f"{rank:>2}. {score:.4f} | {source} | {snippet}")


def retrieve(
    nodes: list[dict[str, Any]],
    vectorstore: FAISS,
    poisoned_documents: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    embedder = vectorstore.embeddings
    poison_texts = [record["content_chunk"][0] for record in poisoned_documents]
    poison_vectors = (
        np.asarray(embedder.embed_documents(poison_texts)) if poison_texts else None
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": args.top_k}
    )

    paraphraser = None
    grada_reranker = None
    mask_token = "[MASK]"
    if args.defense == "paraphrasing":
        paraphraser = QueryParaphraser(args.paraphrase_model_path)
    elif args.defense == "grada":
        grada_reranker = GradaReranker(
            args.grada_model_path, embedder.device, args.grada_alpha
        )
    elif args.defense == "robust_masking":
        tokenizer_path = args.mask_tokenizer_path or args.embedding_model_path
        mask_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        mask_token = mask_tokenizer.mask_token or "[MASK]"

    rng = random.Random(args.seed)
    outputs: list[dict[str, Any]] = []
    diagnosed = 0
    for node in tqdm(nodes, desc="Retrieving", ncols=100):
        queries = node.get("queries") or []
        if not queries:
            continue
        query = queries[0]
        retrieval_query = query
        try:
            if paraphraser is not None:
                retrieval_query = paraphraser.paraphrase(
                    query, args.paraphrase_max_new_tokens
                )
            documents = retriever.invoke(retrieval_query)
            if grada_reranker is not None:
                top_documents = documents[: args.rerank_top_k]
                ranked_texts = grada_reranker.rerank(
                    [document.page_content for document in top_documents], query
                )
            elif args.defense == "robust_masking":
                documents = robust_mask_rerank(
                    query,
                    documents,
                    embedder,
                    args.mask_rate,
                    args.mask_samples,
                    mask_token,
                    args.mask_embedding_batch_size,
                    rng,
                )
                ranked_texts = [document.page_content.strip() for document in documents]
            else:
                ranked_texts = [document.page_content.strip() for document in documents]

            if diagnosed < args.diagnosis_limit:
                print_diagnosis(retrieval_query, documents, poison_vectors, embedder)
                diagnosed += 1
            outputs.append(
                {
                    "id": node.get("id"),
                    "topic": node.get("wiki_title"),
                    "query": query,
                    "top_k_docs": ranked_texts,
                    "summary": node.get("summary"),
                }
            )
        except Exception as error:
            print(f"Retrieval failed for node {node.get('id')}: {error}")
    return outputs


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    topic_directory = (
        Path(args.topic_dir).expanduser()
        if args.topic_dir
        else Path(args.base_dir).expanduser()
        / "final_final_dataset"
        / args.category
        / args.topic
    )
    clean_index = (
        Path(args.clean_index_path).expanduser()
        if args.clean_index_path
        else topic_directory / f"faiss_index_{args.embedding_prefix}"
    )
    query_path = (
        Path(args.query_path).expanduser()
        if args.query_path
        else topic_directory / f"topic_tree_{args.topic}_with_summary_queries.json"
    )
    poison_path = (
        Path(args.poison_path).expanduser()
        if args.poison_path
        else topic_directory
        / f"{args.topic}_optimized_seo_docs_{args.stance}_bert_{args.budget}.json"
    )
    output_path = (
        Path(args.output_path).expanduser()
        if args.output_path
        else topic_directory
        / (
            f"{args.topic}_poisoned_top{args.top_k}_results_{args.stance}_"
            f"{args.embedding_prefix}_budget{args.budget}_{args.defense}.json"
        )
    )
    return clean_index, query_path, poison_path, output_path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--defense",
        choices=("none", "grada", "paraphrasing", "robust_masking"),
        default="none",
        help="Retrieval defense to apply.",
    )
    parser.add_argument("--category", default="technology_business")
    parser.add_argument("--topic", default="social_media")
    parser.add_argument("--stance", default="oppose")
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--rerank_top_k", type=int, default=10)
    parser.add_argument("--embedding_prefix", default="contriever_NoNorm")
    parser.add_argument("--embedding_model_path")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--base_dir", default=str(project_root))
    parser.add_argument("--topic_dir")
    parser.add_argument("--clean_index_path")
    parser.add_argument("--query_path")
    parser.add_argument("--poison_path")
    parser.add_argument("--output_path")
    parser.add_argument("--save_poisoned_index")
    parser.add_argument("--diagnosis_limit", type=int, default=10)
    parser.add_argument("--grada_model_path", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--grada_alpha", type=float, default=0.4)
    parser.add_argument(
        "--paraphrase_model_path", default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument("--paraphrase_max_new_tokens", type=int, default=64)
    parser.add_argument("--mask_rate", type=float, default=0.2)
    parser.add_argument("--mask_samples", type=int, default=3)
    parser.add_argument("--mask_tokenizer_path")
    parser.add_argument("--mask_embedding_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.mask_rate < 1.0:
        parser.error("--mask_rate must be in the range [0, 1).")
    if args.mask_samples < 1:
        parser.error("--mask_samples must be positive.")
    if args.top_k < 1 or args.rerank_top_k < 1:
        parser.error("--top_k and --rerank_top_k must be positive.")
    args.embedding_model_path = args.embedding_model_path or DEFAULT_EMBEDDING_MODELS.get(
        args.embedding_prefix.lower()
    )
    if not args.embedding_model_path:
        parser.error(
            "Unknown --embedding_prefix; provide --embedding_model_path explicitly."
        )
    return args


def main() -> None:
    args = parse_args()
    if DEPENDENCY_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Retrieval dependencies are not installed. Install PyTorch, Transformers, "
            "Sentence Transformers, LangChain Community, FAISS, scikit-learn, NetworkX, "
            "NumPy, and tqdm before running this script."
        ) from DEPENDENCY_IMPORT_ERROR
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    clean_index, query_path, poison_path, output_path = resolve_paths(args)

    print(f"Defense: {args.defense}")
    print(f"Clean index: {clean_index}")
    print(f"Queries: {query_path}")
    print(f"Poison data: {poison_path}")
    print(f"Output: {output_path}")

    if not query_path.is_file():
        raise FileNotFoundError(f"Query data not found: {query_path}")
    with query_path.open("r", encoding="utf-8") as handle:
        query_nodes = json.load(handle)
    poisoned_documents = load_poison_data(poison_path)
    if not poisoned_documents:
        raise ValueError("The poison data contains no usable passages.")

    embedder = UniversalEmbeddingWrapper(
        args.embedding_model_path,
        device=args.device,
        batch_size=args.embedding_batch_size,
    )
    vectorstore = inject_poison(
        clean_index,
        poisoned_documents,
        embedder,
        Path(args.save_poisoned_index).expanduser()
        if args.save_poisoned_index
        else None,
    )
    results = retrieve(query_nodes, vectorstore, poisoned_documents, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} retrieval results to {output_path}.")


if __name__ == "__main__":
    main()
