import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  
import nltk
import torch
from nltk.corpus import stopwords
from transformers import BertTokenizer, GPT2Tokenizer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path(
    os.getenv("TOPIC_FLIPRAG_PIPELINE_ROOT", PROJECT_ROOT / "topic-fliprag" / "pipeline_data")
)

COMMON_WORDS = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it']
DEFAULT_STOPWORDS = set(COMMON_WORDS + ['for', 'on', 'with', 'as', 'by', 'an', 'be'])


def load_stopwords():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
            return set(stopwords.words('english'))
        except Exception:
            return DEFAULT_STOPWORDS


STOPWORDS = load_stopwords()


def get_inputs_filter_ids(inputs, tokenizer):
    tokens = [w for w in tokenizer.tokenize(inputs) if w.isalpha() and w not in STOPWORDS]
    return tokenizer.convert_tokens_to_ids(tokens)


def get_sub_masks(tokenizer, device, prob=False):
    # masking for all subwords in the vocabulary
    vocab = tokenizer.get_vocab()

    def is_special_token(w):
        if isinstance(tokenizer, BertTokenizer) and w.startswith('##'):
            return True
        if isinstance(tokenizer, GPT2Tokenizer) and not w.startswith('Ġ'):
            return True
        if w[0] == '[' and w[-1] == ']':
            return True
        if w[0] == '<' and w[-1] == '>':
            return True
        if w in ['=', '@', 'Ġ=', 'Ġ@'] and w in vocab:
            return True
        return False

    filter_ids = [vocab[w] for w in vocab if is_special_token(w)]
    if prob:
        prob_mask = torch.ones(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = 0.
    else:
        prob_mask = torch.zeros(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = -1e9
    return prob_mask


def get_poly_sub_masks(tokenizer, device, prob=False):
    filter_ids = [tokenizer.dict[w] for w in tokenizer.dict.tok2ind
                  if not w.isalnum()]
    if prob:
        prob_mask = torch.ones(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = 0.
    else:
        prob_mask = torch.zeros(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = -1e9
    return prob_mask


def create_constraints(seq_len, tokenizer, device, prob=False):
    stopword_ids = [tokenizer.vocab[w] for w in COMMON_WORDS[:5] if w in tokenizer.vocab]
    if prob:
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device)
    else:
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device) - 1e9

    for t in range(seq_len):
        if t >= seq_len // 2:
            masks[t, stopword_ids] = 1.0 if prob else 0.0
        else:
            masks[t] = 1.0 if prob else 0.
    return masks


def create_poly_constraints(seq_len, tokenizer, device, prob=False):
    stopword_ids = [tokenizer.dict[w] for w in COMMON_WORDS[:5] if w in tokenizer.dict.tok2ind]
    if prob:
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device)
    else:
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device) - 1e9

    for t in range(seq_len):
        if t >= seq_len // 3:
            masks[t, stopword_ids] = 1.0 if prob else 0.0
        else:
            masks[t] = 1.0 if prob else 0.
    return masks





import argparse
import os
import json
import sys
from collections import defaultdict
import json
from torch.nn.functional import cosine_similarity
import torch
import tqdm
from transformers import  BertTokenizerFast,  AutoModelForSequenceClassification
from torch import cuda
from tqdm import tqdm



BIRCH_ALPHAS = [1.0, 0.5, 0.1]
BIRCH_GAMMA = 0.6
BOS_TOKEN = '[unused0]'

device = 'cuda' if cuda.is_available() else 'cpu'
device_cpu = torch.device("cpu")
NBERT_MODEL_PATH = os.getenv(
    "TOPIC_FLIPRAG_NBERT_MODEL_PATH",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
)
STEP1_INPUT_FILE = os.getenv("TOPIC_FLIPRAG_STEP1_INPUT_FILE")
STEP2_INPUT_DIR = os.getenv("TOPIC_FLIPRAG_STEP2_INPUT_DIR")
OVERWRITE = os.getenv("TOPIC_FLIPRAG_OVERWRITE", "0") == "1"


def strip_known_suffix(file_name: str, stance: str) -> str:
    suffixes = [
        f"_Step_1_{stance}.json",
        f"_random_docs_{stance}.json",
        f"_sim_docs_{stance}.json",
    ]
    for suffix in suffixes:
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return Path(file_name).stem


def infer_stance_from_file_name(file_name: str):
    for stance in ("support", "oppose"):
        suffixes = [
            f"_Step_1_{stance}.json",
            f"_random_docs_{stance}.json",
            f"_sim_docs_{stance}.json",
        ]
        if any(file_name.endswith(suffix) for suffix in suffixes):
            return stance
    return None


def is_stage1_cluster_file(file_name: str, stance: str) -> bool:
    return file_name.endswith(f"_random_docs_{stance}.json") or file_name.endswith(f"_sim_docs_{stance}.json")


def extract_topic_from_file_name(file_name: str, stance: str) -> str:
    if file_name.endswith(f"_Step_1_{stance}.json"):
        return file_name[: -len(f"_Step_1_{stance}.json")]
    return strip_known_suffix(file_name, stance)


def normalize_stage1_item(item: dict) -> dict:
    generated_docs = item.get("generated_docs", [])
    attack_passage = ""
    if isinstance(generated_docs, list) and generated_docs:
        first_doc = generated_docs[0]
        if isinstance(first_doc, str):
            attack_passage = first_doc.strip()

    queries = item.get("queries", [])
    query_list = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    return {
        "cluster_id": item.get("cluster_id"),
        "cluster_method": item.get("cluster_method"),
        "root_topic": item.get("root_topic"),
        "root_summary": item.get("root_summary"),
        "cluster_size": item.get("cluster_size"),
        "region_nodes": item.get("region_nodes", []),
        "source_passage": item.get("source_passage"),
        "attack_passage": attack_passage,
        "query_list": query_list,
        "queries": query_list,
        "topics": item.get("topics", []),
    }


def load_attack_items(data_path: Path, stance: str):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {data_path}, got {type(data).__name__}")

    if data_path.name.endswith(f"_Step_1_{stance}.json"):
        return data

    return [normalize_stage1_item(item) for item in data]

def main():
    parser = argparse.ArgumentParser('Collision_Attack')

    parser.add_argument('--mode', default='test', type=str,
                        help='train/test')

    # target known model config
    parser.add_argument("--experiment_name", default='collision.pointwise', type=str)
    parser.add_argument("--target", type=str, default='nb_bert', help='test on what model')
    parser.add_argument("--target_type", type=str, default='none', help='target model of what kind of trigger')

    parser.add_argument("--data_name", default="dl", type=str)
    parser.add_argument("--method", default="nature", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--model_path",
        default=os.getenv("TOPIC_FLIPRAG_ATTACK_MODEL_PATH", "models/nbbert_embedding_adv.pt"),
        type=str,
    )
    parser.add_argument("--transformer_model", default="cross-encoder/ms-marco-MiniLM-L-12-v2", type=str, required=False, help="Bert model to use (cross-encoder/ms-marco-MiniLM-L-12-v2,bert-base-uncased).")
    parser.add_argument('--stemp', type=float, default=1.0, help='temperature of softmax')
    parser.add_argument('--lr', type=float, default=0.005, help='optimization step size')
    parser.add_argument('--max_iter', type=int, default=5, help='maximum iteraiton')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--min_len', type=int, default=5, help='Min sequence length')
    parser.add_argument("--beta", default=0., type=float, help="Coefficient for language model loss.")
    parser.add_argument("--amount", default=0, type=int, help="adv_Data amount.")
    parser.add_argument('--save', action='store_true', help='Save collision to file')
    parser.add_argument('--verbose', action='store_true', default=True,  help='Print every iteration')
    parser.add_argument("--lm_model_dir", type=str, help="Path to pre-trained language model")
    parser.add_argument('--perturb_iter', type=int, default=50, help='PPLM iteration')
    parser.add_argument("--kl_scale", default=0.0, type=float, help="KL divergence coefficient")
    parser.add_argument("--topk", default=10, type=int, help="Top k sampling for beam search")
    parser.add_argument("--num_beams", default=3, type=int, help="Number of beams")
    parser.add_argument("--num_filters", default=100, type=int, help="Number of num_filters words to be filtered")
    parser.add_argument('--nature', action='store_true', help='Nature collision')
    parser.add_argument('--pat', action='store_true', help='PAT.')
    parser.add_argument('--regularize', action='store_true', help='Use regularize to decrease perplexity')
    parser.add_argument('--fp16', default=True, action='store_true', help='fp16')
    parser.add_argument('--patience_limit', type=int, default=3, help="Patience for early stopping.")
    parser.add_argument("--seed", default=42, type=str, help="random seed")
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Only process one JSON file. Accepts either *_Step_1_*.json or "
             "the upstream *_random_docs_*.json / *_sim_docs_*.json file.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Process every *_random_docs_*.json / *_sim_docs_*.json file in a directory.",
    )
    
    # nature: True True beta=0.015 stemp=0.02, num_beams=10, topk=150, max_iter=5
    # python run_collision.py --nature --beta=0.02 --stemp=0.1 --num_beams=1 --topk=50 --max_iter=5 --mode=train
    # constrains: True; False beta=0.85 stemp=1.0, num_beams=5, topk=40, max_iter=30
    # python run_collision.py --regularize --beta=0.85 --stemp=1.0 --num_beams=5 --topk=40  --max_iter=30 --mode=train
    # aggressive: False; False  beta=0 stemp=1.0, num_beams=5, topk=50, max_iter=30
    # python ASC_topic_nbbert_passage_opinion.py --beta=0.0 --stemp=1.0 --num_beams=3 --topk=10 --max_iter=30 --mode=train
#python ASC_topic_73.py --beta=0.0 --stemp=1.0 --num_beams=5 --topk=50 --max_iter=2 --mode=train
    args = parser.parse_args()


    tokenizer = BertTokenizerFast.from_pretrained(NBERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(NBERT_MODEL_PATH)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False




    # --- Path configuration ---
    INPUT_ROOT = PIPELINE_ROOT / "Step_1_result"
    OUTPUT_ROOT = PIPELINE_ROOT / "final_result"

    DEFAULT_STANCE = os.getenv("TOPIC_FLIPRAG_STANCE", "oppose")
    SELECTED_INPUT_DIR = Path(args.input_dir or STEP2_INPUT_DIR).expanduser() if (args.input_dir or STEP2_INPUT_DIR) else None

    def resolve_step1_file(raw_path: str) -> Path:
        input_path = Path(raw_path).expanduser()
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON not found: {input_path}")

        inferred_stance = infer_stance_from_file_name(input_path.name)
        if inferred_stance and (
            input_path.name.endswith(f"_Step_1_{inferred_stance}.json")
            or is_stage1_cluster_file(input_path.name, inferred_stance)
        ):
            return input_path

        topic = strip_known_suffix(input_path.name, DEFAULT_STANCE)
        category = input_path.parent.name
        candidate = INPUT_ROOT / category / f"{topic}_Step_1_{DEFAULT_STANCE}.json"
        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            f"Could not find matching Step_1 result for {input_path}. Expected: {candidate}"
        )

    def get_step1_files():
        if SELECTED_INPUT_DIR:
            input_dir = SELECTED_INPUT_DIR
            if not input_dir.exists():
                raise FileNotFoundError(f"Input directory not found: {input_dir}")

            patterns = [
                "*_random_docs_*.json",
                "*_sim_docs_*.json",
                "*_Step_1_*.json",
            ]
            file_list = []
            for pattern in patterns:
                file_list.extend(input_dir.rglob(pattern))
            return sorted(set(file_list))

        selected_file = args.input_json or STEP1_INPUT_FILE
        if selected_file:
            return [resolve_step1_file(selected_file)]
        pattern = f"*_Step_1_{DEFAULT_STANCE}.json"
        return sorted(INPUT_ROOT.rglob(pattern))

    def resolve_category(data_path: Path):
        if SELECTED_INPUT_DIR:
            try:
                relative_parent = data_path.parent.relative_to(SELECTED_INPUT_DIR)
                if str(relative_parent) == ".":
                    return ""
                return relative_parent.as_posix()
            except ValueError:
                pass
        try:
            relative_parts = data_path.relative_to(INPUT_ROOT).parts
            return relative_parts[0] if len(relative_parts) > 1 else data_path.parent.name
        except ValueError:
            return data_path.parent.name

    def process_all_files():
        file_list = get_step1_files()
        
        print(f"🔍 Found {len(file_list)} files to process.")

        for data_path in file_list:
            file_stance = infer_stance_from_file_name(data_path.name) or DEFAULT_STANCE
            category = resolve_category(data_path)
            topic = extract_topic_from_file_name(data_path.name, file_stance)

            output_dir = OUTPUT_ROOT / category if category else OUTPUT_ROOT
            output_dir.mkdir(parents=True, exist_ok=True)
            final_path = output_dir / f"{topic}_adv_trigger_{file_stance}_final.json"

            category_label = category or "."
            print(f"\n🚀 Processing Topic: {topic} in Category: {category_label} with Stance: {file_stance}")
            print(f"Input: {data_path}")

            # 3. Load the source data.
            data = load_attack_items(data_path, file_stance)

            # 4. Check for resumable output.
            trigger_results = []
            start_idx = 0
            if final_path.exists() and not OVERWRITE:
                try:
                    with open(final_path, "r", encoding="utf-8") as f:
                        trigger_results = json.load(f)
                    start_idx = len(trigger_results)
                    print(f"[Resume] idx={start_idx} for {topic}")
                except Exception as e:
                    print(f"[Warn] Could not load existing file {final_path}, starting fresh.")
            elif final_path.exists() and OVERWRITE:
                print(f"[Overwrite] restarting {topic} from scratch")

            # 5. Main processing loop.
            for idx in range(start_idx, len(data)):
                item = data[idx]

                query_list = item.get("query_list", [])
                target_passage = item.get("attack_passage", "")

                if not target_passage:
                    continue

                # Generate an adversarial trigger.
                trigger, new_score, trigger_cands = gen_aggressive_collision(
                    inputs_a=query_list,
                    inputs_b=None,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    margin=None,
                    args=args,
                    target_passge=target_passage
                )

                print(f"[{topic}][{idx}/{len(data)}] Trigger: {trigger}")

                trigger_results.append({
                    "idx": idx,
                    "topic": topic,
                    "category": category,
                    "cluster_id": item.get("cluster_id"),
                    "cluster_method": item.get("cluster_method"),
                    "root_topic": item.get("root_topic"),
                    "root_summary": item.get("root_summary"),
                    "cluster_size": item.get("cluster_size"),
                    "region_nodes": item.get("region_nodes", []),
                    "source_passage": item.get("source_passage"),
                    "attack_passage": target_passage,
                    "query_list": query_list,
                    "trigger": trigger,
                    "new_score": new_score
                })

                # Save after each item so interrupted runs can resume.
                with open(final_path, "w", encoding="utf-8") as f:
                    json.dump(trigger_results, f, indent=4, ensure_ascii=False)

        print("\n✅ [Done] All files and triggers generated.")

    process_all_files()

def find_filters(queries, model, tokenizer, device, k=100):
    words = [w for w in tokenizer.vocab if w.isalpha() and w not in STOPWORDS]
    combined_scores = torch.zeros(len(words), device=device)
    for query in queries:
        inputs = tokenizer.batch_encode_plus([[query, w] for w in words], pad_to_max_length=True)
        all_input_ids = torch.tensor(inputs['input_ids'], device=device)
        all_token_type_ids = torch.tensor(inputs['token_type_ids'], device=device)
        all_attention_masks = torch.tensor(inputs['attention_mask'], device=device)
        n = len(words)
        batch_size = 512
        n_batches = n // batch_size + 1
        all_scores = []

        for i in tqdm(range(n_batches), desc='Processing queries'):
            input_ids = all_input_ids[i * batch_size: (i + 1) * batch_size]
            token_type_ids = all_token_type_ids[i * batch_size: (i + 1) * batch_size]
            attention_masks = all_attention_masks[i * batch_size: (i + 1) * batch_size]
            outputs = model.forward(input_ids, attention_masks, token_type_ids)
            scores = outputs[0][:, 1]
            all_scores.append(scores)

        all_scores = torch.cat(all_scores)
        combined_scores += all_scores

    _, top_indices = torch.topk(combined_scores, k)
    filters = set([words[i.item()] for i in top_indices])
    return [w for w in filters if w.isalpha()]

def find_filters_anchor(queries, anchor_list, model, tokenizer, device, k=150):
    combined_anchor = ' '.join(anchor_list)
    words = list(set(combined_anchor.split()))
    words = [w for w in words if w.isalpha()]
    combined_scores = torch.zeros(len(words), device=device)
    for query in queries:
        pairs = [[query, w] for w in words]
        inputs = tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_masks = inputs['attention_mask'].to(device)
        token_type_ids = inputs.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        n = len(words)
        batch_size = 512
        n_batches = (n + batch_size - 1) // batch_size
        all_scores = []

        for i in tqdm(range(n_batches), desc='Processing queries'):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            batch_input_ids = input_ids[batch_slice]
            batch_attention_masks = attention_masks[batch_slice]
            batch_token_type_ids = token_type_ids[batch_slice] if token_type_ids is not None else None

            # Forward pass
            with torch.no_grad():
                if batch_token_type_ids is not None:
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_masks,
                        token_type_ids=batch_token_type_ids
                    )
                else:
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_masks
                    )
            logits = outputs.logits
            scores = logits[:, 1]
            all_scores.append(scores)
        all_scores = torch.cat(all_scores)
        combined_scores += all_scores
    _, top_indices = torch.topk(combined_scores, k)
    filters = [words[i] for i in top_indices.tolist()]

    return filters


def find_top_relevant_tokens(queries, model, tokenizer, device, k=150):
    words = [w for w in tokenizer.vocab.keys() if w.isalpha() and w.lower() not in STOPWORDS]
    word_ids = tokenizer.convert_tokens_to_ids(words)
    word_ids_tensor = torch.tensor(word_ids, device=device)
    embedding_layer = model.get_input_embeddings()

    with torch.no_grad():
        word_embeddings = embedding_layer(word_ids_tensor)
    combined_scores = torch.zeros(len(words), device=device)

    for query in tqdm(queries, desc='Processing queries'):
        query_inputs = tokenizer(query, return_tensors='pt').to(device)
        input_ids = query_inputs['input_ids']
        with torch.no_grad():
            query_embedding = embedding_layer(input_ids).mean(dim=1)
        similarities = cosine_similarity(query_embedding, word_embeddings)
        combined_scores += similarities.squeeze()

    top_scores, top_indices = torch.topk(combined_scores, k)
    top_tokens = [words[i] for i in top_indices.cpu().numpy()]

    return top_tokens

def get_queries_tokens(queries, tokenizer, top_n=30):
    tokens_in_queries = defaultdict(set)

    for idx, text in enumerate(queries):
        query_tokens = tokenizer.tokenize(text)
        filtered_tokens = [t.lower() for t in query_tokens if t.isalpha() and len(t) > 2 and t.lower() not in STOPWORDS]
        for token in filtered_tokens:
            tokens_in_queries[token].add(idx)

    token_freq = {token: len(query_indices) for token, query_indices in tokens_in_queries.items()}
    sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    top_tokens = [token for token, freq in sorted_tokens[:top_n]]

    return top_tokens

def gen_aggressive_collision(inputs_a, inputs_b, model, tokenizer, device, margin=None, args=None,target_passge=None):
    def relaxed_to_word_embs(x):
        masked_x = x + input_mask + sub_mask
        if args.regularize:
            masked_x += stopwords_mask
        p = torch.softmax(masked_x / args.stemp, -1)
        x = torch.mm(p, word_embedding)
        return p, x.unsqueeze(0)

    def ids_to_emb(input_ids):
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        cls_emb = word_embedding[tokenizer.cls_token_id].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        sep_emb = word_embedding[tokenizer.sep_token_id].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        input_emb = torch.cat([cls_emb, input_emb, sep_emb], dim=1)
        return input_emb

    def ids_to_emb_passage(input_ids):
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
          # (1, 1, hidden_size)
        sep_emb = word_embedding[tokenizer.sep_token_id].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        input_emb = torch.cat([input_emb, sep_emb], dim=1)
        return input_emb

    def ids_to_emb_forbeams(input_ids):
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        cls_emb = word_embedding[tokenizer.cls_token_id].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        sep_emb = word_embedding[tokenizer.sep_token_id].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        batch_cls = torch.cat([cls_emb] * args.topk, 0)
        batch_sep = torch.cat([sep_emb] * args.topk, 0)
        input_emb = torch.cat([batch_cls, input_emb, batch_sep], dim=1)
        return input_emb

    def ids_to_emb_forbeams_passage(input_ids):
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        return input_emb

    word_embedding = model.get_input_embeddings().weight.detach()
    vocab_size = word_embedding.size(0)
    input_mask = torch.zeros(vocab_size, device=device)
    filters = find_top_relevant_tokens(inputs_a, model, tokenizer, device, k=args.num_filters)
    remove_tokens =get_queries_tokens(inputs_a, tokenizer)
    remove_ids = tokenizer.convert_tokens_to_ids(remove_tokens)
    remove_ids.append(tokenizer.vocab['.'])
    input_mask[remove_ids] = 0.68
    num_filters_ids = tokenizer.convert_tokens_to_ids(filters)
    input_mask[num_filters_ids] = 0.68
    sub_mask = get_sub_masks(tokenizer, device)
    input_mask[tokenizer.convert_tokens_to_ids(['.', '@', '=','_'])] = -1e9
    unk_ids = tokenizer.encode('<unk>', add_special_tokens=False)
    input_mask[unk_ids] = -1e9
    query_ids_list = [torch.tensor(tokenizer.encode(query, add_special_tokens=False), device=device) for query in inputs_a]
    query_ids=query_ids_list
    seq_len = args.seq_len
    passage_ids=torch.tensor(tokenizer.encode(target_passge,max_length=400, add_special_tokens=False,truncation=True), device=device)
    passage_ids=passage_ids.unsqueeze(0)
    stopwords_mask = create_constraints(seq_len, tokenizer, device)
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * args.topk, device=device)
    batch_sep_embeds = word_embedding[sep_tensor].unsqueeze(1)
    cls_tensor = torch.tensor([tokenizer.cls_token_id]* args.topk, device=device)
    batch_cls_embeds = word_embedding[cls_tensor].unsqueeze(1)

    if args.target == "nb_bert":
        labels = torch.tensor([[0, 1]] * len(inputs_a), dtype=torch.float, device=device)
    repetition_penalty = 1.0
    best_collision = None
    best_score = -1e9
    prev_score = -1e9
    collision_cands = []
    patience = 0
    var_size = (seq_len, vocab_size)
    z_i = torch.zeros(*var_size, requires_grad=True, device=device)

    def pad_embeds(embeds, max_length):
        current_length = embeds.shape[1]
        if current_length == max_length:
            return embeds
        padding_size = max_length - current_length
        padding_tensor = torch.zeros((embeds.shape[0], padding_size, embeds.shape[2]), device=embeds.device)
        padded_embeds = torch.cat([embeds, padding_tensor], dim=1)
        return padded_embeds

    def pad_sequence(seq, max_length, pad_value=0):
        seq_len = seq.shape[0]
        pad_len = max_length - seq_len
        if pad_len > 0:
            padding = torch.full((pad_len, *seq.shape[1:]), pad_value, dtype=seq.dtype, device=device)
            seq = torch.cat([seq, padding], dim=0)
        return seq.unsqueeze(0)

    patience_score=0

    for it in tqdm(range(args.max_iter), desc="Main Iteration"):
        if patience_score<2:
            optimizer = torch.optim.Adam([z_i], lr=args.lr)
            iter_num=args.perturb_iter
        elif patience_score==2:
            iter_num=300
            optimizer = torch.optim.Adam([z_i], lr=0.005)

        else:
            iter_num=500
            optimizer = torch.optim.Adam([z_i], lr=0.005)

        for j in range(iter_num):
            total_loss = 0.0
            input_embeds_list=[]
            attention_mask_list = []
            token_type_ids_list = []
            for q_idx, query_id in tqdm(enumerate(query_ids), total=len(query_ids), desc="Processing Queries", leave=False):
                query_id = query_id.unsqueeze(0)  # [1, seq_len]
                query_emb = ids_to_emb(query_id)

                passage_emb=ids_to_emb_passage(passage_ids)
                p_inputs, inputs_embeds = relaxed_to_word_embs(z_i)
                trigger_passage_emb=torch.cat([inputs_embeds,passage_emb],dim=1)
                concat_inputs_emb = torch.cat([query_emb, trigger_passage_emb], dim=1)
                input_embeds_list.append(concat_inputs_emb)
                seq_len_1 = concat_inputs_emb.shape[1]
                attention_mask = torch.ones(seq_len_1, dtype=torch.long, device=device)
                attention_mask_list.append(attention_mask)

                query_len = query_emb.shape[1]
                passage_len = trigger_passage_emb.shape[1]
                token_type_ids = torch.cat([
                    torch.zeros(query_len, dtype=torch.long, device=device),
                    torch.ones(passage_len, dtype=torch.long, device=device)
                ], dim=0)
                token_type_ids_list.append(token_type_ids)
            max_length = max(emb.shape[1] for emb in input_embeds_list)
            padded_input_embeds_list = []
            padded_attention_mask_list = []
            padded_token_type_ids_list = []
            for emb, attn_mask, token_type_id in zip(input_embeds_list, attention_mask_list, token_type_ids_list):
                padded_emb = pad_embeds(emb, max_length)
                padded_input_embeds_list.append(padded_emb)
                padded_attn_mask = pad_sequence(attn_mask, max_length, pad_value=0)
                padded_attention_mask_list.append(padded_attn_mask)
                padded_token_type_id = pad_sequence(token_type_id, max_length, pad_value=0)
                padded_token_type_ids_list.append(padded_token_type_id)



            input_embeds = torch.cat(padded_input_embeds_list, dim=0)
            attention_mask_tensor = torch.cat(padded_attention_mask_list, dim=0)
            token_type_ids_tensor = torch.cat(padded_token_type_ids_list, dim=0)

            labels_tensor = labels
            optimizer.zero_grad()

            if args.target == "nb_bert":
                outputs = model(inputs_embeds=input_embeds,
                                attention_mask=attention_mask_tensor,
                                token_type_ids=token_type_ids_tensor,
                                labels=labels_tensor)

                loss, cls_logits = outputs[0], outputs[1]
                cls_logits_score = cls_logits[:, 1]

            loss.backward()
            optimizer.step()


            if j%10==0:
                avg_loss = loss.item()
                print(f"Iteration {j}, Average Loss: {avg_loss}")
        z_i = z_i.detach()

        _, topk_tokens = torch.topk(z_i, args.topk)
        probs_i = torch.softmax(z_i / args.stemp, -1).unsqueeze(0).expand(args.topk, seq_len, vocab_size)
        output_so_far = None
        for t in range(seq_len):
            t_topk_tokens = topk_tokens[t]
            t_topk_onehot = torch.nn.functional.one_hot(t_topk_tokens, vocab_size).float()
            next_clf_scores = []
            for j in range(args.num_beams):
                next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
                if output_so_far is None:
                    context = probs_i.clone()
                else:
                    output_len = output_so_far.shape[1]
                    beam_topk_output = output_so_far[j].unsqueeze(0).expand(args.topk, output_len)
                    beam_topk_output = torch.nn.functional.one_hot(beam_topk_output, vocab_size)
                    context = torch.cat([beam_topk_output.float(), probs_i[:, output_len:].clone()], 1)
                context[:, t] = t_topk_onehot
                context_embeds1 = torch.einsum('blv,vh->blh', context, word_embedding)
                total_clf_scores = None
                query_count = 0
                beam_embedding_list=[]
                beams_attention_mask_list = []
                beams_token_type_ids_list = []
                for _, query_id in tqdm(enumerate(query_ids), total=len(query_ids), desc="Processing Queries", leave=False):  # Process each query.

                    query1=query_id.unsqueeze(0)

                    batch_query_ids = torch.cat([query1] * args.topk, 0)
                    batch_passage_ids=torch.cat([passage_ids]*args.topk,0)
                    batch_passage_emb=ids_to_emb_forbeams_passage(batch_passage_ids)
                    context_embeds = torch.cat([context_embeds1,batch_passage_emb,batch_sep_embeds], 1)
                    batch_query_emb = ids_to_emb_forbeams(batch_query_ids)
                    concat_batch_inputs_emb = torch.cat([batch_query_emb, context_embeds], dim=1)
                    seq_len1=concat_batch_inputs_emb.shape[1]
                    query_length_beams=batch_query_emb.shape[1]
                    trigger_length_beams=context_embeds.shape[1]
                    attention_beams=torch.ones((args.topk, seq_len1), dtype=torch.long, device=device)
                    beams_attention_mask_list.append(attention_beams)
                    token_type_ids_beams = torch.cat([torch.zeros((args.topk, query_length_beams), dtype=torch.long, device=device),
                                                    torch.ones((args.topk, trigger_length_beams), dtype=torch.long, device=device)], dim=1)
                    beam_embedding_list.append(concat_batch_inputs_emb)
                    beams_token_type_ids_list.append(token_type_ids_beams)

                for beam_emb, attn_mask, token_type_id in tqdm(zip(beam_embedding_list, beams_attention_mask_list, beams_token_type_ids_list), total=len(beam_embedding_list), desc="Processing Beams"):

                    if args.target == "nb_bert":
                        outputs_1 = model(inputs_embeds = beam_emb,
                                        attention_mask=attn_mask,
                                        token_type_ids=token_type_id,
                        )

                        clf_logits = outputs_1[0]
                        clf_logits_score = clf_logits[:, 1]
                    if total_clf_scores is None:
                        total_clf_scores = clf_logits_score
                    else:
                        total_clf_scores += clf_logits_score
                    query_count += 1
                if total_clf_scores is not None and query_count > 0:
                    avg_clf_scores = total_clf_scores / query_count
                clf_scores = avg_clf_scores.detach().float()
                next_beam_scores.scatter_(0, t_topk_tokens, clf_scores)
                next_clf_scores.append(next_beam_scores.unsqueeze(0))

            next_clf_scores = torch.cat(next_clf_scores, 0)
            next_scores = next_clf_scores + input_mask + sub_mask

            if args.regularize:
                next_scores += stopwords_mask[t]

            if output_so_far is None:
                next_scores[1:] = -1e9

            if output_so_far is not None and repetition_penalty > 1.0:
                lm_model.enforce_repetition_penalty_(next_scores, 1, args.num_beams, output_so_far, repetition_penalty)
            next_scores = next_scores.view(1, args.num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, args.num_beams, dim=1, largest=True, sorted=True)
            next_sent_beam = []
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[0], next_scores[0])):
                beam_id = torch.div(beam_token_id, vocab_size, rounding_mode='trunc')
                token_id = beam_token_id % vocab_size
                next_sent_beam.append((beam_token_score, token_id, beam_id))

            next_batch_beam = next_sent_beam
            assert len(next_batch_beam) == args.num_beams
            beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=device)
            beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=device)

            if output_so_far is None:
                output_so_far = beam_tokens.unsqueeze(1)
            else:
                output_so_far = output_so_far[beam_idx, :]
                output_so_far = torch.cat([output_so_far, beam_tokens.unsqueeze(1)], dim=-1)
        pad_output_so_far = torch.cat([output_so_far, batch_passage_ids[:args.num_beams],sep_tensor[:args.num_beams].unsqueeze(1)], 1)
        final_input_list = []
        final_type_list = []

        max_length = 0
        for _, query_id in tqdm(enumerate(query_ids), total=len(query_ids), desc="Processing Queries", leave=False):
            query1 = query_id.unsqueeze(0)
            batch_query_ids = torch.cat([query1] * args.topk, 0)
            concat_query_ids = torch.cat([batch_query_ids[:args.num_beams], pad_output_so_far], 1)
            max_length = max(max_length, concat_query_ids.size(1))
        for _, query_id in tqdm(enumerate(query_ids), total=len(query_ids), desc="Padding Queries", leave=False):
            query1 = query_id.unsqueeze(0)
            batch_query_ids = torch.cat([query1] * args.topk, 0)

            batch_query_emb = ids_to_emb_forbeams(batch_query_ids)
            concat_query_ids = torch.cat([batch_query_ids[:args.num_beams], pad_output_so_far], 1)
            token_type_ids = torch.cat([torch.zeros_like(batch_query_ids[:args.num_beams]), torch.ones_like(pad_output_so_far)], 1)
            pad_length = max_length - concat_query_ids.size(1)

            if pad_length > 0:
                padding = torch.zeros((concat_query_ids.size(0), pad_length), dtype=torch.long).to(device)
                concat_query_ids = torch.cat([concat_query_ids, padding], 1)
                token_type_ids = torch.cat([token_type_ids, torch.ones((token_type_ids.size(0), pad_length), dtype=torch.long).to(device)], 1)
            final_input_list.append(concat_query_ids)
            final_type_list.append(token_type_ids)


        clf_logits_score_sum = 0
        num_elements = 0
        for concat_query_ids, token_type_ids in tqdm(zip(final_input_list, final_type_list), total=len(final_input_list), desc="Processing Queries", leave=False):
            # [10, 2]
            if args.target == "mini":
                outputs_2 = model(
                    input_ids=concat_query_ids,
                    token_type_ids=token_type_ids
                )
                clf_logits = outputs_2[0]
                clf_logits_score = clf_logits[:, 0]
            elif args.target == "nb_bert":
                outputs_2 = model(
                    input_ids=concat_query_ids,
                    token_type_ids=token_type_ids
                )
                clf_logits = outputs_2[0]
                clf_logits_score = clf_logits[:, 1]
            elif args.target == "mini_adv" or args.target == "nb_bert_adv":
                outputs_2 = model(
                    input_ids_pos=concat_query_ids,
                    input_ids_neg=concat_query_ids,
                    token_type_ids_pos=token_type_ids,
                    token_type_ids_neg=token_type_ids,
                )
                clf_logits = outputs_2[0]
                clf_logits_score = clf_logits[:, 1]
            elif args.target == "bge":
                outputs_2 = model(
                    query_id=batch_query_ids[:args.num_beams],
                    passage_id=pad_output_so_far,
                )
                clf_logits = outputs_2[0]
                clf_logits_score = clf_logits[:]

            if clf_logits_score_sum is None:
                clf_logits_score_sum = clf_logits_score
            else:
                clf_logits_score_sum += clf_logits_score
            num_elements += 1
        actual_clf_scores = clf_logits_score_sum / num_elements
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        if args.verbose:
            decoded = [
                f'{actual_clf_scores[i].item():.4f}, '
                f'{tokenizer.decode(output_so_far[i].cpu().tolist())}'
                for i in sorter
            ]

        valid_idx = sorter[0]
        valid = True
        curr_best = output_so_far[valid_idx]
        next_z_i = torch.nn.functional.one_hot(curr_best, vocab_size).float()
        eps = 0.1
        next_z_i = (next_z_i * (1 - eps)) + (1 - next_z_i) * eps / (vocab_size - 1)
        z_i = torch.nn.Parameter(torch.log(next_z_i), True)

        curr_score = actual_clf_scores[valid_idx].item()
        if valid and curr_score > best_score:
            patience = 0
            patience_score=0
            best_score = curr_score
            best_collision = tokenizer.decode(curr_best.cpu().tolist())
            print(curr_score)

        if curr_score <= prev_score:
            # break
            patience += 1
            patience_score+=1
        if patience > args.patience_limit:
            break
        prev_score = curr_score
        print(best_collision)
    return best_collision, best_score, collision_cands

if __name__ == '__main__':
    main()
