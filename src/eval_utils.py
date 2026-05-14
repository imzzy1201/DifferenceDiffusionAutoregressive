from config import *
import torch
from transformers import PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
from tqdm import tqdm


def eval_ngram_entropy(contents, n: int, pad_token_id: int):
    counts = {}
    total_ngrams = 0
    for seq in contents:
        for i in range(len(seq) - n + 1):
            ngram = tuple(seq[i: i + n])
            if pad_token_id in ngram:
                continue
            counts[ngram] = counts.get(ngram, 0) + 1
            total_ngrams += 1
            if total_ngrams >= args.eval_max_num_grams and args.eval_max_num_grams > 0:
                break
        if total_ngrams >= args.eval_max_num_grams and args.eval_max_num_grams > 0:
            break
    total_ngrams = max(total_ngrams, 1)
    probs = {ngram: count / total_ngrams for ngram, count in counts.items()}
    entropy = 0
    for p in probs.values():
        if p > 0:
            entropy -= p * torch.log(torch.tensor(p))
    return entropy.item(), probs, {"total_ngrams": total_ngrams, "top_10_ngrams": sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]}


def split_into_sentences(seq: List[int], sentence_split_token_ids: List[int]) -> List[List[int]]:
    sentences = []
    last_split = -1
    for i, token_id in enumerate(seq):
        if token_id in sentence_split_token_ids:
            sentences.append(seq[last_split + 1: i + 1])
            last_split = i
    if last_split != len(seq) - 1:
        sentences.append(seq[last_split + 1:])
    return sentences


def compute_semantic_variances_local(contents: List[List[int]], tokenizer: PreTrainedTokenizerFast, embedding_models_dict: Dict[str, SentenceTransformer], accelerator=None, gather_embeddings: bool = False) -> Dict[str, Dict[str, float]]:
    if len(embedding_models_dict) == 0:
        return {}

    sentence_split_token_ids = [
        ids[0] for ids in [tokenizer.encode(token, add_special_tokens=False) for token in eos_tokens_list]
        if len(ids) == 1
    ]
    results = {}

    # Pre-collect all texts and sentences to enable batched encoding
    all_texts = []
    all_sentences_flat = []
    sentence_ranges = []  # (start_index, length)

    disable_tqdm = not accelerator.is_local_main_process if accelerator else False

    # Decode and split once
    for seq in tqdm(contents, desc="Preprocessing texts", disable=disable_tqdm):
        if tokenizer.pad_token_id in seq:
            seq = seq[: seq.index(tokenizer.pad_token_id)]

        # Decode text
        text = tokenizer.decode(seq, skip_special_tokens=True)
        if text:
            all_texts.append(text)

        # Split sentences
        sentences = split_into_sentences(seq, sentence_split_token_ids)
        if len(sentences) >= 2:
            sentence_texts = [tokenizer.decode(sent_tokens, skip_special_tokens=True).strip() for sent_tokens in sentences]
            current_start = len(all_sentences_flat)
            count = len(sentence_texts)
            all_sentences_flat.extend(sentence_texts)
            sentence_ranges.append((current_start, count))

    for model_name, emb_model in embedding_models_dict.items():
        # 1. Inter-sequence embeddings (full texts)
        if all_texts:
            print(f"Encoding full texts with {model_name}...")
            # Batch encode
            raw_inter_embeddings = emb_model.encode(all_texts, convert_to_numpy=True, show_progress_bar=not disable_tqdm)
            # Normalize
            inter_embeddings = raw_inter_embeddings / np.linalg.norm(raw_inter_embeddings, axis=1, keepdims=True)
        else:
            dim = emb_model.get_sentence_embedding_dimension()
            inter_embeddings = np.zeros((0, dim))

        inter_embeddings_sum = np.sum(inter_embeddings, axis=0) if len(inter_embeddings) > 0 else np.zeros(emb_model.get_sentence_embedding_dimension())
        inter_embeddings_squared_sum = np.sum(inter_embeddings**2, axis=0) if len(inter_embeddings) > 0 else np.zeros(emb_model.get_sentence_embedding_dimension())
        inter_embeddings_count = len(inter_embeddings)

        # 2. Adjacent sentence similarity
        adjacent_sentence_similarity_sum = 0.0
        adjacent_sentence_similarity_count = 0

        if all_sentences_flat:
            print(f"Encoding sentences with {model_name}...")
            # Batch encode sentences
            raw_sent_embs = emb_model.encode(all_sentences_flat, convert_to_numpy=True, show_progress_bar=not disable_tqdm)
            norms = np.linalg.norm(raw_sent_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9  # avoid division by zero
            sent_embs = raw_sent_embs / norms

            # Aggregate per sequence
            for start, count in sentence_ranges:
                seq_embs = sent_embs[start: start + count]
                adjacent_similarity = np.mean(np.sum(seq_embs[:-1] * seq_embs[1:], axis=-1))
                adjacent_sentence_similarity_sum += adjacent_similarity
                adjacent_sentence_similarity_count += 1

        results[model_name] = {
            "adjacent_sentence_similarity_sum": float(adjacent_sentence_similarity_sum),
            "adjacent_sentence_similarity_count": adjacent_sentence_similarity_count,
            "inter_embeddings_sum": inter_embeddings_sum,
            "inter_embeddings_squared_sum": inter_embeddings_squared_sum,
            "inter_embeddings_count": inter_embeddings_count,
        }
        if gather_embeddings:
            results[model_name]["full_embeddings"] = inter_embeddings
    return results


def compute_metrics(contents, tokenizer: PreTrainedTokenizerFast, **kwargs):
    unigram_entropy, unigram_probs, info = eval_ngram_entropy(contents, n=1, pad_token_id=tokenizer.pad_token_id)
    _2_gram_entropy, _, _ = eval_ngram_entropy(contents, n=2, pad_token_id=tokenizer.pad_token_id)
    _3_gram_entropy, _, _ = eval_ngram_entropy(contents, n=3, pad_token_id=tokenizer.pad_token_id)
    ret = {
        "1_gram_entropy": unigram_entropy,
        "2_gram_entropy": _2_gram_entropy,
        "3_gram_entropy": _3_gram_entropy,
    }
    return ret, unigram_probs, {"total_tokens": info["total_ngrams"], "top_10_unigrams": info["top_10_ngrams"]}
