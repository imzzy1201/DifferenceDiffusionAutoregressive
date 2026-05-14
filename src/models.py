from config import *
from eval_utils import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoTokenizer
from tqdm import tqdm
from accelerate import Accelerator
from typing import Union, Optional


def init_context_scope(core: Union[CoreVariables, dict]):
    if isinstance(core, CoreVariables):
        core = asdict(core)

    global context_scope, core_str, model_str, input_masking, label_masking, unidirectional_context, weighting_function, block_length, remasking_strategy, external_generation, resample
    context_scope = core["context_scope"]
    assert context_scope in ["uc", "bc"]

    input_masking_value = core.get("input_masking", None)
    label_masking_value = core.get("label_masking", None)
    weighting_function_value = core.get("weighting_function", None)

    input_masking = input_masking_value if input_masking_value is not None else (context_scope == "bc")
    label_masking = label_masking_value if label_masking_value is not None else (context_scope == "bc")
    unidirectional_context = (context_scope == "uc")
    weighting_function = weighting_function_value if weighting_function_value is not None else (context_scope == "bc")
    block_length = core["block_length"] if core.get("block_length", None) is not None else 1
    remasking_strategy = core["remasking_strategy"] if core.get("remasking_strategy", None) is not None else "rr"
    external_generation = core.get("external_generation", None)
    resample = core["resample"] if core.get("resample", None) is not None else False

    # Build canonical objective name to match paper notation.
    objective_parts = [context_scope]
    if input_masking:
        objective_parts.append("im")
    if label_masking:
        objective_parts.append("lm")
    if weighting_function:
        objective_parts.append("wf")
    core_str = "+".join(objective_parts)
    model_str = context_scope
    if block_length != 1:
        core_str += f"=bl{block_length}"
        core_str += f"+{remasking_strategy}"
    if resample:
        core_str += "+rs"

    print("Context scope:", context_scope)
    print("Input masking:", input_masking)
    print("Label masking:", label_masking)
    print("Weighting function:", weighting_function)
    print("Block length:", block_length)
    print("Remasking strategy:", remasking_strategy)
    print("External generation:", external_generation)
    print("Resample:", resample)

    assert (not unidirectional_context) or (block_length == 1), "Block length > 1 not supported with unidirectional context."
    assert remasking_strategy in ["rr", "lcr", "dlcr", "er"], f"Unsupported remasking strategy: {remasking_strategy}"

init_context_scope(args.core)


class DataCollator:
    def __init__(self, tokenizer, padding="max_length", max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length or args.max_length

    def __call__(self, batch):
        # 1. Pad inputs
        batch = self.tokenizer(
            [example["text"] for example in batch],
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"]
        attention_mask = batch.pop("attention_mask")
        lengths = attention_mask.sum(dim=1)
        raw_input_ids = input_ids.clone()
        batch_size, seq_len = input_ids.shape

        # 2. Sample t ~ U(noise_min, noise_max)
        t = torch.rand(batch_size, device=input_ids.device) * (args.noise_max - args.noise_min) + args.noise_min

        # 3. Generate mask
        rand_matrix = torch.rand(batch_size, seq_len, device=input_ids.device)
        t_expanded = t.unsqueeze(-1).expand(-1, seq_len)
        mask_prob = (rand_matrix < t_expanded)  # & (attention_mask.bool())

        # 4. Create labels: -100 for unmasked, original id for masked
        labels = input_ids.clone()
        # labels[~attention_mask.bool()] = -100  # Ignore padding tokens
        if label_masking:
            labels[~mask_prob] = -100  # We only compute loss on masked tokens
        if input_masking:
            input_ids[mask_prob] = self.tokenizer.mask_token_id

        # 5. Create attention mask
        # expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len)
        if unidirectional_context:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device, dtype=torch.bool)
        else:
            attention_mask = torch.ones((batch_size, 1, seq_len, seq_len), device=input_ids.device, dtype=torch.bool)

        batch["input_ids"] = input_ids
        batch["raw_input_ids"] = raw_input_ids
        batch["labels"] = labels
        batch["t"] = t  # Store t for loss scaling
        batch["attention_mask"] = attention_mask  # Pass as boolean 4D mask
        batch["lengths"] = lengths

        return batch


def init_model(fast_tokenizer: PreTrainedTokenizerFast):
    if args.flash_attention:
        print("[WARNING] Flash attention is not supported in this script yet.")
    if args.model == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        config = Qwen2Config(
            vocab_size=len(fast_tokenizer),
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            max_position_embeddings=512,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            torch_dtype=torch_dtype,
            use_cache=unidirectional_context,
            is_decoder=unidirectional_context,
        )
        model = Qwen2ForCausalLM(config)
    elif args.model == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(
            vocab_size=len(fast_tokenizer),
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=512,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            torch_dtype=torch_dtype,
            use_cache=unidirectional_context,
            is_decoder=unidirectional_context,
        )
        model = LlamaForCausalLM(config)
    else:
        raise ValueError(f"Unsupported context scope: {args.model}")
    model = model.to(torch_dtype)
    model.train()
    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")
    return model


loss_fct = torch.nn.CrossEntropyLoss(reduction="none")


def train_batch(model: AutoModelForCausalLM, batch: dict, accelerator: Optional[Accelerator] = None):
    labels = batch.pop("labels")
    if context_scope == "uc":
        labels = labels[:, 1:]
    outputs = model(**batch, is_causal=unidirectional_context)
    logits = outputs.logits
    if context_scope == "uc":
        logits = logits[:, :-1, :]
    loss_per_token = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss_per_token = loss_per_token.view(labels.shape)
    mask = (labels != -100).float()
    loss_per_token = loss_per_token * mask
    loss_per_sample = loss_per_token.sum(dim=1)
    loss_per_sample = loss_per_sample / labels.shape[-1]
    if weighting_function:
        t = batch.pop("t")
        loss_per_sample = loss_per_sample / t
    loss = loss_per_sample.mean()
    # loss = loss / gradient_accumulation_steps
    if torch.isfinite(loss):
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
    return loss.item()


def evaluate_ntp_loss_and_entropy(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, eval_dataloader: DataLoader, accelerator: Optional[Accelerator] = None):
    with torch.inference_mode():
        sum_loss = torch.zeros(1, device=model.device)
        sum_entropy = torch.zeros(1, device=model.device)

        for batch in tqdm(eval_dataloader, desc="Evaluating (NTP Loss)", disable=not accelerator.is_local_main_process if accelerator else False):
            input_ids = batch["raw_input_ids"].to(model.device)
            batch_size = input_ids.size(0)

            def evaluate_one_batch():
                nonlocal sum_loss, sum_entropy
                if unidirectional_context:
                    outputs = model(input_ids=input_ids)
                    start_idx = args.eval_prompt_length
                    relevant_logits = outputs.logits[:, start_idx - 1:-1, :]
                    relevant_labels = input_ids[:, start_idx:]
                    mask = (relevant_labels != tokenizer.pad_token_id).float()

                    loss_log_probs = F.log_softmax(relevant_logits, dim=-1)
                    target_log_probs = torch.gather(loss_log_probs, dim=-1, index=relevant_labels.unsqueeze(-1)).squeeze(-1)
                    token_loss = -target_log_probs
                    seq_counts = mask.sum(dim=1).clamp(min=1.0)
                    sum_loss += ((token_loss * mask).sum(dim=1) / seq_counts).sum()

                    entropy_logits = relevant_logits / args.eval_temperature
                    if not args.eval_allow_eos:
                        entropy_logits[:, :, tokenizer.eos_token_id] = -1e10
                    entropy_per_token = torch.distributions.Categorical(logits=entropy_logits).entropy()
                    sum_entropy += ((entropy_per_token * mask).sum(dim=1) / seq_counts).sum()
                else:
                    curr_input_ids = input_ids.clone()
                    curr_input_ids[:, args.eval_prompt_length:] = tokenizer.mask_token_id
                    full_attn_mask = torch.ones((batch_size, 1, max_length, max_length), dtype=torch.bool).to(model.device)

                    seq_loss_sum = torch.zeros(batch_size, device=model.device)
                    seq_entropy_sum = torch.zeros(batch_size, device=model.device)
                    seq_counts = torch.zeros(batch_size, device=model.device)
                    stop_step = min(max_length, input_ids.size(1))

                    for step in range(args.eval_prompt_length, stop_step):
                        target_tokens = input_ids[:, step]
                        step_mask = (target_tokens != tokenizer.pad_token_id).float()
                        if step_mask.sum() == 0:
                            curr_input_ids[:, step] = target_tokens
                            continue

                        outputs = model(input_ids=curr_input_ids, attention_mask=full_attn_mask, is_causal=False)
                        logits = outputs.logits[:, step - (1 if context_scope == "uc" else 0), :]
                        loss_log_probs = F.log_softmax(logits, dim=-1)
                        target_log_probs = torch.gather(loss_log_probs, dim=1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                        seq_loss_sum += (-target_log_probs) * step_mask

                        entropy_logits = logits / args.eval_temperature
                        if not args.eval_allow_eos:
                            entropy_logits[:, tokenizer.eos_token_id] = -1e10
                        entropy_per_token = torch.distributions.Categorical(logits=entropy_logits).entropy()
                        seq_entropy_sum += entropy_per_token * step_mask

                        seq_counts += step_mask
                        curr_input_ids[:, step] = target_tokens

                    seq_counts = seq_counts.clamp(min=1.0)
                    sum_loss += (seq_loss_sum / seq_counts).sum()
                    sum_entropy += (seq_entropy_sum / seq_counts).sum()

            if model_dtype == "float32":
                evaluate_one_batch()
            else:
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    evaluate_one_batch()

        if accelerator:
            sum_loss = accelerator.reduce(sum_loss, reduction="sum")
            sum_entropy = accelerator.reduce(sum_entropy, reduction="sum")

        avg_loss = sum_loss.item() / len(eval_dataloader.dataset)
        avg_entropy = sum_entropy.item() / len(eval_dataloader.dataset)
        return {"ntp_loss": avg_loss, "entropy": avg_entropy}


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def evaluate(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, eval_dataloader: DataLoader, accelerator: Optional[Accelerator] = None, embedding_models_dict={}, verbose: bool = False, return_logits: bool = False, external_contents=None, ref_model=None):
    if return_logits:
        assert accelerator is None, "return_logits is not supported with accelerator."
    with torch.no_grad():
        contents = []
        local_contents = []  # For semantic variance calculation (before gather)
        embeddings = {}
        all_logits = [] if return_logits else None

        external = False
        if external_generation is not None:
            external = True
            print("Evaluating external generations from:", external_generation)
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            import json
            texts = []
            with open(external_generation, "r") as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data["conversation"]["message"]["content"])
            # !!! WARNING: truncated to args.max_length
            token_ids = tokenizer(texts, padding="max_length", max_length=args.max_length, truncation=True, return_tensors="pt")["input_ids"]
            eval_dataset = torch.utils.data.TensorDataset(token_ids)
            eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
            if accelerator:
                eval_dataloader = accelerator.prepare(eval_dataloader)
        if external_contents is not None:
            assert not external, "Cannot use both external_contents and external_generation."
            external = True
            eval_dataset = torch.utils.data.TensorDataset(torch.tensor(external_contents))
            eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
            if accelerator:
                eval_dataloader = accelerator.prepare(eval_dataloader)

        sum_logp = torch.zeros(1, device=model.device)
        use_block_gen = (block_length > 1)
        if return_logits:
            assert not use_block_gen, "return_logits is not supported with block generation."

        if external:
            for batch in tqdm(eval_dataloader, desc="Loading external generations", disable=not accelerator.is_local_main_process if accelerator else False):
                if external_generation is not None:
                    local_contents.extend(batch[0][:, args.eval_prompt_length:].cpu().numpy().tolist())
                else:
                    local_contents.extend(batch[0].cpu().numpy().tolist())
            device = accelerator.device if accelerator else torch.device("cpu")
            contents = accelerator.gather(torch.tensor(local_contents, device=device)).cpu().numpy().tolist() if accelerator else local_contents
        else:
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process if accelerator else False):
                outputs = None
                input_ids = batch["raw_input_ids"].to(model.device)
                if unidirectional_context:
                    input_ids = input_ids[:, : args.eval_prompt_length]
                else:
                    input_ids[:, args.eval_prompt_length:] = tokenizer.mask_token_id
                    full_attn_mask = torch.ones((input_ids.size(0), 1, max_length, max_length), dtype=torch.bool).to(model.device)

                def generate_ntp():
                    # Generate by next token prediction
                    nonlocal input_ids, full_attn_mask, sum_logp, outputs, all_logits
                    cur_logp = torch.zeros(input_ids.size(0), device=model.device)
                    n_token = torch.zeros(input_ids.size(0), device=model.device)
                    stopped = torch.zeros(input_ids.size(0), dtype=torch.bool, device=model.device)
                    batch_logits = [] if return_logits else None
                    # batch_prefix_ntp = []  # NTP for this batch (first step only)
                    for step in range(args.eval_prompt_length, max_length):
                        if unidirectional_context:
                            outputs = model(input_ids=input_ids)
                        else:
                            outputs = model(input_ids=input_ids, attention_mask=full_attn_mask, is_causal=False)
                        if context_scope == "uc":
                            next_token_logits = outputs.logits[:, step - 1, :]
                        else:
                            next_token_logits = outputs.logits[:, step, :]
                        next_token_logits_scaled = next_token_logits / args.eval_temperature
                        if not args.eval_allow_eos:
                            # Create a mask for all tokens except EOS
                            not_eos_mask = torch.ones(next_token_logits_scaled.shape[-1], dtype=torch.bool, device=next_token_logits_scaled.device)
                            not_eos_mask[tokenizer.eos_token_id] = False

                        if return_logits:
                            if not args.eval_allow_eos:
                                # Save logits excluding EOS token
                                batch_logits.append(next_token_logits_scaled[:, not_eos_mask].detach().cpu())
                            else:
                                batch_logits.append(next_token_logits_scaled.detach().cpu())

                        if not args.eval_allow_eos:
                            next_token_logits_scaled[:, tokenizer.eos_token_id] = -1e10
                        probs = F.softmax(next_token_logits_scaled, dim=-1)
                        # if step == args.eval_prompt_length:  # Only store first step NTP
                        #     batch_prefix_ntp.append(probs.cpu().numpy())
                        if step > 0:
                            stopped |= input_ids[:, step - 1] == tokenizer.pad_token_id
                        next_tokens = torch.multinomial(probs, num_samples=1)
                        if unidirectional_context:
                            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                        else:
                            input_ids[:, step] = next_tokens.squeeze(-1)
                        cur_logp += torch.log(torch.gather(probs, dim=1, index=input_ids[:, step].unsqueeze(-1).long()).squeeze(-1)) * (1 - stopped.float())
                        n_token += 1 - stopped.float()
                    n_token = torch.clamp(n_token, min=1.0)
                    sum_logp += (cur_logp / n_token).sum()
                    outputs = input_ids

                    if return_logits:
                        batch_logits_tensor = torch.stack(batch_logits, dim=1)
                        for i in range(input_ids.size(0)):
                            seq_len = int(n_token[i].item())
                            all_logits.append(batch_logits_tensor[i, :seq_len, :].float().numpy())
                    # prefix_ntp.extend(batch_prefix_ntp)  # Add to global list

                def generate_block():
                    # Generate by block
                    nonlocal input_ids, full_attn_mask, outputs
                    x = input_ids
                    prompt_len = args.eval_prompt_length
                    gen_len = x.shape[1] - prompt_len

                    num_blocks = (gen_len - 1) // block_length + 1

                    for num_block in range(num_blocks):
                        block_start = prompt_len + num_block * block_length
                        block_end = min(prompt_len + (num_block + 1) * block_length, x.shape[1])
                        current_block_len = block_end - block_start

                        for i in range(current_block_len):
                            mask_index = (x == tokenizer.mask_token_id)
                            # If no masked token, break early
                            if not mask_index.any():
                                break

                            outputs_model = model(x, attention_mask=full_attn_mask, is_causal=False)
                            logits = outputs_model.logits

                            if not args.eval_allow_eos:
                                logits[:, :, tokenizer.eos_token_id] = -torch.inf

                            logits_with_noise = add_gumbel_noise(logits, temperature=args.eval_temperature)
                            x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

                            p = F.softmax(logits, dim=-1)
                            confidence = torch.squeeze(
                                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # (B, L)

                            if remasking_strategy == "rr":
                                confidence = torch.rand_like(confidence)
                            elif remasking_strategy == "er":
                                log_probs = F.log_softmax(logits, dim=-1)
                                confidence = torch.sum(p * log_probs, dim=-1)
                                # Higher entropy means less confidence
                            elif remasking_strategy not in ["lcr", "dlcr"]:
                                raise ValueError(f"Unsupported remasking strategy: {remasking_strategy}")

                            if block_end < x.shape[1]:
                                confidence[:, block_end:] = -np.inf

                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index, confidence, -np.inf)

                            _, select_indices = torch.topk(confidence, k=1, dim=-1)  # (B, 1)
                            transfer_index = torch.zeros_like(x, dtype=torch.bool).scatter_(1, select_indices, True)

                            if remasking_strategy == "dlcr":
                                confidence_threshold = 0.9
                                high_confidence_mask = confidence > confidence_threshold
                                # ALso transfer tokens with confidence above threshold
                                transfer_index = transfer_index | (high_confidence_mask & mask_index)

                            if resample:
                                logits_with_noise = add_gumbel_noise(logits, temperature=args.eval_temperature)
                                x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

                            x[transfer_index] = x0[transfer_index]

                    outputs = x

                generate_one = generate_block if use_block_gen else generate_ntp

                if model_dtype == "float32":
                    generate_one()
                else:
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        generate_one()

                # Store local outputs before gather for semantic variance calculation
                local_outputs = outputs[:, args.eval_prompt_length:].cpu().numpy().tolist()
                local_contents.extend(local_outputs)

                if accelerator:
                    outputs = accelerator.gather(outputs)
                contents.extend(outputs[:, args.eval_prompt_length:].cpu().numpy().tolist())

        if args.eval_ce:
            if ref_model is None:
                ref_model = model
            sum_ref_logp = torch.zeros(1, device=ref_model.device)
            idx = 0
            for batch in tqdm(eval_dataloader, desc="Evaluating cross entropy", disable=not accelerator.is_local_main_process if accelerator else False):
                input_ids = batch["raw_input_ids"].to(ref_model.device)
                batch_size = input_ids.size(0)

                # reconstruct outputs using local_contents
                outputs = input_ids.clone()
                batch_contents = local_contents[idx:idx + batch_size]
                idx += batch_size
                block_len = len(batch_contents[0])
                outputs[:, args.eval_prompt_length:args.eval_prompt_length + block_len] = torch.tensor(batch_contents, device=ref_model.device)

                curr_input_ids = outputs.clone()
                curr_input_ids[:, args.eval_prompt_length:] = tokenizer.mask_token_id

                cur_ref_logp = torch.zeros(batch_size, device=ref_model.device)
                n_token = torch.zeros(batch_size, device=ref_model.device)
                stopped = torch.zeros(batch_size, dtype=torch.bool, device=ref_model.device)
                full_attn_mask = torch.ones((batch_size, 1, max_length, max_length), dtype=torch.bool).to(ref_model.device)

                def compute_ce_step():
                    nonlocal cur_ref_logp, n_token, stopped
                    for step in range(args.eval_prompt_length, outputs.size(1)):
                        target_tokens = outputs[:, step]

                        outputs2 = ref_model(input_ids=curr_input_ids, attention_mask=full_attn_mask, is_causal=False)
                        if context_scope == "uc":
                            next_token_logits = outputs2.logits[:, step - 1, :]
                        else:
                            next_token_logits = outputs2.logits[:, step, :]
                        next_token_logits_scaled = next_token_logits / args.eval_temperature

                        if not args.eval_allow_eos:
                            next_token_logits_scaled[:, tokenizer.eos_token_id] = -1e10

                        probs = F.softmax(next_token_logits_scaled, dim=-1)
                        if step > 0:
                            stopped |= curr_input_ids[:, step - 1] == tokenizer.pad_token_id
                        curr_input_ids[:, step] = target_tokens
                        cur_ref_logp += torch.log(torch.gather(probs, dim=1, index=target_tokens.unsqueeze(-1).long()).squeeze(-1)) * (1 - stopped.float())
                        n_token += 1 - stopped.float()

                if model_dtype == "float32":
                    compute_ce_step()
                else:
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        compute_ce_step()

                n_token = torch.clamp(n_token, min=1.0)
                sum_ref_logp += (cur_ref_logp / n_token).sum()

        if args.eval_ce:
            if sum_ref_logp is not None and accelerator:
                sum_ref_logp = accelerator.reduce(sum_ref_logp, reduction="sum")
            cross_entropy = -sum_ref_logp.item() / len(eval_dataloader.dataset)

        semantic_variance_metrics = {}
        local_stats = compute_semantic_variances_local(contents=local_contents, tokenizer=tokenizer, embedding_models_dict=embedding_models_dict, accelerator=accelerator, gather_embeddings=verbose)

        for model_name, stats in local_stats.items():
            adjacent_sentence_similarity_sum = torch.tensor(stats["adjacent_sentence_similarity_sum"], device=model.device)
            adjacent_sentence_similarity_count = torch.tensor(stats["adjacent_sentence_similarity_count"], device=model.device, dtype=torch.long)
            inter_embeddings_sum = torch.tensor(stats["inter_embeddings_sum"], device=model.device)
            inter_embeddings_squared_sum = torch.tensor(stats["inter_embeddings_squared_sum"], device=model.device)
            inter_embeddings_count = torch.tensor(stats["inter_embeddings_count"], device=model.device, dtype=torch.long)

            if accelerator:
                adjacent_sentence_similarity_sum = accelerator.reduce(adjacent_sentence_similarity_sum, reduction="sum")
                adjacent_sentence_similarity_count = accelerator.reduce(adjacent_sentence_similarity_count, reduction="sum")
                inter_embeddings_sum = accelerator.reduce(inter_embeddings_sum, reduction="sum")
                inter_embeddings_squared_sum = accelerator.reduce(inter_embeddings_squared_sum, reduction="sum")
                inter_embeddings_count = accelerator.reduce(inter_embeddings_count, reduction="sum")

            if not accelerator or accelerator.is_main_process:
                adjacent_sentence_similarity_count = adjacent_sentence_similarity_count.clamp(min=1)
                inter_embeddings_count = inter_embeddings_count.clamp(min=1)

                adjacent_sentence_similarity = adjacent_sentence_similarity_sum.item() / adjacent_sentence_similarity_count.item()
                semantic_variance_metrics[f"semantic_coherence({model_name})"] = adjacent_sentence_similarity

                global_mean = inter_embeddings_sum.cpu().numpy() / inter_embeddings_count.item()
                global_mean_squared = global_mean**2
                global_squared_mean = inter_embeddings_squared_sum.cpu().numpy() / inter_embeddings_count.item()
                inter_variance = np.sum(global_squared_mean - global_mean_squared)
                semantic_variance_metrics[f"semantic_diversity({model_name})"] = float(inter_variance)

            if verbose:
                cur_embeddings = stats["full_embeddings"]
                if accelerator:
                    # gather from all processes
                    # each cur_embeddings is np array of shape (num_seqs_local, emb_dim)
                    cur_embeddings = accelerator.gather_for_metrics(torch.tensor(cur_embeddings, device=model.device)).cpu().numpy()
                embeddings[model_name] = cur_embeddings

        if not accelerator or accelerator.is_main_process:
            metrics, _, info = compute_metrics(contents=contents, tokenizer=tokenizer)
            print(f"Extra Info (compute_metrics): {info}")
            print(f"Example generated sequences: {tokenizer.decode(contents[0])}")
        else:
            metrics = {}

    res = {}
    if args.eval_ce:
        res["cross_entropy"] = cross_entropy

    if args.eval_ntp_loss_and_entropy and not external and not return_logits:
        loss_entropy_metrics = evaluate_ntp_loss_and_entropy(model, tokenizer, eval_dataloader, accelerator)
        res.update(loss_entropy_metrics)
    res.update(metrics)
    res.update(semantic_variance_metrics)
    print(f"Evaluation results: {res}")
    res = {f"eval/{k}": v for k, v in res.items()}
    if verbose:
        extra = {
            "embeddings": embeddings,
            # "prefix_ntp": prefix_ntp,
        }
        if return_logits:
            extra["logits"] = all_logits
        return contents, res, extra
    if return_logits:
        return contents, res, all_logits
    return contents, res
