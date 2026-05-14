from config import *
import models

import csv
import json
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


BASE_MODEL_TYPE = "bc+im+lm+wf"
BLOCK_LENGTHS = [2, 8, 32]
ALL_REMASK_STRATEGIES = ["rr", "lcr", "dlcr", "er"]
CONFIDENCE_REMASK_STRATEGIES = ["lcr", "dlcr"]


def get_checkpoint_sort_key(ckpt_name: str):
	if ckpt_name.startswith("checkpoint-step-"):
		return int(ckpt_name.split("-")[-1])
	if ckpt_name.startswith("checkpoint-epoch-"):
		return int(ckpt_name.split("-")[-1])
	return 0


def get_step_from_state(model_dir: str, ckpt_name: str):
	if "checkpoint-eval" in ckpt_name:
		return float("inf")
	ckpt_path = os.path.join(model_dir, ckpt_name)
	state_path = os.path.join(ckpt_path, "trainer_state.json")
	fallback = get_checkpoint_sort_key(ckpt_name)
	if os.path.exists(state_path):
		try:
			with open(state_path, "r", encoding="utf-8") as f:
				state = json.load(f)
			return state.get("wandb_step", fallback)
		except Exception:
			return fallback
	return fallback


def build_decode_variants():
	variants = []

	# Baseline decode setting.
	variants.append(
		CoreVariables(
			context_scope="bc",
			input_masking=True,
			label_masking=True,
			weighting_function=True,
			block_length=1,
			remasking_strategy="rr",
			resample=False,
		)
	)

	# block_length in {2, 8, 32} x remask in {rr, lcr, dlcr, er}.
	for block_length in BLOCK_LENGTHS:
		for remasking_strategy in ALL_REMASK_STRATEGIES:
			variants.append(
				CoreVariables(
					context_scope="bc",
					input_masking=True,
					label_masking=True,
					weighting_function=True,
					block_length=block_length,
					remasking_strategy=remasking_strategy,
					resample=False,
				)
			)

	# block_length in {2, 8, 32} x confidence remask in {lcr, dlcr} with resample.
	for block_length in BLOCK_LENGTHS:
		for remasking_strategy in CONFIDENCE_REMASK_STRATEGIES:
			variants.append(
				CoreVariables(
					context_scope="bc",
					input_masking=True,
					label_masking=True,
					weighting_function=True,
					block_length=block_length,
					remasking_strategy=remasking_strategy,
					resample=True,
				)
			)

	return variants


def configure_tokenizer(tokenizer_path: str):
	tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.mask_token = "[MASK]"
	tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
	return tokenizer


def load_embedding_models(accelerator: Accelerator):
	embedding_models_dict = {}
	if args.embedding_models:
		if accelerator.is_main_process:
			print(f"Loading embedding models: {args.embedding_models}")
		for model_config in args.embedding_models:
			model_name, model_path = model_config.split("=", 1)
			embedding_models_dict[model_name] = SentenceTransformer(model_path)
			if accelerator.is_main_process:
				print(f"Loaded embedding model: {model_name} from {model_path}")
		accelerator.wait_for_everyone()
	return embedding_models_dict


def save_results(decode_dir: str, rows: list, metadata: dict):
	csv_path = os.path.join(decode_dir, f"decode_metrics.csv")
	meta_path = os.path.join(decode_dir, f"decode_config.json")

	field_names = sorted({k for row in rows for k in row.keys()})
	if "model_type" in field_names:
		field_names.remove("model_type")
	field_names = ["model_type"] + field_names
	with open(csv_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=field_names)
		writer.writeheader()
		writer.writerows(rows)

	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2, ensure_ascii=True)

	print(f"Saved decode metrics CSV to: {csv_path}")
	print(f"Saved decode config to: {meta_path}")


def main():
	accelerator = Accelerator()

	args.eval_ntp_loss_and_entropy = False
	args.eval_ce = True

	decode_dir = os.path.join(args.output_dir, "decode")
	if accelerator.is_main_process:
		os.makedirs(decode_dir, exist_ok=True)
	accelerator.wait_for_everyone()

	base_model_dir = os.path.join(args.output_dir, BASE_MODEL_TYPE)
	if not os.path.isdir(base_model_dir):
		raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")

	ckpt_names = [
		name
		for name in os.listdir(base_model_dir)
		if name.startswith("checkpoint-") and os.path.isdir(os.path.join(base_model_dir, name))
	]
	if not ckpt_names:
		raise FileNotFoundError(f"No checkpoints found in: {base_model_dir}")
	ckpt_names.sort(key=lambda name: get_step_from_state(base_model_dir, name))
	latest_ckpt_name = ckpt_names[-1]
	latest_ckpt_path = os.path.join(base_model_dir, latest_ckpt_name)

	if accelerator.is_main_process:
		print(f"Using base model: {BASE_MODEL_TYPE}")
		print(f"Using checkpoint: {latest_ckpt_path}")

	tokenizer = configure_tokenizer(base_model_dir)

	eval_dataset = get_dataset(
		num_samples=eval_num_samples,
		max_length=max_length,
		split="eval",
		vocab_size=vocab_size,
		seed=seed + 10000,
	)
	data_collator = models.DataCollator(tokenizer=tokenizer)
	eval_dataloader = DataLoader(
		eval_dataset,
		collate_fn=data_collator,
		batch_size=eval_batch_size,
		drop_last=False,
		num_workers=4,
		shuffle=False,
	)

	model = AutoModelForCausalLM.from_pretrained(latest_ckpt_path, torch_dtype=torch_dtype)
	model = model.to(accelerator.device)
	model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
	model.eval()

	embedding_models_dict = load_embedding_models(accelerator)
	variants = build_decode_variants()

	if accelerator.is_main_process:
		print(f"Evaluating {len(variants)} decode variants...")

	rows = []
	for i, core in enumerate(variants, start=1):
		models.init_context_scope(asdict(core))
		if accelerator.is_main_process:
			print(f"[{i}/{len(variants)}] Evaluating {models.core_str}")

		_, metrics = models.evaluate(
			model=model,
			tokenizer=tokenizer,
			eval_dataloader=eval_dataloader,
			accelerator=accelerator,
			embedding_models_dict=embedding_models_dict,
			verbose=False,
			ref_model=model,
		)

		if accelerator.is_main_process:
			row = {"model_type": models.core_str}
			row.update(metrics)
			rows.append(row)

	if accelerator.is_main_process:
		metadata = {
			"base_model_type": BASE_MODEL_TYPE,
			"checkpoint": latest_ckpt_path,
			"output_dir": args.output_dir,
			"decode_dir": decode_dir,
			"num_variants": len(variants),
			"variants": [asdict(v) for v in variants],
			"metrics": [
				"eval/1_gram_entropy",
				"eval/2_gram_entropy",
				"eval/3_gram_entropy",
				"eval/cross_entropy",
				"eval/semantic_coherence(<embedding_model>)",
				"eval/semantic_diversity(<embedding_model>)",
			],
		}
		save_results(decode_dir, rows, metadata)


if __name__ == "__main__":
	main()
