from config import *
from models import *
import os
import time
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, get_constant_schedule_with_warmup
import json
import wandb
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
from dataclasses import asdict


print("Done imports.")


def train():
    accelerator = Accelerator()

    # Training uses NTP loss/entropy + generation metrics, not cross-entropy.
    args.eval_ntp_loss_and_entropy = True
    args.eval_ce = False

    total_batch_size_per_step = batch_size * accelerator.num_processes
    if global_batch_size % total_batch_size_per_step != 0:
        raise ValueError(f"Global batch size ({global_batch_size}) must be divisible by " f"batch_size * num_processes ({batch_size} * {accelerator.num_processes} = {total_batch_size_per_step}).")
    gradient_accumulation_steps = global_batch_size // total_batch_size_per_step
    accelerator.gradient_accumulation_steps = gradient_accumulation_steps

    # if accelerator.is_main_process:
    #     print(f"Calculated gradient accumulation steps: {gradient_accumulation_steps}")

    model_dir = os.path.join(output_dir, core_str)
    if accelerator.is_main_process:
        if os.path.exists(model_dir):
            if args.force:
                print("[WARNING] Overwriting existing output directory.")
            else:
                yn = input("Output directory exists. Overwrite? (y/n): ")
                if yn.lower() != "y":
                    print("Exiting...")
                    return
        os.makedirs(model_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        run_name = args.name
        if run_name is None:
            run_name = f"{core_str}_{args.dataset}_{args.model}_{args.args_hash}"
        wandb.init(entity="DandA", project="control_exp_train", name=run_name, config=asdict(args))
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(asdict(args), f, indent=4)

        log_path = os.path.join(model_dir, "train.jsonl")
        with open(log_path, "w") as f:
            pass

    train_dataset = get_dataset(num_samples=num_samples, max_length=max_length, split="train", vocab_size=vocab_size, seed=seed)
    eval_dataset = get_dataset(num_samples=eval_num_samples, max_length=max_length, split="eval", vocab_size=vocab_size, seed=seed + 10000)

    # Ensure no padding/dropping needed for perfect alignment
    # if len(train_dataset) % (batch_size * accelerator.num_processes) != 0:  # This should not be necessary
    #     raise ValueError(f"Train dataset size ({len(train_dataset)}) must be divisible by " f"batch_size * num_processes ({batch_size} * {accelerator.num_processes}).")
    if len(eval_dataset) % (eval_batch_size * accelerator.num_processes) != 0:
        raise ValueError(
            f"Eval dataset size ({len(eval_dataset)}) must be divisible by "
            f"batch_size * num_processes ({eval_batch_size} * {accelerator.num_processes})."
        )

    # 1. Initialize Tokenizer (Generate once)
    if accelerator.is_main_process:
        print(f"Generating tokenizer with target vocab_size={vocab_size}...")
        get_tokenizer(dataset=train_dataset, vocab_size=vocab_size, output_path=model_dir)
    accelerator.wait_for_everyone()

    # Wrap in Transformers tokenizer
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    fast_tokenizer.pad_token = fast_tokenizer.eos_token
    fast_tokenizer.pad_token_id = fast_tokenizer.eos_token_id
    fast_tokenizer.mask_token = "[MASK]"
    fast_tokenizer.mask_token_id = fast_tokenizer.convert_tokens_to_ids("[MASK]")

    if not args.eval_allow_eos:
        # Filter out sequences containing EOS
        def filter_eos(example):
            ids = fast_tokenizer.encode(example["text"], padding="max_length", truncation=True, max_length=args.max_length)
            return fast_tokenizer.eos_token_id not in ids
        # print("Filtering out sequences containing EOS token from datasets...")
        # train_dataset = train_dataset.filter(filter_eos)
        print("WARNING: disabled EOS filtering")
    print(f"Dataset sizes: train={len(train_dataset)}, eval={len(eval_dataset)}")

    data_collator = DataCollator(tokenizer=fast_tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        shuffle=False,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=eval_batch_size,
        drop_last=True,
        num_workers=4,
        shuffle=False,
    )

    model = init_model(fast_tokenizer)

    # 4. Manual Training Loop
    device = accelerator.device
    print(f"Using device: {device}")
    model.to(device)

    # steps_per_epoch = num_samples // batch_size
    if optimizer_cls == "adamw":
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_cls == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_cls}")

    # Calculate num_training_steps correctly based on global_batch_size
    # num_training_steps = (len(train_dataset) * num_epochs) // global_batch_size

    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        # num_training_steps=num_training_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    # Load embedding models for semantic variance calculation
    embedding_models_dict = {}
    if args.embedding_models:
        if accelerator.is_main_process:
            print(f"Loading embedding models: {args.embedding_models}")
        for model_config in args.embedding_models:
            model_name, model_path = model_config.split("=", 1)
            embedding_models_dict[model_name] = SentenceTransformer(model_path)
            if accelerator.is_main_process:
                print(f"Loaded embedding model: {model_name} from {model_path}")
        accelerator.wait_for_everyone()  # Ensure all processes have loaded models

    model.train()
    print("Starting training...")
    num_trained_tokens = 0
    step = 0
    start_time = time.perf_counter()
    total_steps = math.ceil(num_epochs * len(train_dataloader) / gradient_accumulation_steps)

    def save_checkpoint(ckpt_dir: str, wandb_step: int, metrics: dict):
        accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
        fast_tokenizer.save_pretrained(ckpt_dir)
        with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
            json.dump({"wandb_step": wandb_step, "metrics": metrics}, f, indent=4)

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}/{num_epochs}...")

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=not accelerator.is_local_main_process)
        optimizer.zero_grad()
        epoch_loss = 0.0
        eta_minutes = 0.0

        for it, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                num_trained_tokens += batch['input_ids'].numel()

                if model_dtype == "float32":
                    loss = train_batch(model=model, batch=batch, accelerator=accelerator)
                else:
                    with torch.autocast(device_type="cuda", dtype=torch_dtype):
                        loss = train_batch(model=model, batch=batch, accelerator=accelerator)

                if accelerator.sync_gradients:
                    if gradient_clip_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), gradient_clip_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    log_msg = {"train/epoch": epoch + it / len(train_dataloader), "train/loss": loss, "train/lr": optimizer.param_groups[0]["lr"], "train/num_trained_tokens": num_trained_tokens * accelerator.num_processes}
                    step += 1
                    elapsed = time.perf_counter() - start_time
                    avg_step_time = elapsed / max(step, 1)
                    remaining_steps = max(total_steps - step, 0)
                    eta_minutes = (remaining_steps * avg_step_time) / 60.0
                    if args.save_step > 0 and step % args.save_step == 0:
                        if accelerator.is_main_process:
                            print(f"Saving checkpoint at step {step}...")
                            step_dir = os.path.join(model_dir, f"checkpoint-step-{step}")
                            save_checkpoint(step_dir, wandb_step=wandb.run.step, metrics=log_msg)
                    if args.eval_step > 0 and step % args.eval_step == 0:
                        if accelerator.is_main_process:
                            print("Evaluating model...")
                        model.eval()
                        contents, res = evaluate(model, fast_tokenizer, eval_dataloader, accelerator, embedding_models_dict)
                        model.train()
                        log_msg.update(res)
                        if accelerator.is_main_process:
                            with open(log_path, "a") as f:
                                json.dump(log_msg, f, ensure_ascii=True)
                                f.write("\n")

                    if accelerator.is_main_process:
                        wandb.log(log_msg)

            epoch_loss += loss
            progress_bar.set_postfix(loss=loss, eta_min=f"{eta_minutes:.1f}")

        avg_loss = epoch_loss / len(train_dataloader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.10f}")

        log_msg = {"train/epoch": epoch + 1, "train/loss": avg_loss, "train/lr": optimizer.param_groups[0]["lr"], "train/num_trained_tokens": num_trained_tokens * accelerator.num_processes}
        if args.save_epoch:
            if accelerator.is_main_process:
                print(f"Saving checkpoint for epoch {epoch + 1}...")
                epoch_dir = os.path.join(model_dir, f"checkpoint-epoch-{epoch + 1}")
                save_checkpoint(epoch_dir, wandb_step=wandb.run.step, metrics=log_msg)
        if args.eval_epoch:
            # Evaluation at the end of epoch
            if accelerator.is_main_process:
                print("Evaluating model...")
            model.eval()
            contents, res = evaluate(model, fast_tokenizer, eval_dataloader, accelerator, embedding_models_dict)
            model.train()
            log_msg.update(res)
            if accelerator.is_main_process:
                with open(log_path, "a") as f:
                    json.dump(log_msg, f, ensure_ascii=True)
                    f.write("\n")
        if accelerator.is_main_process:
            wandb.log(log_msg)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()
