"""
Run python fineweb_data.py first to preprocess and save the dataset to ./temp_datasets/fineweb/
Then use get_dataset to load the dataset for training/evaluation.
"""

import os
import datasets
from pathlib import Path

# disable datasets caching
from datasets import disable_caching
disable_caching()


def get_dataset(num_samples=1_000_000, max_length=256, split: str = "train", seed=None, **kwargs):
    assert split in ["train", "eval"], "split must be 'train' or 'eval'"
    base_dir = Path("./temp_datasets/fineweb")
    train_shards_dir = base_dir / "train_shards"

    if split == "train" and train_shards_dir.exists():
        shard_dirs = sorted([p for p in train_shards_dir.iterdir() if p.is_dir()])
        if not shard_dirs:
            raise FileNotFoundError(f"No train shards found under {train_shards_dir}")
        ds_list = [datasets.load_from_disk(str(p)) for p in shard_dirs]
        ds = datasets.concatenate_datasets(ds_list)
    else:
        ds = datasets.load_from_disk(str(base_dir / split))

    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if num_samples is not None:
        ds = ds.select(range(num_samples))
    return ds


def preprocess_dataset(num_train_samples=3_000_000, num_eval_samples=5_000, shard_size=200_000, **kwargs):
    from modelscope.msdatasets import MsDataset
    import tqdm

    raw_ds = MsDataset.load('AI-ModelScope/fineweb_deduplicated', split='train', use_streaming=True)

    base_dir = Path("./temp_datasets/fineweb")
    train_shards_dir = base_dir / "train_shards"
    train_shards_dir.mkdir(parents=True, exist_ok=True)

    train_ds_list = []
    eval_ds_list = []
    shard_idx = 0

    pbar = tqdm.tqdm(total=num_train_samples + num_eval_samples)
    for i, example in enumerate(raw_ds):
        if example["token_count"] > 512:
            if len(eval_ds_list) < num_eval_samples:
                eval_ds_list.append(example)
                pbar.update(1)
            elif len(train_ds_list) < num_train_samples:
                train_ds_list.append(example)
                pbar.update(1)

            if len(train_ds_list) >= shard_size:
                shard_path = train_shards_dir / f"shard_{shard_idx:05d}"
                datasets.Dataset.from_list(train_ds_list).save_to_disk(str(shard_path))
                shard_idx += 1
                train_ds_list = []

        if (shard_idx * shard_size + len(train_ds_list)) >= num_train_samples and len(eval_ds_list) >= num_eval_samples:
            break

    if train_ds_list:
        shard_path = train_shards_dir / f"shard_{shard_idx:05d}"
        datasets.Dataset.from_list(train_ds_list).save_to_disk(str(shard_path))

    eval_ds = datasets.Dataset.from_list(eval_ds_list)
    os.makedirs(str(base_dir), exist_ok=True)
    eval_ds.save_to_disk(str(base_dir / "eval"))

    print("Datasets saved to ./temp_datasets/fineweb/ (train_shards + eval)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess the fineweb dataset")
    parser.add_argument("--num_train_samples", type=int, default=10_000_000, help="Number of training samples to preprocess")
    parser.add_argument("--num_eval_samples", type=int, default=5_000, help="Number of evaluation samples to preprocess")
    args = parser.parse_args()
    preprocess_dataset(num_train_samples=args.num_train_samples, num_eval_samples=args.num_eval_samples)
