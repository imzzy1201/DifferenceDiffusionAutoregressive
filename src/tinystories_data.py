import datasets


def get_dataset(num_samples=1_000_000, max_length=256, split: str = "train", seed=None, **kwargs):
    assert split in ["train", "eval"], "split must be 'train' or 'eval'"
    ds = datasets.load_dataset("roneneldan/TinyStories", split="train" if split == "train" else "validation")
    ds = ds.filter(lambda example: len(example["text"]) > 512)
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if num_samples is not None:
        ds = ds.select(range(num_samples))
    return ds
