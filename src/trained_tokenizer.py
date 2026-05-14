from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast


def get_tokenizer(dataset, vocab_size=10000, output_path=".", **kwargs):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    ds_first_1m = dataset.select(range(min(1_000_000, len(dataset))))
    tokenizer.train_from_iterator((x["text"] for x in ds_first_1m), trainer=trainer)
    # tokenizer.save(output_file)
    # print(f"Tokenizer saved to {output_file}")

    # Save as PreTrainedTokenizerFast to ensure compatibility
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )
    fast_tokenizer.eos_token = "[PAD]"
    fast_tokenizer.save_pretrained(output_path)
    print(f"Tokenizer saved to {output_path}")
