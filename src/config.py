import dotenv

dotenv.load_dotenv()

from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List
from transformers import HfArgumentParser
import random
import torch
import numpy as np
import hashlib
import json
import time
import os
import copy


@dataclass
class CoreVariables:
    context_scope: str = field(default="bc", metadata={"help": "Context scope to use", "choices": ["uc", "bc"], "aliases": ["--context_scope", "--context-scope"]})
    input_masking: Optional[bool] = field(default=None, metadata={"help": "Input masking component in the training objective.", "aliases": ["--im"]})
    label_masking: Optional[bool] = field(default=None, metadata={"help": "Label masking component in the training objective.", "aliases": ["--lm"]})
    weighting_function: Optional[bool] = field(default=None, metadata={"help": "Weighting function component in the training objective.", "aliases": ["--wf"]})
    remasking_strategy: Optional[str] = field(default="rr", metadata={"help": "Remasking strategy for each epoch. Use abbreviations: rr - random remasking, lcr - low confidence remasking, dlcr - dynamic low confidence remasking, er - entropy remasking", "choices": ["rr", "lcr", "dlcr", "er"]})
    block_length: int = field(default=1, metadata={"help": "Block length. 1 means autoregressive.", "aliases": ["--bl"]})
    external_generation: Optional[str] = field(default=None, metadata={"help": "Whether to use generated content from an external model for evaluation. Format: generation json file path"})
    resample: bool = field(default=False, metadata={"help": "Whether to resample the token at each inference step", "aliases": ["--rs"]})


@dataclass
class ScriptArguments:
    name: Optional[str] = field(default=None, metadata={"help": "Name for the experiment."})
    output_dir: str = field(default="./workdir/train", metadata={"help": "The output directory where the results will be written.", "aliases": ["-o"]})
    force: bool = field(default=False, metadata={"help": "If true, overwrite the output directory.", "aliases": ["-f"]})
    eval_ntp_loss_and_entropy: bool = field(default=True, metadata={"help": "If true, evaluate next-token prediction loss and entropy."})
    eval_ce: bool = field(default=False, metadata={"help": "If true, evaluate cross entropy. Use with bc context scope."})
    args_hash: Optional[str] = field(default=None, metadata={"help": "Hash of the arguments for this run."})
    run_start_time: Optional[str] = field(default=None, metadata={"help": "Start time of the run."})

    core: CoreVariables = field(default_factory=CoreVariables)

    dataset: str = field(default="fineweb", metadata={"help": "Dataset type to use", "choices": ["tinystories","fineweb"]})
    tokenizer: str = field(default="trained", metadata={"help": "Tokenizer type to use", "choices": ["trained"]})
    model: str = field(default="llama", metadata={"help": "Context scope to use", "choices": ["qwen2", "llama"]})
    dtype: str = field(default="bfloat16", metadata={"help": "Data type for training", "choices": ["float16", "bfloat16", "float32"]})
    flash_attention: bool = field(default=True, metadata={"help": "If true, use flash attention."})
    num_samples: Optional[int] = field(default=4000000, metadata={"help": "Number of samples."})
    vocab_size: int = field(default=5000, metadata={"help": "Target vocabulary size (may not be exact)."})
    max_length: int = field(default=512, metadata={"help": "Chunk size."})
    num_epochs: int = field(default=1, metadata={"help": "Total number of training epochs to perform."})
    batch_size: int = field(default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    eval_batch_size: int = field(default=32, metadata={"help": "Batch size for evaluation."})
    global_batch_size: int = field(default=1024, metadata={"help": "Total batch size across all GPUs and accumulation steps."})
    num_warmup_steps: int = field(default=100, metadata={"help": "Number of steps for the warmup in the lr scheduler."})
    learning_rate: float = field(default=1e-3, metadata={"help": "The learning rate for Adam."})
    gradient_clip_norm: Optional[float] = field(default=1.0, metadata={"help": "Gradient clip norm."})
    optimizer_cls: str = field(default="adamw", metadata={"help": "Optimizer to use", "choices": ["adamw", "sgd"]})
    eval_num_samples: int = field(default=512, metadata={"help": "Number of sequences for evaluation."})
    eval_temperature: float = field(default=1.0, metadata={"help": "Temperature for evaluation."})
    eval_prompt_length: int = field(default=32, metadata={"help": "Prompt length for evaluation."})
    eval_max_num_grams: int = field(default=0, metadata={"help": "Maximum number of n-grams to consider during evaluation. 0 means all."})
    eval_allow_eos: bool = field(default=False, metadata={"help": "Allow EOS token during evaluation generation."})
    save_epoch: bool = field(default=True, metadata={"help": "If true, save model at each epoch."})
    save_step: int = field(default=50, metadata={"help": "If > 0, save model every save_step steps."})
    eval_epoch: bool = field(default=True, metadata={"help": "If true, evaluate model at each epoch."})
    eval_step: int = field(default=100, metadata={"help": "If > 0, evaluate model every eval_step steps."})
    seed: int = field(default=0, metadata={"help": "Random seed for initialization."})
    embedding_models: List[str] = field(default_factory=lambda: ["gemma=google/embeddinggemma-300m", "qwen3-0.6B=Qwen/Qwen3-Embedding-0.6B", "minilm=sentence-transformers/all-MiniLM-L6-v2", "bge=BAAI/bge-m3"], metadata={"help": "List of embedding model names for semantic variance calculation. Format: model_name=model_path"})
    noise_min: float = field(default=1e-3, metadata={"help": "Minimum noise level for t sampling."})
    noise_max: float = field(default=1.0, metadata={"help": "Maximum noise level for t sampling."})


parser = HfArgumentParser((ScriptArguments, CoreVariables))
args, core_args = parser.parse_args_into_dataclasses()
args.core = core_args


def make_core_from_context_scope(scope: str) -> CoreVariables:
    valid_scopes = {
        "uc",
        "uc+im",
        "uc+lm",
        "uc+lm+wf",
        "uc+im+lm",
        "uc+im+lm+wf",
        "bc+im+lm",
        "bc+im+lm+wf",
    }
    assert scope in valid_scopes, f"Unsupported context scope: {scope}"

    context_scope = "bc" if scope.startswith("bc") else "uc"
    core = CoreVariables(context_scope=context_scope)
    core.input_masking = "+im" in scope
    core.label_masking = "+lm" in scope
    core.weighting_function = "+wf" in scope
    return core


if os.environ.get("RESUME_ARGS", "False") in ["1", "true", "True"]:
    if os.path.exists(args.output_dir):
        candidate_models = []
        for name in os.listdir(args.output_dir):
            model_dir = os.path.join(args.output_dir, name)
            config_path = os.path.join(model_dir, "config.json")
            if (
                os.path.isdir(model_dir)
                and (not name.startswith("decode"))
                and (not name.startswith("."))
                and name != "flags"
                and os.path.isfile(config_path)
            ):
                candidate_models.append(name)

        if candidate_models:
            # Deterministic resume source to avoid random os.listdir order.
            first_model = sorted(candidate_models)[0]
            first_config_path = os.path.join(args.output_dir, first_model, "config.json")
            print(f"Resuming config from {first_config_path}...")
            with open(first_config_path, "r", encoding="utf-8") as f:
                base_cfg = json.load(f)

            if "core" not in base_cfg:
                raise ValueError("Missing 'core' in config.json. Please use checkpoints saved with canonical core fields.")

            core_dict = base_cfg.pop("core")
            valid_core_keys = {f.name for f in fields(CoreVariables)}
            core_dict = {k: v for k, v in core_dict.items() if k in valid_core_keys}
            args.core = CoreVariables(**core_dict)

            valid_keys = {f.name for f in fields(ScriptArguments)}
            valid_keys.remove("core")
            valid_keys.remove("output_dir")
            for k, v in base_cfg.items():
                if k in valid_keys:
                    setattr(args, k, v)
output_dir = args.output_dir
seed = args.seed
batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
global_batch_size = args.global_batch_size
num_samples = args.num_samples
vocab_size = args.vocab_size
max_length = args.max_length
num_epochs = args.num_epochs
learning_rate = args.learning_rate
num_warmup_steps = args.num_warmup_steps
gradient_clip_norm = args.gradient_clip_norm
eval_num_samples = args.eval_num_samples
optimizer_cls = args.optimizer_cls
model = args.model
model_dtype = args.dtype


def seed_all(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_all(seed)


if args.dtype == "float32":
    torch_dtype = torch.float32
elif args.dtype == "bfloat16":
    torch_dtype = torch.bfloat16
elif args.dtype == "float16":
    torch_dtype = torch.float16
else:
    raise ValueError("Unsupported dtype")


if args.dataset == "tinystories":
    from tinystories_data import get_dataset
elif args.dataset == "fineweb":
    from fineweb_data import get_dataset
else:
    raise ValueError("Unsupported dataset type")


if args.tokenizer == "trained":
    from trained_tokenizer import get_tokenizer
else:
    raise ValueError("Unsupported tokenizer type")


if len(args.embedding_models) == 1 and args.embedding_models[0] == "None":
    args.embedding_models = []

args_dict = asdict(args)
args_dict.pop("name", None)
args_dict.pop("output_dir", None)
args_dict.pop("force", None)
args_dict.pop("eval_ntp_loss_and_entropy", None)
args_dict.pop("args_hash", None)
args_dict.pop("run_start_time", None)
args_dict.pop("core", None)
args_hash = hashlib.md5(json.dumps(args_dict, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:8]
args.args_hash = args_hash
args.run_start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


eos_tokens_list = [
    ".", ". ", " .", " . ",
    "。", "。 ", " 。", " 。 ",
    "?", "? ", " ?", " ? ",
    "？", "？ ", " ？", " ？ ",
    "!", "! ", " !", " ! ",
    "！", "！ ", " ！", " ！ ",
]
