from dataclasses import dataclass
from typing import Optional

import verifiers as vf

ENDPOINTS = {
    "gemini-2.5-flash": {
        "model": "google/gemini-2.5-flash",
        "url": "https://openrouter.ai/api/v1",
        "key": "OPENAI_API_KEY",
    },
    "gemini-2.5-flash-lite": {
        "model": "google/gemini-2.5-flash-lite",
        "url": "https://openrouter.ai/api/v1",
        "key": "OPENAI_API_KEY",
    },
    "qwen3-a22b": {
        "model": "qwen/qwen3-235b-a22b-2507",
        "url": "https://openrouter.ai/api/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-4.1": {
        "model": "openai/gpt-4.1",
        "url": "https://openrouter.ai/api/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-4.1-mini": {
        "model": "openai/gpt-4.1-mini",
        "url": "https://openrouter.ai/api/v1",
        "key": "OPENAI_API_KEY",
    },
    "gpt-4.1-nano": {
        "model": "openai/gpt-4.1-nano",
        "url": "https://openrouter.ai/api/v1",
        "key": "OPENAI_API_KEY",
    },
    "deepseek-v3": {
        "model": "deepseek/deepseek-chat-v3-0324",
        "url": "https://openrouter.ai/api/v1",
        "key": "OPENAI_API_KEY",
    },
}


@dataclass
class TrainConfig:
    # General / environment
    run_name: str = "bm25s"
    env_id: str = "ragent.environments.bm25s"
    hf_dataset: str = "nampdn-ai/devdocs.io"
    llm_name_judge: str = "google/gemini-2.5-flash"
    model_name: str = "Qwen/Qwen2.5-0.5B"

    # Core batch settings
    per_device_train_batch_size: int = 8
    num_generations: int = 16
    gradient_accumulation_steps: int = 4

    # Sampling configuration
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None

    # Length limits
    max_prompt_length: int = 1024
    max_completion_length: int = 2048
    max_seq_len: int = 131072
    mask_truncated_completions: bool = True

    # Optimization settings
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = 10
    max_steps: int = 500
    num_iterations: int = 1

    # Gradient control
    max_grad_norm: float = 0.01

    # KL regularization
    beta: float = 0.001
    sync_ref_model: bool = True
    ref_model_sync_steps: int = 100
    ref_model_mixup_alpha: float = 0.5

    # Loss configuration
    loss_type: str = "dr_grpo"
    epsilon: float = 0.2
    delta: Optional[float] = None

    # Overlapped training and inference
    num_batches_ahead: int = 1
    async_generation_timeout: float = 300.0
    max_concurrent: int = 1024

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    per_device_eval_batch_size: int = 16


def prepare_args(cfg: TrainConfig):
    args = vf.grpo_defaults(run_name=cfg.run_name)

    # Core batch settings
    args.per_device_train_batch_size = cfg.per_device_train_batch_size
    args.num_generations = cfg.num_generations
    args.gradient_accumulation_steps = cfg.gradient_accumulation_steps

    # Sampling configuration
    args.temperature = cfg.temperature
    args.top_p = cfg.top_p
    args.top_k = cfg.top_k

    # Length limits
    args.max_prompt_length = cfg.max_prompt_length
    args.max_completion_length = cfg.max_completion_length
    args.max_seq_len = cfg.max_seq_len
    args.mask_truncated_completions = cfg.mask_truncated_completions

    # Optimization settings
    args.learning_rate = cfg.learning_rate
    args.lr_scheduler_type = cfg.lr_scheduler_type
    args.warmup_steps = cfg.warmup_steps
    args.max_steps = cfg.max_steps
    args.num_iterations = cfg.num_iterations

    # Gradient control
    args.max_grad_norm = cfg.max_grad_norm

    # KL regularization
    args.beta = cfg.beta
    args.sync_ref_model = cfg.sync_ref_model
    args.ref_model_sync_steps = cfg.ref_model_sync_steps
    args.ref_model_mixup_alpha = cfg.ref_model_mixup_alpha

    # Loss configuration
    args.loss_type = cfg.loss_type
    args.epsilon = cfg.epsilon
    args.delta = cfg.delta

    # Overlapped training and inference
    args.num_batches_ahead = cfg.num_batches_ahead
    args.async_generation_timeout = cfg.async_generation_timeout
    args.max_concurrent = cfg.max_concurrent

    # Evaluation
    args.eval_strategy = cfg.eval_strategy
    args.eval_steps = cfg.eval_steps
    args.per_device_eval_batch_size = cfg.per_device_eval_batch_size

    return args
