import verifiers as vf

"""
# quick eval
vf-eval bm25s (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B \
    --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml ragent/train.py
"""

vf_env = vf.load_environment(env_id="ragent.environments.bm25s", hf_dataset="nampdn-ai/devdocs.io", llm_name_judge="google/gemini-2.5-flash")
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-0.5B")

args = vf.grpo_defaults(run_name="bm25s")

# Core batch settings
args.per_device_train_batch_size = 8    # Prompts per GPU per step
args.num_generations = 16               # Completions per prompt (group size)
args.gradient_accumulation_steps = 4    # Steps before optimizer update

# Sampling configuration
args.temperature = 1.0          # Higher = more diverse completions
args.top_p = 1.0               # Nucleus sampling threshold
args.top_k = None              # Top-k filtering (None = disabled)

# Length limits
args.max_prompt_length = 1024     # Truncate prompts (left-truncated)
args.max_completion_length = 2048  # Truncate completions
args.max_seq_len = 131072         # Model's context window
args.mask_truncated_completions = True

# Optimization settings
args.learning_rate = 1e-6              # Conservative default
args.lr_scheduler_type = "constant_with_warmup"
args.warmup_steps = 10                 # Gradual warmup
args.max_steps = 500                   # Total training steps
args.num_iterations = 1                # PPO-style updates per batch

# Gradient control
args.max_grad_norm = 0.01              # Aggressive clipping for stability

# KL regularization
args.beta = 0.001                      # KL penalty coefficient
args.sync_ref_model = True             # Update reference model
args.ref_model_sync_steps = 100        # How often to sync
args.ref_model_mixup_alpha = 0.5       # Mix ratio for updates

# Loss configuration
args.loss_type = "dr_grpo"             # Recommended: no length bias
args.epsilon = 0.2                     # Clipping bound (lower)
args.delta = None                      # Optional upper clipping bound

# Overlapped training and inference
args.num_batches_ahead = 1      # Batches to generate ahead
args.async_generation_timeout = 300.0  # Timeout in seconds
args.max_concurrent = 1024      # Max concurrent env requests

# Add evaluation dataset
args.eval_strategy = "steps"
args.eval_steps = 100
args.per_device_eval_batch_size = 16

trainer = vf.GRPOTrainer(
    env=vf_env,
    model=model,
    processing_class=tokenizer,
    args=args,
)
# trainer.train()