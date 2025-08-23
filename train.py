import tyro
import verifiers as vf

from ragent.utils import TrainConfig, prepare_args

"""
# Quick eval
vf-eval bm25s (-m model_name in endpoints.py)

Inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B \
    --enforce-eager --disable-log-requests

Training (override defaults with CLI flags via tyro):
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml ragent/train.py \
    --hf-dataset nampdn-ai/devdocs.io \
    --model-name Qwen/Qwen2.5-0.5B \
    --per-device-train-batch-size 4 \
    --num-generations 8 \
    --max-steps 100
"""


def main(cfg: TrainConfig) -> None:
    vf_env = vf.load_environment(
        env_id=cfg.env_id,
        hf_dataset=cfg.hf_dataset,
        llm_name_judge=cfg.llm_name_judge,
    )
    model, tokenizer = vf.get_model_and_tokenizer(cfg.model_name)

    args = prepare_args(cfg)

    trainer = vf.GRPOTrainer(
        env=vf_env,
        model=model,
        processing_class=tokenizer,
        args=args,
    )

    trainer.train()


if __name__ == "__main__":
    config = tyro.cli(TrainConfig)
    main(config)