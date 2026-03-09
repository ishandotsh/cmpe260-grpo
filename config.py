from trl import GRPOConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_DATASET = "qwedsacf/competition_math"
EVAL_DATASET = "HuggingFaceH4/MATH-500"

SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the following problem step by step. "
    "Put your final answer within \\boxed{}."
)

DIFFICULTY_LEVELS = [1, 2, 3, 4, 5]

WANDB_PROJECT = "grpo-difficulty-variance"


def get_grpo_config(
    sampling: str,
    output_dir: str | None = None,
    wandb_project: str | None = None,
    max_steps: int = -1,
    run_name: str | None = None,
) -> GRPOConfig:
    if run_name is None:
        run_name = f"grpo-{sampling}"
    if output_dir is None:
        output_dir = f"./output-{sampling}"
    if wandb_project is None:
        wandb_project = WANDB_PROJECT

    return GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_completion_length=1024,
        learning_rate=5e-6,
        beta=0.04,
        max_grad_norm=0.1,
        num_train_epochs=3,
        max_steps=max_steps,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        bf16=True,
        seed=1337,
    )
