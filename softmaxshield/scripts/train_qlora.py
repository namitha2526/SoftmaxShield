"""QLoRA fine-tuning script for training a 1.5B-3B open code model on T4 GPU."""

from __future__ import annotations

import argparse
import os


def train_qlora(
    model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    dataset_path: str = "data/finetuning_dataset.jsonl",
    output_dir: str = "./softmaxshield_qlora_adapter",
    max_steps: int = 100,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
) -> None:
    """Execute 4-bit QLoRA instruction tuning on free-tier GPU (T4 15GB VRAM friendly)."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ImportError as e:
        raise ImportError(
            "QLoRA fine-tuning requires 'torch', 'transformers', 'peft', 'trl', 'bitsandbytes', and 'datasets'. "
            "Install them via: pip install torch transformers peft trl bitsandbytes datasets"
        ) from e

    print(f"=== Starting SoftmaxShield QLoRA Fine-Tuning ===")
    print(f"Base Model: {model_id}")
    print(f"Dataset   : {dataset_path}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # 1. 4-bit Quantization Configuration (NormalFloat4 + Double Quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16,
    )

    # 2. Tokenizer Loading
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Quantized Model Loading
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map=device_map,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA Adapter Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Load and Split Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]

    # 6. Formatting Template Function
    def format_prompt(sample: dict) -> str:
        return (
            f"<|im_start|>system\n"
            f"You are a verified Python code generator running under SoftmaxShield. "
            f"Use only verified functions and refuse hallucinated APIs.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{sample['instruction']}\n{sample['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{sample['output']}<|im_end|>"
        )

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_steps=max_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none",
    )

    # 8. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        formatting_func=format_prompt,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("\n--- Commencing QLoRA Training ---")
    trainer.train()

    print(f"\n--- Saving Fine-Tuned LoRA Adapter to {output_dir} ---")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoftmaxShield QLoRA Fine-Tuning")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--dataset", default="data/finetuning_dataset.jsonl")
    parser.add_argument("--output-dir", default="./softmaxshield_qlora_adapter")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    train_qlora(
        model_id=args.model_id,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
