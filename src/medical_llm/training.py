from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_llm.data import read_jsonl


def train(config: dict[str, Any], train_path: str | Path) -> dict[str, Any]:
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template

    model_config = config["model"]
    data_config = config["data"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["name"],
        revision=model_config["revision"],
        max_seq_length=int(data_config["max_sequence_length"]),
        dtype=None,
        load_in_4bit=bool(model_config["load_in_4bit"]),
    )
    tokenizer = get_chat_template(tokenizer, chat_template=model_config["chat_template"])
    lora = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora["rank"]),
        target_modules=list(lora["target_modules"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        use_gradient_checkpointing=str(lora["gradient_checkpointing"]),
        random_state=int(config["project"]["seed"]),
        use_rslora=bool(lora["use_rslora"]),
        loftq_config=None,
    )
    examples = read_jsonl(train_path)
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": item.question},
                {"role": "assistant", "content": item.reference},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        for item in examples
    ]
    dataset = Dataset.from_dict({"text": texts})
    run_dir = Path(config["project"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = config["training"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=int(data_config["max_sequence_length"]),
        dataset_num_proc=2,
        packing=bool(settings["packing"]),
        args=TrainingArguments(
            per_device_train_batch_size=int(settings["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
            warmup_steps=int(settings["warmup_steps"]),
            num_train_epochs=float(settings["epochs"]),
            learning_rate=float(settings["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=int(settings["logging_steps"]),
            optim=str(settings["optimizer"]),
            weight_decay=float(settings["weight_decay"]),
            lr_scheduler_type=str(settings["scheduler"]),
            seed=int(config["project"]["seed"]),
            output_dir=str(run_dir / "checkpoints"),
            save_strategy="steps",
            save_steps=int(settings["save_steps"]),
            report_to="none",
        ),
    )
    stats = trainer.train()
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    pd.DataFrame(trainer.state.log_history).to_json(
        run_dir / "trainer_log.json", orient="records", indent=2
    )
    metadata = {
        "train_samples": len(examples),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "runtime_seconds": stats.metrics.get("train_runtime"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "adapter_dir": str(adapter_dir),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
