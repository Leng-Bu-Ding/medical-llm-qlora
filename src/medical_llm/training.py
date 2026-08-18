from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medical_llm.config import config_sha256
from medical_llm.data import MedicalExample, read_jsonl
from medical_llm.provenance import environment_summary, write_json


def prompt_completion_record(example: MedicalExample) -> dict[str, Any]:
    return {
        "prompt": [{"role": "user", "content": example.question}],
        "completion": [{"role": "assistant", "content": example.reference}],
        "sample_id": example.sample_id,
        "qtype": example.qtype,
    }


def validated_eos_token(tokenizer: Any) -> str:
    """Return a real tokenizer EOS token instead of a trainer placeholder."""
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise RuntimeError("the chat tokenizer must define a non-empty eos_token")
    token_id = tokenizer.convert_tokens_to_ids(eos_token)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or (unknown_id is not None and token_id == unknown_id):
        raise RuntimeError(f"tokenizer eos_token {eos_token!r} is not in the vocabulary")
    return eos_token


def _plot_losses(log_history: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    train_points = [(row["step"], row["loss"]) for row in log_history if "loss" in row]
    eval_points = [(row["step"], row["eval_loss"]) for row in log_history if "eval_loss" in row]
    if not train_points and not eval_points:
        return
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if train_points:
        axis.plot(*zip(*train_points, strict=True), label="train loss")
    if eval_points:
        axis.plot(*zip(*eval_points, strict=True), marker="o", label="validation loss")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.set_title("QLoRA training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def audit_completion_loss_mask(batch: dict[str, Any]) -> dict[str, int]:
    """Fail fast unless a real batch contains masked and supervised token positions."""
    labels = batch.get("labels")
    if labels is None:
        raise RuntimeError("training batch has no labels for completion-only loss audit")
    flattened = labels.reshape(-1)
    masked_tokens = int((flattened == -100).sum().item())
    supervised_tokens = int((flattened != -100).sum().item())
    if masked_tokens == 0 or supervised_tokens == 0:
        raise RuntimeError(
            "completion-only loss audit expected masked prompt and supervised answer tokens"
        )
    return {
        "masked_prompt_or_padding_tokens": masked_tokens,
        "supervised_answer_tokens": supervised_tokens,
    }


def pinned_adapter_config(
    adapter_config: dict[str, Any], *, base_model: str, revision: str
) -> dict[str, Any]:
    adapter_config = dict(adapter_config)
    adapter_config["base_model_name_or_path"] = base_model
    adapter_config["revision"] = revision
    return adapter_config


def pin_adapter_base_revision(
    adapter_dir: str | Path, *, base_model: str, revision: str
) -> dict[str, Any]:
    """Persist the immutable base revision used to train a PEFT adapter."""
    config_path = Path(adapter_dir) / "adapter_config.json"
    adapter_config = pinned_adapter_config(
        json.loads(config_path.read_text(encoding="utf-8")),
        base_model=base_model,
        revision=revision,
    )
    config_path.write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return adapter_config


def train(
    config: dict[str, Any],
    train_path: str | Path,
    validation_path: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    max_steps: int | None = None,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    lora_rank: int | None = None,
) -> dict[str, Any]:
    # Unsloth must patch Transformers/TRL before either package is imported.
    # Importing TRL first can replace the real EOS with the invalid <EOS_TOKEN>
    # placeholder (unslothai/unsloth#2797).
    from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: I001
    from unsloth.chat_templates import get_chat_template
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = config["lora"]
    selected_rank = int(lora_rank or lora["rank"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=selected_rank,
        target_modules=list(lora["target_modules"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        use_gradient_checkpointing=str(lora["gradient_checkpointing"]),
        random_state=int(config["project"]["seed"]),
        use_rslora=bool(lora["use_rslora"]),
        loftq_config=None,
    )

    train_examples = read_jsonl(train_path)
    if train_limit is not None:
        train_examples = train_examples[:train_limit]
    validation_examples = read_jsonl(validation_path) if validation_path else []
    if validation_limit is not None:
        validation_examples = validation_examples[:validation_limit]
    train_dataset = Dataset.from_list([prompt_completion_record(item) for item in train_examples])
    eval_dataset = (
        Dataset.from_list([prompt_completion_record(item) for item in validation_examples])
        if validation_examples
        else None
    )

    run_dir = Path(output_dir or config["project"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = config["training"]
    actual_max_steps = int(max_steps) if max_steps is not None else -1
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=str(run_dir / "checkpoints"),
            per_device_train_batch_size=int(settings["per_device_train_batch_size"]),
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
            warmup_steps=int(settings["warmup_steps"]),
            num_train_epochs=float(settings["epochs"]),
            max_steps=actual_max_steps,
            learning_rate=float(settings["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=int(settings["logging_steps"]),
            optim=str(settings["optimizer"]),
            weight_decay=float(settings["weight_decay"]),
            lr_scheduler_type=str(settings["scheduler"]),
            seed=int(config["project"]["seed"]),
            data_seed=int(config["project"]["seed"]),
            save_strategy="steps",
            save_steps=int(settings["save_steps"]),
            save_total_limit=int(settings["save_total_limit"]),
            eval_strategy=str(settings["eval_strategy"]) if eval_dataset is not None else "no",
            report_to="none",
            max_length=int(data_config["max_sequence_length"]),
            dataset_num_proc=2,
            packing=bool(settings["packing"]),
            completion_only_loss=bool(settings["completion_only_loss"]),
            eos_token=validated_eos_token(tokenizer),
            include_num_input_tokens_seen="all",
        ),
    )
    loss_mask_audit = audit_completion_loss_mask(next(iter(trainer.get_train_dataloader())))
    stats = trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
    )
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_config = pin_adapter_base_revision(
        adapter_dir,
        base_model=str(model_config["name"]),
        revision=str(model_config["revision"]),
    )
    log_history = list(trainer.state.log_history)
    write_json(log_history, run_dir / "trainer_log.json")
    _plot_losses(log_history, run_dir / "loss_curve.png")

    peak_memory_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    metadata = {
        "status": "smoke_tested" if actual_max_steps > 0 else "measured",
        "train_samples": len(train_examples),
        "validation_samples": len(validation_examples),
        "completion_only_loss": bool(settings["completion_only_loss"]),
        "loss_mask_audit": loss_mask_audit,
        "lora_rank": selected_rank,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "runtime_seconds": stats.metrics.get("train_runtime"),
        "samples_per_second": stats.metrics.get("train_samples_per_second"),
        "steps_per_second": stats.metrics.get("train_steps_per_second"),
        "input_tokens_seen": getattr(trainer.state, "num_input_tokens_seen", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gib": round(peak_memory_bytes / 1024**3, 4) if peak_memory_bytes else None,
        "adapter_dir": str(adapter_dir),
        "adapter_base_model": adapter_config["base_model_name_or_path"],
        "adapter_base_revision": adapter_config["revision"],
        "config_sha256": config_sha256(config),
        "environment": environment_summary(Path(__file__).resolve().parents[2]),
        "trainer_metrics": stats.metrics,
    }
    write_json(config, run_dir / "resolved_config.json")
    write_json(metadata, run_dir / "training_summary.json")
    return metadata
