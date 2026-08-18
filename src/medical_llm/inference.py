from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from medical_llm.data import MedicalExample


def load_finetuned_model(config: dict[str, Any], adapter_path: str | Path):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        expected_revision = str(config["model"]["revision"])
        if adapter_config.get("revision") != expected_revision:
            raise RuntimeError(
                "adapter base revision does not match the configured immutable revision"
            )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=int(config["data"]["max_sequence_length"]),
        dtype=None,
        load_in_4bit=bool(config["model"]["load_in_4bit"]),
    )
    tokenizer = get_chat_template(tokenizer, chat_template=config["model"]["chat_template"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _generate_batch(
    model, tokenizer, questions: Sequence[str], config: dict[str, Any]
) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for question in questions
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    generated = model.generate(
        **encoded,
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        do_sample=bool(config["generation"]["do_sample"]),
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_width = encoded["input_ids"].shape[1]
    return tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)


def prompt_hash(tokenizer: Any, question: str) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def generate_base_and_finetuned(
    config: dict[str, Any],
    examples: Sequence[MedicalExample],
    adapter_path: str | Path,
    output_path: str | Path,
) -> int:
    model, tokenizer = load_finetuned_model(config, adapter_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    batch_size = int(config["generation"]["batch_size"])
    count = 0
    generation_config = dict(config["generation"])
    generation_hash = hashlib.sha256(
        json.dumps(generation_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            questions = [item.question for item in batch]
            if not hasattr(model, "disable_adapter"):
                raise RuntimeError(
                    "loaded fine-tuned model cannot disable its adapter for Base inference"
                )
            disable = model.disable_adapter()
            with disable:
                base = _generate_batch(model, tokenizer, questions, config)
            fine_tuned = _generate_batch(model, tokenizer, questions, config)
            for item, base_text, ft_text in zip(batch, base, fine_tuned, strict=True):
                stream.write(
                    json.dumps(
                        {
                            **item.to_dict(),
                            "prompt_hash": prompt_hash(tokenizer, item.question),
                            "generation_hash": generation_hash,
                            "base_prediction": base_text.strip(),
                            "ft_prediction": ft_text.strip(),
                            "generation": generation_config,
                            "model_revision": config["model"]["revision"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                count += 1
            stream.flush()
    return count
