from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_llm.config import load_config
from medical_llm.external_eval import prepare_pubmedqa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/qlora_llama3_8b.yaml"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    _, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        revision=config["model"]["revision"],
        max_seq_length=config["data"]["max_sequence_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template=config["model"]["chat_template"])
    print(json.dumps(prepare_pubmedqa(config, tokenizer, args.output), indent=2))


if __name__ == "__main__":
    main()
