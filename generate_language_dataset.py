"""Generate small bilingual supervised records for code and long-form control."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    code_prompts = [
        "Faça um script em Python que imprima Hello World.", "Escreva print hello world em Python.",
        "Write Python code to print hello world.", "Create a Python hello world program.",
        "Faça uma função de Fibonacci em Python.", "Write a Fibonacci function in Python.",
        "Crie um script Python para ler um arquivo texto.", "Write Python code to read a UTF-8 text file.",
    ]
    long_prompts = [
        "Explique energia solar de forma simples.", "Defina aprendizado de máquina com detalhes.",
        "Explain solar energy in simple terms.", "Define machine learning with context.",
        "Explique como uma rede neural aprende.", "Explain how a neural network learns.",
    ]
    rows = []
    for prompt in code_prompts:
        rows.extend({"text": prompt, "intent": "chat.code", "kind": "code"} for _ in range(15))
    for prompt in long_prompts:
        rows.extend({"text": prompt, "intent": "chat.longform", "kind": "longform"} for _ in range(15))
    output = Path("data/language_programming_v3.jsonl")
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"examples": len(rows), "path": str(output)}, ensure_ascii=False))


if __name__ == "__main__": main()
