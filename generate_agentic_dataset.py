"""Creates synthetic, reviewable routing examples; no hidden reasoning traces."""
from __future__ import annotations

import json
from pathlib import Path


TOOLS = {
    "tool.calculator": ["calcule {x}", "quanto é {x}", "faça a conta {x}", "resultado de {x}"],
    "tool.clock": ["que horas são", "mostre a hora agora", "qual é a data de hoje", "veja o relógio"],
    "tool.web_search": ["pesquise sobre {topic}", "busque {topic} na web", "procure informações sobre {topic}", "encontre {topic}"],
    "tool.read_text": ["leia o arquivo {file}", "abra o arquivo {file}", "mostre o conteúdo de {file}"],
}


def main() -> None:
    rows = []
    for expr in ("2 + 2", "13 * 7", "(12 + 8) / 4", "100 - 37", "3 ** 5", "18 / 3", "9 * 9", "40 + 12", "81 - 9", "5 % 2"):
        for template in TOOLS["tool.calculator"]:
            rows.append({"text": template.format(x=expr), "intent": "tool.calculator", "available_functions": ["calculator", "clock", "web_search", "read_text"], "reasoning_summary": "a solicitação exige cálculo local", "function_call": {"name": "calculator", "arguments": {"expression": expr}}})
    for template in TOOLS["tool.clock"]:
        for _ in range(10):
            rows.append({"text": template, "intent": "tool.clock", "available_functions": ["calculator", "clock", "web_search", "read_text"], "reasoning_summary": "a solicitação exige o relógio local", "function_call": {"name": "clock", "arguments": {}}})
    for topic in ("energia solar", "Python", "notícias de ciência", "redes neurais evolutivas", "clima em Palmas", "aprendizado de máquina", "astronomia", "história do Brasil", "saúde pública", "programação NumPy"):
        for template in TOOLS["tool.web_search"]:
            rows.append({"text": template.format(topic=topic), "intent": "tool.web_search", "available_functions": ["calculator", "clock", "web_search", "read_text"], "reasoning_summary": "a solicitação exige pesquisa pública", "function_call": {"name": "web_search", "arguments": {"query": topic}}})
    for file in ("notas.txt", "dados.txt", "README.md", "agenda.txt", "tarefas.txt", "memoria.txt", "texto.md", "config.json"):
        for template in TOOLS["tool.read_text"]:
            rows.append({"text": template.format(file=file), "intent": "tool.read_text", "available_functions": ["calculator", "clock", "web_search", "read_text"], "reasoning_summary": "a solicitação exige leitura local", "function_call": {"name": "read_text", "arguments": {"path": file}}})
    for text in ("o que é energia solar", "explique aprendizado de máquina", "defina uma rede neural", "o que significa fotossíntese", "me ajude a entender um conceito"):
        rows.extend({"text": text, "intent": "chat.definition", "available_functions": ["calculator", "clock", "web_search", "read_text"], "reasoning_summary": "a solicitação é conversacional", "function_call": None} for _ in range(12))
    target = Path("data/agentic_like_v2.jsonl"); target.parent.mkdir(exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"examples": len(rows), "path": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
