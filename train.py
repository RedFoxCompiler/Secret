from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lynx_numpy import LynxCore, ToolSpec


def load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r["text"] for r in rows], [r["label"] for r in rows]


def stratified_split(texts: list[str], labels: list[str], seed: int = 9):
    rng = np.random.default_rng(seed)
    train_ids, test_ids = [], []
    for label in sorted(set(labels)):
        ids = np.asarray([i for i, value in enumerate(labels) if value == label])
        rng.shuffle(ids)
        test_ids.extend(ids[:2].tolist())
        train_ids.extend(ids[2:].tolist())
    return ([texts[i] for i in train_ids], [labels[i] for i in train_ids],
            [texts[i] for i in test_ids], [labels[i] for i in test_ids])


def configured_core() -> LynxCore:
    core = LynxCore(dim=128, seed=13)
    core.register_tool(ToolSpec("weather", "Consulta meteorologia", {"location": {"type": "string"}}, ["location"]))
    core.register_tool(ToolSpec("clock", "Consulta hora local"))
    core.register_tool(ToolSpec("calculator", "Calcula expressão", {"expression": {"type": "string"}}, ["expression"]))
    core.register_tool(ToolSpec("reminder", "Cria lembrete", {"message": {"type": "string"}}, ["message"]))
    core.register_tool(ToolSpec("web_search", "Pesquisa pública", {"query": {"type": "string"}}, ["query"]))
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o núcleo Lynx sem backpropagation")
    parser.add_argument("--data", type=Path, default=Path("data/seed_intents.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/lynx_seed.npz"))
    parser.add_argument("--generations", type=int, default=70)
    args = parser.parse_args()
    texts, labels = load_jsonl(args.data)
    train_x, train_y, test_x, test_y = stratified_split(texts, labels)
    core = configured_core()
    evolution = core.fit(train_x, train_y, generations=args.generations)
    train_acc = np.mean([core.predict(x)["intent"] == y for x, y in zip(train_x, train_y)])
    test_predictions = [core.predict(x)["intent"] for x in test_x]
    test_acc = np.mean([prediction == target for prediction, target in zip(test_predictions, test_y)])
    per_class = {}
    for label in sorted(set(test_y)):
        ids = [i for i, target in enumerate(test_y) if target == label]
        per_class[label] = float(np.mean([test_predictions[i] == test_y[i] for i in ids]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    core.save(args.output)
    report = {**evolution, "train_accuracy": float(train_acc), "heldout_accuracy": float(test_acc),
              "per_class_heldout_accuracy": per_class,
              "train_examples": len(train_x), "heldout_examples": len(test_x), "labels": core.labels,
              "notes": "Few-shot seed benchmark; not a production quality claim."}
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
