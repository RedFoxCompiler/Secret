"""Trains the compact hybrid brain from RelEx-PT + agentic examples."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from lynx_numpy.brain import LynxBrain
from lynx_numpy.facts import FactGraph

RELEX = "https://huggingface.co/datasets/NLP-CISUC/RelEx-PT/resolve/main/{}"


def download(path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(RELEX.format(split), headers={"User-Agent": "LynxNumPy/0.3"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
        output.write(response.read())


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino do cérebro híbrido, sem backpropagation")
    parser.add_argument("--relex", type=Path, default=Path("data/relex_train.jsonl"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--generations", type=int, default=32)
    parser.add_argument("--limit-mb", type=int, default=500)
    parser.add_argument("--bootstrap-web", action="store_true", help="marca o cérebro para aprender fatos de pesquisas posteriores via CLI")
    parser.add_argument("--output", type=Path, default=Path("models/lynx_brain"))
    args = parser.parse_args()
    if args.download and not args.relex.exists():
        download(args.relex, "train.jsonl")
    if not args.relex.exists():
        raise SystemExit("RelEx-PT ausente; use --download ou --relex ARQUIVO.")
    subprocess.run([sys.executable, "generate_agentic_dataset.py"], check=True)
    subprocess.run([sys.executable, "generate_language_dataset.py"], check=True)
    agentic = lines(Path("data/agentic_like_v2.jsonl"))
    language = lines(Path("data/language_programming_v3.jsonl"))
    relation_rows = lines(args.relex)
    brain = LynxBrain()
    for spec in brain.tools(): brain.core.register_tool(spec)
    # The router needs representative language, not a replay of the factual
    # corpus. Facts are compiled separately below, keeping phone training fast.
    router_rows = relation_rows[:80]
    texts = [r["text"] for r in agentic] + [r["text"] for r in language] + [r["sentence"] for r in router_rows]
    labels = [r["intent"] for r in agentic] + [r["intent"] for r in language] + ["chat.definition"] * len(router_rows)
    metrics = brain.core.fit(texts, labels, generations=args.generations)
    for row in relation_rows:
        brain.facts.add(row["subject"], row["relation"], row["object"], "RelEx-PT")
    mtp_texts = [r["sentence"] for r in relation_rows] + [r["text"] for r in agentic] + [r["text"] for r in language]
    brain.mtp.fit(mtp_texts)
    manifest = brain.save(args.output, args.limit_mb)
    report = {"router": metrics, "facts": len(brain.facts.facts), "agentic_examples": len(agentic), "language_code_examples": len(language), "web_learning_enabled": args.bootstrap_web, "manifest": manifest}
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
