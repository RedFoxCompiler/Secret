from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from lynx_numpy.language import SparseLanguageCore, load_alpaca_conversations


DATASET_URL = "https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-portuguese/resolve/main/alpaca-gpt4-portuguese.json"


def download_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "LynxNumPy/0.2"})
    with urllib.request.urlopen(request, timeout=180) as response, path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino linguístico local/evolutivo sem backpropagation")
    parser.add_argument("--dataset", type=Path, default=Path("data/alpaca-gpt4-portuguese.json"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--examples", type=int, default=1600)
    parser.add_argument("--validation", type=int, default=120)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--generations", type=int, default=35)
    parser.add_argument("--output", type=Path, default=Path("models/lynx_language_pt.npz"))
    args = parser.parse_args()
    if args.download and not args.dataset.exists():
        download_dataset(args.dataset)
    if not args.dataset.exists():
        raise SystemExit("Dataset ausente. Use --download ou informe --dataset CAMINHO.")
    pairs = load_alpaca_conversations(args.dataset, limit=args.examples + args.validation)
    if len(pairs) <= args.validation:
        raise SystemExit("Poucos exemplos válidos para a divisão solicitada.")
    train, validation = pairs[:-args.validation], pairs[-args.validation:]
    model = SparseLanguageCore(hidden=args.hidden, max_vocab=args.vocab, seed=23)
    summary = model.fit(train, validation, epochs=2, evolution_generations=args.generations)
    summary["heldout_candidate_nll"] = model.heldout_nll(validation)
    prompts = [
        "Explique de forma simples o que é energia solar.",
        "Crie três ideias para estudar melhor.",
    ]
    summary["raw_samples"] = [
        {"prompt": prompt, "response": model.generate(prompt, seed=index + 2)}
        for index, prompt in enumerate(prompts)
    ]
    summary["controlled_samples"] = [
        {"prompt": prompt, **model.respond(prompt)} for prompt in prompts
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    args.output.with_suffix(".metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
