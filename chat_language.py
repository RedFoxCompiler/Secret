from __future__ import annotations

import argparse

from lynx_numpy.language import SparseLanguageCore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="+")
    parser.add_argument("--model", default="models/lynx_language_pt.npz")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--raw", action="store_true", help="usa somente o gerador recorrente experimental")
    args = parser.parse_args()
    model = SparseLanguageCore.load(args.model)
    prompt = " ".join(args.prompt)
    if args.raw:
        print(model.generate(prompt, args.max_tokens, args.temperature))
    else:
        print(model.respond(prompt, args.max_tokens)["text"])


if __name__ == "__main__":
    main()
