from __future__ import annotations

import argparse
import json

from lynx_numpy import LynxCore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+", help="frase a interpretar")
    parser.add_argument("--model", default="models/lynx_seed.npz")
    args = parser.parse_args()
    core = LynxCore.load(args.model)
    text = " ".join(args.text)
    print(json.dumps({"prediction": core.predict(text), "plan": core.plan(text)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
