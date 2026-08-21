from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from lynx_numpy.brain import LynxBrain


def calculator(expression: str):
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant)
    node = ast.parse(expression, mode="eval")
    if not all(isinstance(n, allowed) and not (isinstance(n, ast.Constant) and not isinstance(n.value, (int, float))) for n in ast.walk(node)):
        raise ValueError("expressão contém operação não permitida")
    return eval(compile(node, "<calculator>", "eval"), {"__builtins__": {}}, {})


class _SearchParser(HTMLParser):
    def __init__(self): super().__init__(); self.text=[]; self.skip_depth = 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}: self.skip_depth += 1
    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip_depth: self.skip_depth -= 1
    def handle_data(self, data):
        value = data.strip()
        if not self.skip_depth and 35 < len(value) < 600: self.text.append(value)


def web_search(query: str):
    """Google read-only search through curl; does not bypass robot checks."""
    url = "https://www.google.com/search?gbv=1&hl=pt-BR&q=" + urllib.parse.quote(query)
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "20",
               "--user-agent", "Mozilla/5.0 (compatible; LynxNumPy/0.4; educational)", url]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=25)
    if completed.returncode != 0:
        raise RuntimeError("A busca Google recusou ou bloqueou a requisição curl; tente novamente mais tarde ou forneça uma fonte.")
    raw = completed.stdout
    if any(marker in raw.lower() for marker in ("unusual traffic", "captcha", "detected unusual")):
        raise RuntimeError("Google solicitou verificação antirobô; o Lynx não tenta contorná-la.")
    parser = _SearchParser(); parser.feed(raw)
    ignored = {"google", "sign in", "pesquisar", "privacidade", "terms"}
    snippets = [x for x in parser.text if x.lower() not in ignored][:8]
    return {"query": query, "engine": "google", "transport": "curl", "snippets": snippets}


def read_text(path: str):
    root = Path.cwd().resolve(); target = (root / path).resolve()
    if root not in target.parents and target != root: raise PermissionError("arquivo fora do diretório de trabalho")
    return target.read_text(encoding="utf-8")[:12000]


def learn_url(brain: LynxBrain, url: str) -> dict:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("a fonte deve ser uma URL http(s) válida")
    completed = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "30", url],
                               capture_output=True, text=True, timeout=35)
    if completed.returncode != 0:
        raise RuntimeError("não foi possível ler a fonte externa")
    parser = _SearchParser(); parser.feed(completed.stdout)
    sentences = []
    for block in parser.text:
        sentences.extend(re.split(r"(?<=[.!?])\s+", block))
    learned = sum(brain.facts.learn_sentence(sentence, source=url) for sentence in sentences[:250])
    return {"source": url, "readable_segments": len(parser.text), "learned_facts": learned}


def run(brain: LynxBrain, text: str, auto: bool, network: bool, learn_web: bool = False) -> dict:
    started = time.perf_counter(); result = brain.reply(text); plan = result["plan"]
    if plan.get("type") == "function_call":
        allowed = {"calculator": calculator, "clock": lambda: datetime.now().isoformat(timespec="seconds"), "read_text": read_text}
        if network: allowed["web_search"] = web_search
        if auto and plan["name"] in allowed:
            result["tool_result"] = brain.core.execute(plan, allowed)
            if learn_web and plan["name"] == "web_search":
                snippets = result["tool_result"].get("snippets", [])
                learned = sum(brain.facts.learn_sentence(s, source="web:" + plan["arguments"]["query"]) for s in snippets)
                result["learned_web_facts"] = learned
        else:
            result["text"] = "Plano pronto. Use --auto para executar ferramentas locais; --allow-network habilita pesquisa web."
    elapsed = max(time.perf_counter() - started, 1e-6)
    output = str(result.get("text", result.get("tool_result", ""))).split()
    result["metrics"] = {"output_tokens": len(output), "tokens_per_second": round(len(output) / elapsed, 2), "latency_ms": round(elapsed * 1000, 2)}
    return result


def main():
    parser = argparse.ArgumentParser(description="CLI do Lynx Brain")
    parser.add_argument("message", nargs="*"); parser.add_argument("--brain", type=Path, default=Path("models/lynx_brain"))
    parser.add_argument("--auto", action="store_true", help="executa automaticamente apenas funções permitidas")
    parser.add_argument("--allow-network", action="store_true", help="habilita pesquisa Google de leitura via curl")
    parser.add_argument("--learn-web", action="store_true", help="consolida apenas sentenças X é Y de snippets web; exige --auto e --allow-network")
    parser.add_argument("--learn-url", metavar="URL", help="lê uma fonte externa escolhida e consolida relações simples; exige --allow-network")
    args = parser.parse_args(); brain = LynxBrain.load(args.brain)
    if args.learn_web and not (args.auto and args.allow_network): parser.error("--learn-web exige --auto e --allow-network")
    if args.learn_url and not args.allow_network: parser.error("--learn-url exige --allow-network")
    if args.learn_url:
        report = learn_url(brain, args.learn_url); brain.save(args.brain)
        print(json.dumps({"external_learning": report}, ensure_ascii=False, indent=2)); return
    if args.message:
        result = run(brain, " ".join(args.message), args.auto, args.allow_network, args.learn_web)
        if args.learn_web: brain.save(args.brain)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    print("Lynx CLI. :teach sujeito | relação | objeto; :quit para sair.")
    while True:
        try: text = input("você> ").strip()
        except (EOFError, KeyboardInterrupt): break
        if text in {":quit", ":q"}: break
        if text.startswith(":teach "):
            parts = [p.strip() for p in text[7:].split("|")]
            print("aprendido" if len(parts) == 3 and brain.facts.add(*parts, source="user") else "use :teach sujeito | relação | objeto")
            brain.save(args.brain); continue
        result = run(brain, text, args.auto, args.allow_network, args.learn_web)
        if args.learn_web: brain.save(args.brain)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
