"""Run an eval suite and store its result, or check the stored results.

    python run.py redteam       [--guardrails-url URL]
    python run.py faithfulness  [--n 12] [--seed 7] [--no-calibrate]
    python run.py detection     --data DIR --model PATH [--split test] [--limit N]
    python run.py gate          [--require redteam,faithfulness]

Each suite writes results/<suite>.json: the result plus when and on which
commit it ran. Those files are committed, so a pull request shows the change
in numbers next to the change in code, and gate.py checks them in CI.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import paths


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=paths.ROOT, check=True
    ).stdout.strip()


def commit():
    """The commit the result was measured on; "+dirty" when the working tree
    had uncommitted changes, so the result may not match that commit's code.
    The container has no git, so compose passes EVAL_COMMIT from the host:
        EVAL_COMMIT=$(git describe --always --dirty) docker compose run --rm eval ..."""
    for var in ("EVAL_COMMIT", "GITHUB_SHA"):
        if os.getenv(var):
            return os.environ[var]
    try:
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}+dirty" if dirty else head


def save(suite, result, out=None):
    out = out or paths.RESULTS / f"{suite}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "suite": suite,
        "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit(),
        "result": result,
    }
    out.write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def faithfulness(args):
    import faithfulness as fa
    from llm import OllamaLLM

    writer = OllamaLLM()  # production's report model (OLLAMA_MODEL)
    judge = OllamaLLM(model=fa.JUDGE_MODEL, think=fa.JUDGE_THINK)
    return asyncio.run(
        fa.run(writer, judge, n=args.n, seed=args.seed, calibrate=not args.no_calibrate)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="suite", required=True)
    p = sub.add_parser("redteam")
    p.add_argument("--guardrails-url", default=os.getenv("GUARDRAILS_URL"))
    p = sub.add_parser("faithfulness")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-calibrate", action="store_true")
    p = sub.add_parser("detection")
    p.add_argument(
        "--data", required=True, help="YOLO-seg folder (convert_cardd.py output)"
    )
    p.add_argument("--model", default=str(paths.DATA / "models" / "best.onnx"))
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("gate")
    p.add_argument(
        "--require", default="", help="comma-separated suites that must have results"
    )
    for name, sp in sub.choices.items():
        if name != "gate":
            sp.add_argument("--out", help="default: results/<suite>.json")
    args = parser.parse_args(argv)

    if args.suite == "gate":
        import gate

        return gate.main([s for s in args.require.split(",") if s])
    if args.suite == "redteam":
        import redteam

        result = redteam.run(args.guardrails_url)
    elif args.suite == "faithfulness":
        result = faithfulness(args)
    else:
        import detection

        result = detection.run(args.data, args.model, args.split, args.limit)
    path = save(args.suite, result, Path(args.out) if args.out else None)
    print(json.dumps(result, indent=1, ensure_ascii=False)[:4000])
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
