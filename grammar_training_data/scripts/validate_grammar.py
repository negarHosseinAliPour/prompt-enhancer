"""Validate a single, already-discovered VSL grammar against a full pool of
raw (ungraded) {task_id, description, module_interface} records -- the same
structural-validity check discover_grammar.py's test_grammar()/_try_one()
do internally, but standalone so a winning grammar can be checked against
the FULL pool (e.g. all 2,649 RTL-Coder_small tasks) rather than just the
150-task sample used during the discovery rounds themselves.

Does NOT run iverilog / simulation -- like the discovery loop's own test
step, this only checks that the candidate agent's VSL parses, validates,
and renders successfully. This dataset has no testbenches, so a real
execution score is not possible here regardless.

Usage:
    python3 validate_grammar.py \
        --grammar ../discovered_grammar_multidataset.txt \
        --problems ../converted/rtlcoder_small_problems.jsonl \
        --out ../history/rtlcoder_small_validation.jsonl \
        --concurrency 8
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Reuse vsl_core's own MODEL, which already resolves VSL_MODEL=gpt-oss (and
# any other supported value) to the correct pydantic_ai model object --
# duplicating that resolution here (e.g. passing the literal string
# "gpt-oss" to Agent()) fails, since "gpt-oss" is a VSL_MODEL env var value,
# not a real pydantic_ai model identifier.
from vsl_core import parse_vsl, validate_circuit, render_verilog, VSLParseError, ValidationError, MODEL

from pydantic import BaseModel, Field
from pydantic_ai import Agent


class DiscoveryVSLOutput(BaseModel):
    vsl_code: str = Field(..., description="Circuit logic written in VSL, following the grammar in the system prompt.")


if os.environ.get("VSL_MODEL") == "gpt-oss":
    from pydantic_ai import NativeOutput as _NativeOutput
    _candidate_output_type = _NativeOutput(DiscoveryVSLOutput)
else:
    _candidate_output_type = DiscoveryVSLOutput


async def _try_one(agent: Agent, task_id: str, description: str, module_interface: str) -> dict:
    try:
        result = await agent.run(description)
        vsl_text = result.output.vsl_code
    except Exception as e:
        return {"task_id": task_id, "ok": False, "stage": "agent_call", "error": f"{type(e).__name__}: {e}", "vsl_text": None}

    try:
        ir = parse_vsl(vsl_text, module_interface=module_interface)
    except Exception as e:
        return {"task_id": task_id, "ok": False, "stage": "parse", "error": f"{type(e).__name__}: {e}", "vsl_text": vsl_text}

    try:
        problems = validate_circuit(ir)
    except Exception as e:
        return {"task_id": task_id, "ok": False, "stage": "validate", "error": f"{type(e).__name__}: {e}", "vsl_text": vsl_text}
    if problems:
        return {"task_id": task_id, "ok": False, "stage": "validate", "error": "; ".join(problems), "vsl_text": vsl_text}

    try:
        render_verilog(ir)
    except Exception as e:
        return {"task_id": task_id, "ok": False, "stage": "render", "error": f"{type(e).__name__}: {e}", "vsl_text": vsl_text}

    return {"task_id": task_id, "ok": True, "stage": "done", "error": None, "vsl_text": vsl_text}


async def main(grammar_path: Path, problems_path: Path, out_path: Path, concurrency: int, limit: int | None):
    grammar_text = grammar_path.read_text(encoding="utf-8")
    with open(problems_path, encoding="utf-8") as f:
        problems = [json.loads(l) for l in f if l.strip()]
    if limit:
        problems = problems[:limit]

    agent = Agent(
        MODEL,
        name="Validation Candidate Agent",
        output_type=_candidate_output_type,
        model_settings={"temperature": 0},
        system_prompt=grammar_text,
    )

    sem = asyncio.Semaphore(concurrency)

    async def bound(p):
        async with sem:
            return await _try_one(agent, p["task_id"], p["description"], p.get("module_interface", ""))

    passed = 0
    total = 0
    by_stage = {}
    with open(out_path, "w", encoding="utf-8") as out_f:
        for coro in asyncio.as_completed([bound(p) for p in problems]):
            rec = await coro
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            total += 1
            if rec["ok"]:
                passed += 1
            else:
                by_stage[rec["stage"]] = by_stage.get(rec["stage"], 0) + 1
            if total % 100 == 0 or total == len(problems):
                print(f"[{total}/{len(problems)}] pass rate so far: {passed}/{total} ({passed/total:.1%})")

    print(f"\nFinal: {passed}/{total} passed ({passed/total:.1%})")
    if by_stage:
        print("Failure breakdown by stage:")
        for stage, count in sorted(by_stage.items(), key=lambda kv: -kv[1]):
            print(f"  {stage}: {count}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grammar", required=True, help="Path to the discovered grammar .txt file to validate.")
    ap.add_argument("--problems", required=True, help="Path to the raw {task_id, description, module_interface} JSONL pool.")
    ap.add_argument("--out", required=True, help="Where to write per-task results (JSONL).")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="Optional cap for a quick smoke test before the full run.")
    args = ap.parse_args()
    asyncio.run(main(Path(args.grammar), Path(args.problems), Path(args.out), args.concurrency, args.limit))
