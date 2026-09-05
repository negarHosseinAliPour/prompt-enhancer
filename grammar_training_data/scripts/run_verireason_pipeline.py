import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vsl_core import gir_agent, parse_vsl, validate_circuit, render_verilog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_verireason import check_correctness_verireason


def _fail(task_id, prompt, stage, e):
    return {"task_id": task_id, "rounds": [{"round": -1, "prompt": prompt, "execution_score": 0.0, "vsl_text": None, "error": f"{stage}: {type(e).__name__}: {e}"}]}


async def run_one(problem: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        task_id = problem["task_id"]
        prompt = problem["description"]
        module_interface = problem["module_interface"]
        # the description alone doesn't always name the real ports (unlike
        # VerilogEval's descriptions, which do) -- give the model the real
        # interface too so it uses matching signal names. "prompt" in the
        # saved history stays the plain description, so discovery data
        # (and later grammar generalization) isn't tied to this detail.
        agent_input = (
            prompt
            + "\n\nThe module's exact interface (use these exact signal names):\n"
            + module_interface
        )

        try:
            result = await gir_agent.run(agent_input)
            vsl_text = result.output.vsl_code
        except Exception as e:
            return _fail(task_id, prompt, "agent_call", e)

        try:
            ir = parse_vsl(vsl_text, module_interface=module_interface)
        except Exception as e:
            rec = _fail(task_id, prompt, "parse", e)
            rec["rounds"][0]["vsl_text"] = vsl_text
            return rec

        try:
            problems = validate_circuit(ir)
            if problems:
                rec = _fail(task_id, prompt, "validate", "; ".join(problems))
                rec["rounds"][0]["vsl_text"] = vsl_text
                return rec
        except Exception as e:
            rec = _fail(task_id, prompt, "validate", e)
            rec["rounds"][0]["vsl_text"] = vsl_text
            return rec

        try:
            body = render_verilog(ir)
            # module_interface is the bare header ("module name(...);"), no
            # body and no endmodule -- same convention main_vsl.py uses for
            # VerilogEval's "prompt" field.
            candidate = module_interface + "\n" + body + "\n\nendmodule\n"
        except Exception as e:
            rec = _fail(task_id, prompt, "render", e)
            rec["rounds"][0]["vsl_text"] = vsl_text
            return rec

        try:
            grade = check_correctness_verireason(problem, candidate)
        except Exception as e:
            rec = _fail(task_id, prompt, "grade", e)
            rec["rounds"][0]["vsl_text"] = vsl_text
            return rec

        return {
            "task_id": task_id,
            "rounds": [{
                "round": -1,
                "prompt": prompt,
                "execution_score": grade["score"],
                "vsl_text": vsl_text,
                "gradeable": grade["gradeable"],
                "detail": grade["detail"],
            }],
        }


async def main(problems_path: Path, out_path: Path, concurrency: int, limit: int | None):
    with open(problems_path, encoding="utf-8") as f:
        problems = [json.loads(l) for l in f if l.strip()]
    if limit:
        problems = problems[:limit]

    sem = asyncio.Semaphore(concurrency)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for coro in asyncio.as_completed([run_one(p, sem) for p in problems]):
            rec = await coro
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            r0 = rec["rounds"][0]
            note = r0.get("error", "")
            print(f"{rec['task_id']}: {r0['execution_score']} {note}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="../converted/verireason_problems.jsonl")
    ap.add_argument("--out", default="../history/verireason_gptoss_history.jsonl")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main(Path(args.problems), Path(args.out), args.concurrency, args.limit))
