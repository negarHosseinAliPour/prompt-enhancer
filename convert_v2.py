"""
Converts VerilogEval v2 (spec-to-RTL) into the record format main_vsl_3.py
already reads, so the existing pipeline runs on v2 with no code changes.

v2 ships one problem as three files -- Prob###_name_prompt.txt (the spec),
_ref.sv (the reference module) and _test.sv (the testbench, which
instantiates both RefModule and TopModule). The pipeline instead expects
two JSONL files shaped like v1:

  eval  : {task_id, prompt, canonical_solution, test}
  desc  : {task_id, simple_description, detail_description}

The mapping:

  prompt             the TopModule header extracted from _ref.sv, ending at
                     the closing ");" -- this is the fixed interface the
                     pipeline appends the generated body to
  canonical_solution the reference body + endmodule
  test               _test.sv followed by _ref.sv, since the testbench needs
                     RefModule present to compare against
  descriptions       the spec text from _prompt.txt

ONE DEVIATION WORTH KNOWING. Native v2 spec-to-RTL asks the model to invent
the interface itself; this conversion hands it over fixed. That makes the
task slightly easier than the published v2 numbers, so do not compare
against them directly. It is also unavoidable here: the VSL renderer needs a
declared interface to type its ports against. Baseline and VSL both receive
the same fixed interface, so the comparison between them stays fair -- which
is the comparison the experiment is actually about.

Usage:
    python3 convert_v2.py verilog-eval-v2/dataset_spec-to-rtl outputs/v2 \\
        [--verify]

Writes outputs/v2_eval.jsonl and outputs/v2_desc.jsonl.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def split_ref(ref_text: str):
    """Splits a _ref.sv into (header, body). The header runs from 'module'
    through the ');' that closes the port list; the body is everything
    between that and the final endmodule."""
    text = re.sub(r"\bRefModule\b", "TopModule", ref_text).strip()

    m = re.search(r"module\s+TopModule\s*", text)
    if not m:
        raise ValueError("no TopModule declaration found")
    start = m.start()

    # walk the port list to its matching close paren
    i = text.index("(", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                break
    else:
        raise ValueError("unbalanced port list")

    semi = text.index(";", j)
    header = text[start:semi + 1]

    end = text.rindex("endmodule")
    body = text[semi + 1:end].strip("\n")
    return header, body


def run_sim(source: str, timeout: float = 90.0):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.sv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        cmd = f"iverilog -Wall -Wno-timescale -g2012 -s tb -o {d}/t.vvp {path} && vvp -n {d}/t.vvp"
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return "timeout"
    # parse the verdict before looking at TIMEOUT: several v2 testbenches
    # hit their internal cycle limit and print TIMEOUT even when correct
    m = re.search(r"Mismatches: (\d+) in (\d+) samples", r.stdout)
    if m:
        return "pass" if m.group(1) == "0" else f"fail {m.group(1)}/{m.group(2)}"
    if r.returncode != 0:
        return "compile_error"
    return "no_result"


def convert(dataset_dir):
    d = Path(dataset_dir)
    eval_records, desc_records, skipped = [], [], {}

    for prompt_file in sorted(d.glob("*_prompt.txt")):
        stem = prompt_file.name[: -len("_prompt.txt")]
        ref_file, test_file = d / f"{stem}_ref.sv", d / f"{stem}_test.sv"
        if not (ref_file.exists() and test_file.exists()):
            skipped[stem] = "missing _ref.sv or _test.sv"
            continue

        try:
            header, body = split_ref(ref_file.read_text(encoding="utf-8"))
        except Exception as e:
            skipped[stem] = f"could not split reference: {e}"
            continue

        spec = prompt_file.read_text(encoding="utf-8").strip()

        eval_records.append({
            "task_id": stem,
            "prompt": header + "\n",
            "canonical_solution": body + "\nendmodule\n",
            # the testbench needs RefModule in scope to compare against
            "test": test_file.read_text(encoding="utf-8")
                    + "\n"
                    + ref_file.read_text(encoding="utf-8"),
        })
        desc_records.append({
            "task_id": stem,
            "simple_description": spec.splitlines()[0] if spec else "",
            "detail_description": spec,
        })

    return eval_records, desc_records, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("out_prefix")
    ap.add_argument("--verify", action="store_true",
                    help="Run every problem's own reference solution through "
                         "the grading path and REPORT which ones it cannot "
                         "pass. Nothing is removed -- the benchmark is left "
                         "exactly as published.")
    ap.add_argument("--drop-broken", action="store_true",
                    help="Additionally remove the problems --verify flags. Off "
                         "by default: a problem no solution can pass is "
                         "impossible for every method equally, so keeping it "
                         "lowers all scores by the same amount and biases "
                         "nothing. Removing it, by contrast, means publishing "
                         "numbers for a benchmark that is no longer the one "
                         "everyone else ran.")
    args = ap.parse_args()

    eval_records, desc_records, skipped = convert(args.dataset_dir)
    print(f"Converted {len(eval_records)} problems")

    if args.verify:
        from concurrent.futures import ThreadPoolExecutor
        print("Verifying each reference solution through the grading path...")

        def check(rec):
            src = rec["test"] + "\n" + rec["prompt"] + "\n" + rec["canonical_solution"]
            return rec["task_id"], run_sim(src)

        with ThreadPoolExecutor(8) as ex:
            results = dict(ex.map(check, eval_records))

        bad = {k: v for k, v in results.items() if v != "pass"}
        if not bad:
            print("  all reference solutions pass")
        for tid, why in sorted(bad.items()):
            label = "DROPPED" if args.drop_broken else "KEPT (broken upstream)"
            print(f"  {label} {tid}: reference itself does not pass ({why})")

        if args.drop_broken:
            for tid, why in bad.items():
                skipped[tid] = f"reference does not pass: {why}"
            eval_records = [r for r in eval_records if r["task_id"] not in bad]
            desc_records = [r for r in desc_records if r["task_id"] not in bad]
        elif bad:
            print(f"\n  {len(bad)} problem(s) kept in place. They are impossible "
                  f"for every method equally, so they cost all scores the same\n"
                  f"  {len(bad)}/{len(eval_records)} = "
                  f"{len(bad) / len(eval_records):.1%} and bias no comparison. "
                  f"Report them as a footnote\n  rather than deleting them.")

    for kind, records in (("eval", eval_records), ("desc", desc_records)):
        path = f"{args.out_prefix}_{kind}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(records)} records to {path}")

    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for tid, why in sorted(skipped.items()):
            print(f"  {tid}: {why}")


if __name__ == "__main__":
    main()