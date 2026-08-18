"""
Evaluates the no-VSL baseline (raw description straight to execution_agent,
one single attempt, no revision loop). For this mode, pass@1 and the
resolve rate are the same number, since there is only one round.

Usage:
    python3 eval_baseline.py
"""
import multiprocessing
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

import sys
import pathlib
import json
import os
sys.path.append(os.path.abspath("verilog-eval"))

from verilog_eval.evaluation import evaluate_functional_correctness


def run_eval(problem_file, sample_file, out_name):
    val_output_dir = pathlib.Path("val-output")
    val_output_dir.mkdir(exist_ok=True)

    if not pathlib.Path(sample_file).exists():
        print(f"Error: Sample file '{sample_file}' not found.")
        return

    done_ids = {json.loads(l)["task_id"] for l in open(sample_file) if l.strip()}

    subset_problem_file = f"outputs/problems_subset_baseline_{out_name}.jsonl"
    with open(problem_file) as fin, open(subset_problem_file, "w") as fout:
        for line in fin:
            if json.loads(line)["task_id"] in done_ids:
                fout.write(line)

    print(f"--- Evaluating baseline (no VSL) on {len(done_ids)} tasks for {out_name} ---")

    results = evaluate_functional_correctness(
        sample_file=sample_file,
        problem_file=subset_problem_file,
        k=[1],
        n_workers=4,
        timeout=30.0,
        unit_test=False,
        clean_up=False,
    )

    print(f"[{out_name}] Baseline pass@1 (== resolve rate, single attempt only): {results}")

    out_file = val_output_dir / f"baseline_eval_results_{out_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved to {out_file}")


def main():
    run_eval(
        "verilog-eval/data/VerilogEval_Machine.jsonl",
        "outputs/baseline_samples_machine.jsonl",
        "machine",
    )
    run_eval(
        "verilog-eval/data/VerilogEval_Human.jsonl",
        "outputs/baseline_samples_human.jsonl",
        "human",
    )


if __name__ == "__main__":
    main()