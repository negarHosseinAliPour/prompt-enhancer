"""
Evaluates the true pass@1 (round -1 only, before any rewording/revision)
using the round_minus1_samples_*.jsonl files produced by
extract_round_minus1_samples.py.

This is the counterpart to eval_pipeline.py, which evaluates the final
(oracle-selected, best-of-4) samples files instead -- that one gives the
resolve-rate number, this one gives pass@1.

Usage:
    python3 eval_pass_at_1.py
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
        print(f"Error: Sample file '{sample_file}' not found. "
              f"Run extract_round_minus1_samples.py first.")
        return

    done_ids = {json.loads(l)["task_id"] for l in open(sample_file) if l.strip()}

    subset_problem_file = f"outputs/problems_subset_pass1_{out_name}.jsonl"
    with open(problem_file) as fin, open(subset_problem_file, "w") as fout:
        for line in fin:
            if json.loads(line)["task_id"] in done_ids:
                fout.write(line)

    print(f"--- Evaluating true pass@1 on {len(done_ids)} tasks for {out_name} (round -1 only) ---")

    results = evaluate_functional_correctness(
        sample_file=sample_file,
        problem_file=subset_problem_file,
        k=[1],
        n_workers=4,
        timeout=30.0,
        unit_test=False,
        clean_up=False,
    )

    print(f"[{out_name}] TRUE pass@1 Results:", results)

    out_file = val_output_dir / f"true_pass_at_1_results_{out_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved to {out_file}")


def main():
    run_eval(
        "verilog-eval/data/VerilogEval_Machine.jsonl",
        "outputs/round_minus1_samples_machine.jsonl",
        "machine",
    )
    run_eval(
        "verilog-eval/data/VerilogEval_Human.jsonl",
        "outputs/round_minus1_samples_human.jsonl",
        "human",
    )


if __name__ == "__main__":
    main()