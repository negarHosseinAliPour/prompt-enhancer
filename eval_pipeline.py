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

    subset_problem_file = f"outputs/problems_subset_{out_name}.jsonl"
    with open(problem_file) as fin, open(subset_problem_file, "w") as fout:
        for line in fin:
            if json.loads(line)["task_id"] in done_ids:
                fout.write(line)

    print(f"--- Evaluating {len(done_ids)} completed tasks for {out_name} ---")

    results = evaluate_functional_correctness(
        sample_file=sample_file,
        problem_file=subset_problem_file,
        k=[1],
        n_workers=4,
        timeout=30.0,
        unit_test=False,
        clean_up=False,
    )

    print(f"[{out_name}] Evaluation Results:", results)

    out_file = val_output_dir / f"enhanced_vsl_eval_results_{out_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved evaluation results to {out_file}")


def main():
    run_eval(
    "verilog-eval/data/VerilogEval_Machine_real.jsonl",
    "outputs/enhanced_vsl_forcevsl_samples_gptoss_machine_TRUE_FINAL.jsonl",
    "gptoss_machine_forcevsl_final",
)

if __name__ == "__main__":
    main()