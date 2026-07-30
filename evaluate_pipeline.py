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

def main():
    #problem_file = "verilog-eval/data/example/ExampleEval.jsonl"
    problem_file = "verilog-eval/data/VerilogEval_Machine.jsonl"
    enhanced_sample = "outputs/enhanced_gir_samples.jsonl"
    #enhanced_sample = "outputs/enhanced_samples.jsonl"

    val_output_dir = pathlib.Path("val-output")
    val_output_dir.mkdir(exist_ok=True)

    if not pathlib.Path(enhanced_sample).exists():
        print(f"Error: Enhanced sample file '{enhanced_sample}' not found. Please run main2.py first.")
        return

    done_ids = {json.loads(l)["task_id"] for l in open(enhanced_sample) if l.strip()}

    subset_problem_file = "outputs/problems_subset.jsonl"
    with open(problem_file) as fin, open(subset_problem_file, "w") as fout:
        for line in fin:
            if json.loads(line)["task_id"] in done_ids:
                fout.write(line)

    print(f"--- Evaluating {len(done_ids)} completed tasks ---")


    
    results = evaluate_functional_correctness(
        sample_file=enhanced_sample,
        problem_file=subset_problem_file,
        k=[1],
        n_workers=4,
        timeout=30.0,
        unit_test=False,
        clean_up=False,
    )
    
    print("Evaluation Results:", results)
    
    out_file = val_output_dir / "enhanced_gir_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved evaluation results to {out_file}")

if __name__ == "__main__":
    main()