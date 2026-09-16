# prompt-enhancer

Execution-guided prompt refinement for Verilog code generation, built on Pydantic AI.

The core idea: instead of hand-writing a fixed system prompt for an LLM that writes Verilog, we let the model *discover its own intermediate representation* (VSL -- a small, constrained grammar sitting between natural-language specs and Verilog) by running it against real problems, watching where it fails, and iteratively growing the grammar's rules from those failures. `discover_grammar.py` is that discovery loop; `main_vsl.py` is the harness that runs a fixed grammar against a dataset (with a self-healing/revision loop) and records everything needed to score it -- not just pass/fail, but token usage, wall-clock time, and how many revision rounds it took to converge.

## How the pieces fit together

1. **Get data.** Verilog problems come from a mix of public benchmarks (VerilogEval v1, VerilogEval v2, RTLLM) and Hugging Face datasets (RTL-Coder, MG-Verilog, VeriReason). Some datasets ship with a testbench already; some don't and need one built (`equivalence_testbench/build_equivalence_eval.py` does this by auto-generating and self-checking an equivalence testbench against the dataset's own reference solution).
2. **Discover a grammar.** `discover_grammar.py` takes a pool of real, graded run history (prompt -> VSL code -> execution score) plus a larger raw problem pool (structure only, no grading needed), and iteratively proposes grammar rules, tests them, and keeps what improves results.
3. **Run the grammar.** `main_vsl.py` takes a finished grammar and a dataset, and runs every task through it end-to-end: generate VSL, parse it, validate it, render it to Verilog, grade it, and if it fails, feed the error back and try again (up to a configurable number of rounds). It resumes cleanly if interrupted -- rerunning it after a crash skips whatever's already in the history file rather than redoing it.
4. **Evaluate.** `compute_metrics.py` reduces a full history file down to the two headline numbers: pass@1 (first attempt, no revision) and resolve rate (best score after the full self-healing loop). `add_dataset_summary.py` appends one row per run to `outputs/vsl_master_summary.csv` so different runs/datasets can be compared side by side.

## Top-level layout

**Pipeline scripts (project root)**
- `vsl_core.py` -- the VSL language itself: the Pydantic AI agent that generates it (`gir_agent`), the parser (`parse_vsl`), the structural validator (`validate_circuit`), and the Verilog renderer (`render_verilog`). Everything else in the project is built on top of these four things.
- `discover_grammar.py` -- the grammar discovery loop described above.
- `main_vsl.py` -- the main run harness: takes a grammar + a dataset, runs the generate/validate/grade/self-heal loop over every task, writes history + per-task output files, and resumes safely from where a previous run left off.
- `compute_metrics.py` -- computes pass@1 and resolve-rate over an entire history file (the auto-generated per-run summaries only reflect whatever subset of tasks happened to run in that particular invocation, so this is the script that gives a trustworthy whole-dataset number).
- `add_dataset_summary.py` -- appends a row of metrics for one dataset/run to the master comparison CSV.
- `build_rtllm_data.py` -- walks the raw `RTLLM/` benchmark folder and builds `datasets/rtllm/rtllm_desc.jsonl` / `rtllm_eval.jsonl`.
- `convert_v2.py` -- converts the official VerilogEval v2 dataset into the same `{task_id, prompt, canonical_solution, test}` shape used everywhere else in the project.
- `eval_baseline.py` / `eval_pipeline.py` -- functional-correctness evaluation runners (baseline model output vs. VSL-enhanced output) against a given problem set.

**Grammar files (project root)**
- `grammar.txt`, `discovered_grammar.txt` -- the current finalized grammar and its raw discovery output.
- `grammar_base_rules.txt` -- an optional hand-written starting point that can be fed into `discover_grammar.py` via `--base-rules`, to compare "grammar discovered from scratch" against "grammar discovered starting from a known-good base."

Grammar files are never edited by hand -- they only ever come out of `discover_grammar.py`'s automated loop (the one exception being `--base-rules`, which injects a *starting point*, not a finished grammar).

**Data and outputs**
- `datasets/` -- the canonical, ready-to-use problem pools for evaluation, in the shared `{task_id, prompt/description, canonical_solution, test}` shape: `rtllm/` and `verilogeval_v2/` so far, each with a `_desc.jsonl` (ungraded, structure only) and `_eval.jsonl` (fully graded) pair.
- `grammar_training_data/` -- everything specific to feeding the grammar discovery loop:
  - `converted/` -- raw datasets turned into the `{task_id, description, module_interface}` structural pools discovery needs, plus the older, pre-existing combined reference pool.
  - `history/` -- graded run history from earlier validation/training passes.
  - `scripts/` -- the small conversion/validation scripts that produce the above (`build_mgverilog_problems.py`, `build_rtlcoder_problems.py`, `build_verireason_problems.py`, `grade_verireason.py`, `run_verireason_pipeline.py`, `validate_grammar.py`).
- `equivalence_testbench/` -- `build_equivalence_eval.py`, which turns a raw dataset that only has reference Verilog (no testbench) into a properly graded eval/desc pair, by generating and self-verifying an equivalence testbench against the reference.
- `outputs/` -- one subfolder per dataset/run (`rtlcoder_equiv/`, `mgverilog_equiv/`, `verilogeval_v1/`, `verilogeval_v2/`, `rtllm/`), each holding that run's history/samples/eval files, plus the merged pools ready for the next grammar-discovery run (`equiv_combined_history.jsonl`, `equiv_combined_problems.jsonl` -- RTL-Coder and MG-Verilog combined, 5,808 tasks total) and `vsl_master_summary.csv`.

**External benchmarks (as-downloaded, unmodified)**
- `RTLLM/` -- the raw RTLLM benchmark, one folder per task (`design_description.txt`, `testbench.v`, a `verified_*.v` reference).
- `verilog-eval/` -- the official VerilogEval v1 toolkit and dataset.
- `verilog-eval-v2/` -- the official VerilogEval v2 toolkit and dataset (`dataset_code-complete-iccad2023/`, `dataset_spec-to-rtl/`), including its own evaluation helpers (`count_failures.py`, `pass_rate_to_csv.py`) for breaking down *why* a task failed (syntax error, timeout, bind error, etc.) once it's actually been run.

## Where things stand

The current grammar (`grammar.txt`) has been fully evaluated on VerilogEval v1, VerilogEval v2, and RTLLM (results under `outputs/verilogeval_v1/`, `outputs/verilogeval_v2/`, `outputs/rtllm/`). RTL-Coder and MG-Verilog have been run and merged into a single 5,808-task pool (`outputs/equiv_combined_*.jsonl`), ready to feed into the next round of grammar discovery -- once with no base grammar, once with `grammar_base_rules.txt` as a starting point, so the two can be compared before deciding which one to carry forward into the next full evaluation cycle.
