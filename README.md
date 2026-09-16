# prompt-enhancer
Execution-guided prompt refinement for Verilog code generation, built on Pydantic AI.

## Project layout

Top-level folders, so far:

- **equivalence_testbench/** -- `build_equivalence_eval.py`, the script that turns a raw dataset with reference Verilog code (but no testbench) into a genuinely graded eval/desc pair by auto-generating and self-verifying an equivalence testbench.
- **grammar_checks/** -- one-off scripts and results used to sanity-check a discovered grammar (e.g. `verify_grammar_generalization.py`).
- **grammar_training_data/** -- the raw datasets (`raw/`), converted eval/desc pools (`converted/`), run histories (`history/`), the discover_grammar.py scripts used along the way (`scripts/`), and the grammar files/round-logs produced by earlier discovery runs.
- **outputs/** -- one subfolder per dataset/run, holding that run's eval/desc/history/sample files (e.g. `rtlcoder_equiv/`, `mgverilog_equiv/`, `verilogeval_v1/`, `verilogeval_v2/`, `rtllm/`, `baseline_outputs/`, `checkpoints/`), plus the combined pools (`equiv_combined_history.jsonl`, `equiv_combined_problems.jsonl`) and summary reports.
- **RTLLM/** -- the raw RTLLM benchmark (one folder per task, each with `design_description.txt`, `testbench.v`, and a `verified_*.v` reference solution), used by `build_rtllm_data.py`.
- **val-output/** -- functional-correctness evaluation results (from `eval_pipeline.py` / `eval_baseline.py`), one JSON per run.
- **verilog-eval/** -- the official VerilogEval v1 toolkit/dataset.
- **verilog-eval-v2/** -- the official VerilogEval v2 toolkit/dataset (`dataset_code-complete-iccad2023/`, `dataset_spec-to-rtl/`, plus its own `scripts/`).

Top-level files are the pipeline scripts themselves (`main_vsl.py`, `vsl_core.py`, `discover_grammar.py`, `compute_metrics.py`, the `build_*`/`convert_*` dataset converters, `add_dataset_summary.py`) and the live grammar files (`grammar.txt`, `grammar_base_rules.txt`, `discovered_grammar.txt`).

(To be continued -- reviewing the remaining folders next.)
