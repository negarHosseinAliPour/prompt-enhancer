# prompt-enhancer

An execution-guided pipeline for generating correct Verilog from natural-language specs, built on Pydantic AI.

**Author:** Negar Hosseinali Pour

## Project structure

**Root**
- `vsl_core.py` -- the VSL language: the agent that writes it, the parser, the validator, and the Verilog renderer.
- `discover_grammar.py` -- builds the grammar file from past successful runs.
- `main_vsl.py` -- runs a grammar against a full dataset (generate, validate, grade, self-heal, resume).
- `compute_metrics.py` -- computes pass@1 and resolve-rate from a history file.
- `add_dataset_summary.py` -- appends a run's metrics to `outputs/vsl_master_summary.csv`.
- `build_rtllm_data.py` -- builds the RTLLM eval/desc pool from `RTLLM/`.
- `convert_v2.py` -- converts the official VerilogEval v2 dataset into the project's standard format.
- `eval_baseline.py`, `eval_pipeline.py` -- functional-correctness evaluation runners.
- `grammar.txt`, `discovered_grammar.txt` -- the current grammar (never edited by hand).
- `grammar_base_rules.txt` -- optional starting grammar, passed to `discover_grammar.py` via `--base-rules`.

**`equivalence_testbench/`**
- `build_equivalence_eval.py` -- generates and self-checks a testbench for datasets that only ship a reference solution, no testbench.

**`grammar_training_data/`**
- `converted/` -- raw datasets converted into the structural pools discovery needs.
- `history/` -- graded run history used to build past grammars.
- `scripts/` -- the conversion/grading/validation scripts for these datasets (`build_mgverilog_problems.py`, `build_rtlcoder_problems.py`, `build_verireason_problems.py`, `grade_verireason.py`, `run_verireason_pipeline.py`, `validate_grammar.py`).

**`datasets/`**
- `rtllm/`, `verilogeval_v2/` -- ready-to-use problem pools (`_desc.jsonl` and `_eval.jsonl` pairs).

**`outputs/`**
- `rtlcoder_equiv/`, `mgverilog_equiv/`, `verilogeval_v1/`, `verilogeval_v2/`, `rtllm/` -- history/samples/eval files per run.
- `equiv_combined_history.jsonl`, `equiv_combined_problems.jsonl` -- merged RTL-Coder + MG-Verilog pool (5,808 tasks).
- `vsl_master_summary.csv` -- one row per run, for comparing runs/datasets.

**External benchmarks (unmodified)**
- `RTLLM/` -- raw RTLLM benchmark, one folder per task.
- `verilog-eval/` -- official VerilogEval v1 toolkit and dataset.
- `verilog-eval-v2/` -- official VerilogEval v2 toolkit and dataset.
