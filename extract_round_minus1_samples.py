"""
Extracts round -1 (the very first, unmodified attempt -- no rewording,
no revision, no access to the reference testbench) from a history.jsonl
file and re-renders its VSL text into Verilog, writing the result out
as a samples.jsonl file in the format eval_pipeline.py expects.

Usage:
    python3 extract_round_minus1_samples.py \
        outputs/enhanced_vsl_history_machine.jsonl \
        verilog-eval/data/VerilogEval_Machine.jsonl \
        outputs/round_minus1_samples_machine.jsonl \
        outputs/enhanced_vsl_samples_machine.jsonl
"""
import json
import sys

sys.path.insert(0, ".")  # so vsl_core.py is importable if it's in the cwd

from vsl_core import parse_vsl, validate_circuit, render_verilog, VSLParseError, ValidationError


def main():
    if len(sys.argv) not in (4, 5):
        print("Usage: python3 extract_round_minus1_samples.py <history.jsonl> <problems.jsonl> <output_samples.jsonl> [final_samples.jsonl]")
        sys.exit(1)

    history_file, problems_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    final_samples_file = sys.argv[4] if len(sys.argv) == 5 else None

    # module interface per task_id, needed to re-render VSL into Verilog
    module_interfaces = {}
    with open(problems_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            module_interfaces[data["task_id"]] = data.get("prompt", "")

    # completions from the final samples file, if provided -- used as a
    # fallback for round -1 records that have no vsl_text but did pass,
    # since in that case round -1's own completion IS the final output
    # (no revision happened).
    final_completions = {}
    if final_samples_file:
        with open(final_samples_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                final_completions[data["task_id"]] = data.get("completion", "")

    written = 0
    no_round = 0
    no_vsl_text = 0
    recovered_from_final = 0

    with open(history_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            task_id = rec["task_id"]

            round_minus1 = next(
                (r for r in rec.get("rounds", []) if r.get("round") == -1),
                None,
            )
            if round_minus1 is None:
                # No round -1 at all for this task -- this should not
                # normally happen (every task's history starts with a
                # round -1 attempt), but if it does, we still need this
                # task counted in the pass@1 denominator. Emit a
                # guaranteed-fail completion rather than silently
                # dropping the task from the samples file, which would
                # shrink n and inflate pass@1.
                print(f"WARNING: no round -1 found for {task_id} -- "
                      f"counting as a fail so it's still in the denominator.")
                code = "// no round -1 recorded in history\nendmodule\n"
                fout.write(json.dumps({"task_id": task_id, "completion": code}) + "\n")
                written += 1
                no_round += 1
                continue

            vsl_text = round_minus1.get("vsl_text")
            round_score = round_minus1.get("execution_score")
            module_interface = module_interfaces.get(task_id, "")

            if vsl_text is None:
                # round -1 has no VSL logged. If the recorded score for
                # this round was already 1.0, either (a) the raw prompt
                # passed without needing VSL at all (non-VSL shortcut), or
                # (b) this is round -1 of a task whose final result equals
                # round -1 (no revision needed). In both cases the samples
                # file's completion for this task_id IS what was actually
                # graded for this attempt, so use it directly instead of
                # guessing.
                no_vsl_text += 1
                if round_score == 1.0 and task_id in final_completions:
                    code = final_completions[task_id]
                    recovered_from_final += 1
                    print(f"NOTE: round -1 for {task_id} passed without a "
                          f"logged VSL (non-VSL path) -- recovered its actual "
                          f"completion from the final samples file.")
                elif round_score == 1.0:
                    print(f"WARNING: round -1 for {task_id} passed without a "
                          f"logged VSL, and no final_samples_file was given "
                          f"(or task_id missing from it) -- cannot recover the "
                          f"real completion, counting as a fail to avoid "
                          f"overstating pass@1 with a fake pass.")
                    code = "// PASSED round -1 but completion text could not " \
                           "be recovered -- counted as fail, not a real pass\nendmodule\n"
                else:
                    print(f"WARNING: round -1 for {task_id} has no vsl_text "
                          f"logged and did not pass (score={round_score}) -- "
                          f"counting as a fail.")
                    code = "// no vsl_text logged for round -1, and it did " \
                           "not pass\nendmodule\n"
                fout.write(json.dumps({"task_id": task_id, "completion": code}) + "\n")
                written += 1
                continue

            try:
                ir = parse_vsl(vsl_text, module_interface=module_interface)
                problems = validate_circuit(ir)
                if problems:
                    # A VSL that fails validation never produces Verilog --
                    # the pipeline reports this as score 0.0 for round -1,
                    # so the "completion" is empty / guaranteed to fail.
                    code = "// VSL failed validation\nendmodule\n"
                else:
                    code = render_verilog(ir) + "\n\nendmodule\n"
            except VSLParseError:
                code = "// VSL failed to parse\nendmodule\n"
            except ValidationError:
                code = "// VSL failed validation\nendmodule\n"

            fout.write(json.dumps({"task_id": task_id, "completion": code}) + "\n")
            written += 1

    print(f"Wrote {written} samples to {output_file} (every task in the history is represented)")
    if no_round:
        print(f"{no_round} task(s) had no round -1 at all -- counted as fails")
    if no_vsl_text:
        print(f"{no_vsl_text} task(s) had no vsl_text logged for round -1 -- "
              f"{recovered_from_final} recovered from final samples, "
              f"{no_vsl_text - recovered_from_final} counted as fails (see WARNING lines above)")


if __name__ == "__main__":
    main()