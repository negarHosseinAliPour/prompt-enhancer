import multiprocessing

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

import asyncio
import json
import os
import pathlib
import re
import sys

import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from vsl_core import (
    MODEL,
    CircuitIR,
    VSLOutput,
    VSLParseError,
    ValidationError,
    parse_vsl,
    validate_circuit,
    render_verilog,
    diff_circuit_ir,
    gir_agent,
)


pass_threshold = 1
max_rounds = 3
sys.path.append(os.path.abspath("verilog-eval"))

#--output types

class EnhancedPromptOutput(BaseModel):
    original_intent: str = Field(..., description="What the user is actually trying to accomplish")
    reworded_prompt: str = Field(..., description=("Rreworded version of the prompt"))
    reasoning: str = Field(..., description="reason behind adding improvements and constraints")

class RevisedPromptOutput(BaseModel):
    reworded_prompt: str = Field(..., description="revised prompt incorporating the feedback")
    change_made: str = Field(..., description="what changes was made compared to the previous prompt")

class scoreOutput(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score of the Revised version prompt 0=poor, 1=excellent")
    missing_elements: list[str] = Field(default_factory=list, description="Important constraints/details still missing")
    ambiguities: list[str] = Field(default_factory=list, description="Phrases that are still ambiguous or underspecified")
    feedback: str = Field(..., description="feedback on the prompt to use for next revision")

class VerilogCodeOutput(BaseModel):
    internal_logic: str = Field(..., description="ONLY the internal Verilog logic (e.g., assign statements, always blocks).\
                                 DO NOT include the module declaration header and DO NOT include the 'endmodule' keyword.")

#--agents

reworded_agent = Agent(
    MODEL,
    name="Reworded Agent",
    output_type=EnhancedPromptOutput,
    model_settings={"temperature": 0},
    system_prompt=("You are a prompt rewording agent. Your task is to take an original prompt "
        "and reword it to be more structured, adding constraints and improvements "
        "where the original is vague or incomplete — things like output format, "
        "scope, tone, or edge cases the user didn't specify.\n\n"
        "If you're given real execution/test feedback for the exact raw prompt, "
        "use it: a high execution score with a small, specific mismatch means "
        "the prompt is already mostly correct, so make only the minimal, "
        "targeted fix the feedback points to, leaving the rest verbatim. A low "
        "score or a broad range of mismatches means more substantial rewording "
        "is warranted.\n\n"
        "The prompt may include a fixed specification that the description must "
        "conform to. Always cross-check the description against it. Where the two "
        "disagree, the fixed specification is authoritative and the description is "
        "wrong. Two forms this commonly takes:\n"
        "1. The description implies a different shape or size than the specification "
        "requires — for example describing a single value where the specification "
        "calls for a collection, or vice versa.\n"
        "2. The description states quantities that don't add up against the "
        "specification — counts, sizes, or repetitions whose arithmetic is "
        "inconsistent with the declared result.\n\n"
        "Do not pass such a contradiction through. Infer the most likely intended "
        "behavior from the fixed specification, and say so explicitly in your "
        "reasoning, quoting the specific contradiction you resolved.\n\n"
        "You should provide the original intent of the prompt, the reworded prompt, "
        "and the reasoning behind each constraint or improvement you added."
    ),
 )


score_agent = Agent(
    MODEL,
    name="Score Agent",
    output_type=scoreOutput,
    model_settings={"temperature": 0},
    system_prompt=("You are a strict judge of reworded prompts. You'll be given the "\
    "original prompt and Revised version rewording of it. Score the Revised version: "\
    "from 0.0 to 1.0, based on three things: does it stay true to what the "\
    "user actually wanted, does it actually resolve the ambiguity in the "\
    "original, and does it give a clear sense of format or scope. Point "\
    "out specific things that are missing or still unclear, and give "\
    "feedback the next revision can actually act on."
    ),
)

reviser_agent = Agent(
    MODEL,
    name="Reviser Agent",
    output_type=RevisedPromptOutput,
    model_settings={"temperature": 0},
    system_prompt=("You are a prompt revision agent. You'll get the original prompt, "
        "the current version of the prompt, the ACTUAL Verilog code that was "
        "generated from that prompt, and the real result of compiling/simulating "
        "that code against the reference testbench.\n\n"
        "IMPORTANT: The module interface (the `module ... );` declaration and the "
        "final `endmodule`) is FIXED and handled automatically by the system — "
        "NEVER instruct the code generator to write, repeat, or include the module "
        "declaration or `endmodule` in your revised prompt. Only describe the "
        "internal behavior/logic.\n\n"
        "Your job is ALWAYS to revise the PROMPT — never write or fix Verilog code "
        "yourself.\n\n"
        "HOW TO DIAGNOSE: Work backwards from the generated code to the prompt. "
        "For every significant decision visible in the code, ask: did the prompt "
        "actually require this, or did the model have to choose? Any decision the "
        "model made that the prompt left open is an ambiguity, and if the test "
        "failed, one of those open decisions is likely the cause.\n\n"
        "Weight this by how much failed. A small number of mismatches against a "
        "large number of test samples means the core behavior is right and one "
        "narrow detail is wrong — look for the smallest unspecified decision that "
        "could produce exactly that pattern, rather than rewriting the overall "
        "approach. A total failure means a fundamental mismatch between what the "
        "prompt asked for and what the test expects.\n\n"
        "CRITICAL -- MAKE SURGICAL, TARGETED EDITS, NOT FULL REWRITES:\n"
        "The compiler/simulator feedback often reports mismatches PER OUTPUT "
        "SIGNAL. When this per-signal detail is available:\n"
        "  - Identify EXACTLY which output signal(s) have mismatches.\n"
        "  - Change ONLY the part of the prompt describing the logic for "
        "those specific failing signal(s).\n"
        "  - Copy every sentence describing a signal with zero mismatches "
        "into the revised prompt VERBATIM, using the exact same wording as "
        "before -- do not paraphrase or rewrite it, even if the meaning "
        "would stay the same. That logic is already correct, and even a "
        "same-meaning reword can lead the code-generation model to "
        "produce a subtly different implementation, silently breaking "
        "something that currently passes. Treat any passing signal's "
        "description as frozen text, not as prose you're free to improve.\n"
        "  - Do not restructure, reword, or 'clean up' unrelated parts of "
        "the prompt as a side effect of fixing the failing part.\n"
        "If per-signal detail is not available (e.g. a single combined "
        "mismatch count, or a compile error), then diagnose from the "
        "generated code as described above, but still change as little of "
        "the prompt as possible -- one clear, specific correction at a "
        "time, not a wholesale restructuring.\n\n"
        "Rewrite the prompt so that specific ambiguity cannot recur."),
)

execution_agent = Agent(
    MODEL,
    name="Execution Agent",
    output_type=VerilogCodeOutput,
    model_settings={"temperature": 0},
    system_prompt=(
        "You are an expert Verilog code generator and completion assistant. "
        "Given a detailed prompt describing a Verilog module, your task is to output "
        "ONLY the raw internal logic and body that implements it. "
        "The system has already defined the module interface (the `module ... );` declaration) "
        "and will automatically append the `endmodule` keyword. "
        "DO NOT include the module declaration or the `endmodule` keyword in your response. "
        "Do not include markdown code fences (no ```), do not include any explanation, "
        "comments about your reasoning, or extra text. "
        "Your output must be valid, compilable Verilog code for the module's internal logic and nothing else."
    ),
    )

#---functions

def compute_reward(score_before: float, score_after: float, pass_threshold: float = pass_threshold) -> float:

    if score_after >= pass_threshold:
        return 1.0

    improvement = score_after - score_before
    if improvement <= 0:
        return 0.0

    gap = max(pass_threshold - score_before, 1e-6)
    return round(min(improvement / gap, 0.99), 4)


def strip_trailing_endmodule(text: str) -> str:
    text = text.strip()
    if text.endswith("endmodule"):
        text = text[:-len("endmodule")].rstrip()
    return text

def _parse_vcd(vcd_text: str) -> tuple[dict, list]:
    """Reads a VCD file into (code_to_name, events). Each event is
    (time, {signal_code: value}). VCD only stores changes, so you have
    to accumulate values across events to get the full state at any time."""
    code_to_name = {}
    for m in re.finditer(r"\$var\s+(\w+)\s+(\d+)\s+(\S+)\s+(\S+)(?:\s*(\[[\d:]+\]))?\s+\$end", vcd_text):
        _vtype, _width, code, name, _range = m.groups()
        code_to_name[code] = name

    events = []
    current_time = None
    current_values: dict = {}
    for line in vcd_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current_time is not None:
                events.append((current_time, dict(current_values)))
            try:
                current_time = int(line[1:])
            except ValueError:
                current_time = None
        elif line.startswith("b"):
            parts = line.split()
            if len(parts) == 2:
                val, code = parts
                current_values[code] = val[1:]
        elif line and line[0] in "01xXzZ":
            val, code = line[0], line[1:]
            current_values[code] = val
    if current_time is not None:
        events.append((current_time, dict(current_values)))
    return code_to_name, events


def _build_mismatch_diagnostic(raw_stdout: str, vcd_text: str, max_signals: int = 12) -> str:
    """Cross-checks the mismatch time reported by the simulator against
    the VCD, and returns the exact signal values at that moment (e.g.
    r=1, g_dut=0). Returns an empty string if the VCD or the mismatch
    time isn't available."""
    m = re.search(r"First mismatch occurred at time (\d+)", raw_stdout)
    if not m or not vcd_text.strip():
        return ""
    reported_time = int(m.group(1))

    try:
        code_to_name, events = _parse_vcd(vcd_text)
    except Exception:
        return ""
    if not events:
        return ""

    name_to_code = {v: k for k, v in code_to_name.items()}
    # skip internal testbench signals like tb_mismatch, cap the count
    skip_names = {"tb_mismatch"}
    signal_names = [n for n in code_to_name.values() if n not in skip_names][:max_signals]

    # keep a window around the mismatch time since VCD timestamps might
    # not line up exactly with what the simulator reported
    running_values: dict = {}
    window: list[tuple[int, dict]] = []
    for t, changes in events:
        running_values.update(changes)
        if abs(t - reported_time) <= 10:
            window.append((t, dict(running_values)))

    if not window:
        return ""

    lines = [f"Exact signal values from the waveform around the reported mismatch time ({reported_time}):"]
    for t, snapshot in window:
        parts = [f"{name}={snapshot.get(name_to_code.get(name, ''), '?')}" for name in signal_names]
        lines.append(f"  t={t}: " + ", ".join(parts))
    return "\n".join(lines)


def parse_pass_fraction(error_message: str) -> float:
    """Turns an iverilog result into a 0-1 score. E.g. 'failed: 186 out
    of 213 samples' becomes (213-186)/213, so an improvement from 195 to
    186 mismatches actually shows up in the score."""
    msg = str(error_message).lower()

    if "passed" in msg and "failed" not in msg:
        return 1.0

    if "compil" in msg or "syntax error" in msg:
        return 0.0

    match = re.search(r"failed:\s*(\d+)\s*out of\s*(\d+)\s*samples", msg)
    if match:
        mismatches, total = int(match.group(1)), int(match.group(2))
        if total > 0:
            return round((total - mismatches) / total, 4)

    if "timeout" in msg or "timed out" in msg:
        return 0.15

    return 0.3


def check_correctness_with_details(problem: dict, completion: str, timeout: float, completion_id: int = 0) -> dict:
    """Does the same check as verilog_eval's check_correctness, but
    keeps the raw simulator output too (not just a pass/fail summary).
    This way the testbench's own $display details make it to the reviser."""
    import multiprocessing
    import re as _re
    import subprocess as _subprocess
    from threading import Timer as _Timer
    from verilog_eval.execution import (
        create_tempdir, reliability_guard, swallow_io, time_limit, TimeoutException,
    )

    def unsafe_execute(result):
        with create_tempdir():
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            reliability_guard()

            verilog_test = problem["test"] + "\n" + problem["prompt"] + "\n" + completion
            with open("{}.sv".format(problem["task_id"]), "w") as f:
                f.write(verilog_test)

            try:
                with swallow_io():
                    with time_limit(timeout):
                        cmd = ("iverilog -Wall -Winfloop -Wno-timescale -g2012 "
                               "-s tb -o test.vvp {}.sv; vvp -n test.vvp".format(problem["task_id"]))
                        p = _subprocess.Popen(cmd, shell=True, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE)
                        timer = _Timer(timeout, p.kill)
                        try:
                            timer.start()
                            out, err = p.communicate()
                        finally:
                            timer.cancel()

                        out, err = out.decode("utf-8"), err.decode("utf-8")

                        vcd_text = ""
                        if os.path.exists("wave.vcd"):
                            try:
                                with open("wave.vcd", "r", errors="replace") as vf:
                                    vcd_text = vf.read()
                            except OSError:
                                vcd_text = ""

                        match = _re.search(r"Mismatches: ([0-9]*) in ([0-9]*) samples", out)
                        if match:
                            # a valid sim result means the code actually compiled and ran,
                            # even if iverilog wrote a warning to stderr
                            cor, tot = [int(i) for i in match.groups()]
                            if cor == 0:
                                result.append(("passed", out, err, vcd_text))
                            else:
                                result.append((f"failed: {cor} out of {tot} samples.", out, err, vcd_text))
                        elif "syntax error" in err:
                            result.append(("failed: syntax error.", out, err, vcd_text))
                        elif len(err) > 0:
                            result.append(("failed: compile error.", out, err, vcd_text))
                        else:
                            result.append(("failed: info string not matched.", out, err, vcd_text))
            except TimeoutException:
                result.append(("timed out", "", "", ""))
            except BaseException as e:
                result.append((f"failed: {e}", "", "", ""))

            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=unsafe_execute, args=(result,))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append(("timed out", "", "", ""))

    summary, raw_stdout, raw_stderr, vcd_text = result[0]
    mismatch_diagnostic = ""
    if summary != "passed" and raw_stdout and vcd_text:
        try:
            mismatch_diagnostic = _build_mismatch_diagnostic(raw_stdout, vcd_text)
        except Exception:
            mismatch_diagnostic = ""

    return {
        "task_id": problem["task_id"],
        "passed": summary == "passed",
        "result": summary,
        "raw_stdout": raw_stdout,
        "raw_stderr": raw_stderr,
        "mismatch_diagnostic": mismatch_diagnostic,
        "completion_id": completion_id,
    }


async def _grade_code(code: str, eval_problem: dict | None, task_id: str | None):
    """Shared grading logic, used by both the VSL path and the execution_agent path."""
    if eval_problem is None:
        print(f"  [{task_id}] WARNING: no reference problem found; skipping real grading.")
        return {
            "code": code,
            "score": 0.0,
            "feedback": f"No reference problem/testbench available for task_id={task_id!r}; cannot grade.",
            "gradeable": False,
        }

    try:
        test_result = check_correctness_with_details(problem=eval_problem, completion=code, timeout=10.0, completion_id=0)
    except Exception as e:
        return {"code": code, "score": 0.0, "feedback": f"Execution harness raised an exception: {e}", "gradeable": True}

    passed = bool(test_result.get("passed", False))
    if passed:
        return {"code": code, "score": 1.0, "feedback": "All testbench vectors passed.", "gradeable": True}

    error_message = test_result.get("result", "Compilation or simulation failed.")
    score = parse_pass_fraction(str(error_message))

    # add the testbench's own $display output to the feedback too
    raw_stdout = test_result.get("raw_stdout", "")
    feedback = str(error_message)
    if raw_stdout.strip():
        feedback += "\n\n--- Simulator output detail ---\n" + raw_stdout.strip()

    # exact signal values at the mismatch, so the reviser can actually
    # diagnose instead of just guessing
    mismatch_diagnostic = test_result.get("mismatch_diagnostic", "")
    if mismatch_diagnostic.strip():
        feedback += "\n\n--- Exact signal values at the point of failure ---\n" + mismatch_diagnostic.strip()

    return {"code": code, "score": score, "feedback": feedback, "gradeable": True}


async def _evaluate_prompt_async(prompt_text, eval_problem, task_id, use_gir=True):
    """VSL path: description goes into gir_agent, which outputs VSL, then
    parse_vsl, validate_circuit, and render_verilog run -- all deterministic
    (no LLM) after that one model call.

    execution_agent fallback is off for now, so we can test VSL quality
    on its own. If VSL fails, we return score 0 instead of falling back
    to a different path that would hide whether the problem was VSL's fault."""
    circuit_ir = None
    vsl_text = None

    if use_gir:
        gir_result = await gir_agent.run(prompt_text)
        vsl_text = gir_result.output.vsl_code
        print(f"  [{task_id}] [DEBUG] Raw VSL from model:\n{vsl_text}\n")

        problems = None
        module_interface = eval_problem.get("prompt", "") if eval_problem else ""
        try:
            circuit_ir = parse_vsl(vsl_text, module_interface=module_interface)
            problems = validate_circuit(circuit_ir)
            print(f"  [{task_id}] [DEBUG] Validation problems: {problems or 'none'}")
        except VSLParseError as e:
            problems = [f"VSL parse error: {e}"]
            circuit_ir = None
            print(f"  [{task_id}] [DEBUG] VSL PARSE FAILED: {e}")

        if circuit_ir is not None and not problems:
            try:
                code_body = render_verilog(circuit_ir)
                code = code_body + "\n\nendmodule\n"
                print(f"  [{task_id}] [DEBUG] Deterministically rendered Verilog:\n{code}\n")
                result = await _grade_code(code, eval_problem, task_id)
                result["circuit_ir"] = circuit_ir
                result["vsl_text"] = vsl_text
                return result
            except ValidationError as e:
                problems = [f"render_verilog ValidationError: {e}"]
                print(f"  [{task_id}] [DEBUG] render_verilog raised ValidationError: {e}")

        print(f"  [{task_id}] [DEBUG] VSL FAILED (fallback disabled) -- reporting score=0: {problems}")
        return {
            "code": f"// VSL FAILED, fallback disabled\n// VSL text:\n// {vsl_text}\n",
            "score": 0.0,
            "feedback": f"VSL failed to produce valid code: {'; '.join(problems)}",
            "gradeable": True,
            "circuit_ir": circuit_ir,
            "vsl_text": vsl_text,
        }
    else:
        exec_input = prompt_text

    execution_result = await execution_agent.run(exec_input)
    code = strip_trailing_endmodule(execution_result.output.internal_logic) + "\n\nendmodule\n"
    print(f"  [{task_id}] [DEBUG] execution_agent (use_gir=False path) produced:\n{code}\n")

    result = await _grade_code(code, eval_problem, task_id)
    result["circuit_ir"] = circuit_ir
    result["vsl_text"] = vsl_text  # None on this path (execution_agent doesn't produce VSL)
    return result


async def enhance_prompt(prompt: str, mode: str = "enhanced", max_rounds: int = max_rounds, pass_threshold: float = pass_threshold, task_id: str | None = None, eval_problem: dict | None = None, use_gir: bool = False):

    if mode == "baseline":
        execution_result = await execution_agent.run(prompt)
        final_code = strip_trailing_endmodule(execution_result.output.internal_logic) + "\n\nendmodule\n"
        grade = await _grade_code(final_code, eval_problem, task_id)
        return {
            "original_prompt": prompt,
            "final_output": final_code,
            "final_execution_score": grade["score"],
            "history": [{
                "round": -1,
                "prompt": prompt,
                "execution_score": grade["score"],
                "reward": None,
                "change_made": "Baseline: raw prompt straight to execution_agent, no VSL, no revision",
                "execution_feedback": grade["feedback"],
            }],
        }


    base_eval = await _evaluate_prompt_async(prompt, eval_problem, task_id, use_gir=use_gir)
    print(f"  [{task_id}] [DEBUG] Base (raw prompt) execution_score: {base_eval['score']}")
    print(f"  [{task_id}] [DEBUG] Base (raw prompt) execution_feedback (full): {base_eval['feedback']}")
    if base_eval["score"] >= pass_threshold:
        print(f"  [{task_id}] raw prompt already passes (score={base_eval['score']}), skipping enhancement")
        return {
            "original_prompt": prompt,
            "original_intent": "n/a - raw prompt passed without enhancement",
            "history": [{
                "round": -1,
                "prompt": prompt,
                "text_quality_score": None,
                "execution_score": base_eval["score"],
                "reward": None,
                "text_feedback": None,
                "execution_feedback": base_eval["feedback"],
                "change_made": "No enhancement needed",
            }],
            "final_prompt": prompt,
            "final_execution_score": base_eval["score"],
            "final_output": base_eval["code"],
        }

    history = [{
        "round": -1,
        "prompt": prompt,
        "code": base_eval["code"],
        "circuit_ir": base_eval.get("circuit_ir"),
        "vsl_text": base_eval.get("vsl_text"),
        "text_quality_score": None,
        "execution_score": base_eval["score"],
        "reward": None,
        "text_feedback": None,
        "execution_feedback": base_eval["feedback"],
        "change_made": "Raw prompt, no enhancement",
    }]
    reworded_result = await reworded_agent.run([
        "Prompt to reword:", prompt,
        f"Execution result of testing this EXACT raw prompt, unmodified "
        f"(execution_score={base_eval['score']} out of 1.0 -- higher is "
        f"better; a score close to 1.0 means the prompt is ALREADY very "
        f"good and only needs a small, targeted fix, not a full rewrite):",
        base_eval["feedback"],
        "Code that was generated from this exact raw prompt:", base_eval["code"],
        "If the execution_score above is high (e.g. above 0.7), make ONLY "
        "the minimal change needed to address what the feedback identifies "
        "as wrong (e.g. sync vs async reset, an off-by-one, a priority "
        "order) -- copy everything else from the raw prompt VERBATIM. Do "
        "not restructure the FSM, invent a different state encoding, or "
        "otherwise rewrite parts of the prompt that had zero mismatches.",
    ])
    reworded_output = reworded_result.output
    current_prompt = reworded_output.reworded_prompt

    score_result = await score_agent.run([
        "Original prompt:", prompt,
        "Revised version:", current_prompt,
    ])
    score_output = score_result.output

    exec_eval = await _evaluate_prompt_async(current_prompt, eval_problem, task_id, use_gir=use_gir)
    print(f"  [{task_id}] [DEBUG] Round 0 execution_score: {exec_eval['score']}")
    print(f"  [{task_id}] [DEBUG] Round 0 execution_feedback (full): {exec_eval['feedback']}")

    reward = compute_reward(base_eval["score"], exec_eval["score"], pass_threshold)
    history.append({
        "round": 0,
        "prompt": current_prompt,
        "code": exec_eval["code"],
        "circuit_ir": exec_eval.get("circuit_ir"),
        "vsl_text": exec_eval.get("vsl_text"),
        "text_quality_score": score_output.score,
        "execution_score": exec_eval["score"],
        "reward": reward,
        "text_feedback": score_output.feedback,
        "execution_feedback": exec_eval["feedback"],
        "change_made": "Initial reworded prompt",
    })

    round_num = 0

    while exec_eval["score"] < pass_threshold and round_num < max_rounds:
        round_num += 1

        feedback_lower = exec_eval["feedback"].lower()
        if exec_eval["score"] == 0.0 and ("compil" in feedback_lower or "syntax" in feedback_lower):
            print(f"  [{task_id}] round {round_num}: compile error, regenerating from same prompt")
            retry_eval = await _evaluate_prompt_async(current_prompt, eval_problem, task_id, use_gir=use_gir)
            history.append({
                "round": round_num,
                "prompt": current_prompt,
                "code": retry_eval["code"],
                "circuit_ir": retry_eval.get("circuit_ir"),
                "vsl_text": retry_eval.get("vsl_text"),
                "text_quality_score": score_output.score,
                "execution_score": retry_eval["score"],
                "reward": compute_reward(exec_eval["score"], retry_eval["score"], pass_threshold),
                "text_feedback": score_output.feedback,
                "execution_feedback": retry_eval["feedback"],
                "change_made": "Regenerated after compile error (prompt unchanged)",
            })
            exec_eval = retry_eval
            continue

        history_summary = "\n\n".join([
            f"--- Attempt {h['round']} (execution_score={h['execution_score']}) ---\n"
            f"Prompt used:\n{h['prompt']}\n"
            f"Result: {h['execution_feedback']}"
            for h in history
        ])

        # always revise from the best round so far, not just the last one --
        # if a round made things worse, building on it just makes it worse
        best_so_far = max(history, key=lambda h: h["execution_score"])
        base_for_revision = best_so_far["prompt"]
        base_code_for_revision = best_so_far["code"]
        base_feedback_for_revision = best_so_far["execution_feedback"]

        previous_circuit_ir = best_so_far.get("circuit_ir")
        current_circuit_ir = exec_eval.get("circuit_ir")
        if previous_circuit_ir is not None and current_circuit_ir is not None:
            structural_changes = diff_circuit_ir(previous_circuit_ir, current_circuit_ir)
            structural_note = (
                "\n".join(structural_changes) if structural_changes
                else "No structural change detected -- the wording changed but the "
                     "underlying circuit logic did not. This is likely why the score "
                     "did not improve."
            )
        else:
            structural_note = ("CircuitIR not available for this round (use_gir may be False, "
                                "or this round used the plain-text fallback path).")

        revise_input = [
            "Original prompt:", prompt,
            "Best-scoring prompt so far (this is your BASE to revise from -- "
            f"execution_score={best_so_far['execution_score']}):", base_for_revision,
            "Code generated from that best-scoring prompt:", base_code_for_revision,
            "Execution feedback for that best-scoring prompt:", base_feedback_for_revision,
            "Text-quality missing elements:", str(score_output.missing_elements),
            "Text-quality ambiguities:", str(score_output.ambiguities),
            "Text-quality feedback:", score_output.feedback,
            "Most recent attempt's code (for reference; this may score lower "
            "than the best-scoring prompt above -- diagnose the mistake here, "
            "but apply the fix to the BEST-SCORING prompt, not this one):",
            exec_eval["code"],
            "Most recent attempt's execution feedback:", exec_eval["feedback"],
            "Exact structural changes in circuit logic between the best-scoring "
            "attempt and the most recent attempt:",
            structural_note,
            "Full history of every previous attempt in this session, with their "
            "actual execution scores (higher is better) - do NOT revert to an "
            "approach that already scored lower than a later attempt:",
            history_summary,
            "IMPORTANT: Your revised prompt MUST be based on the BEST-SCORING "
            "prompt above, with only the specific fix applied for the mistake "
            "you diagnosed from the most recent attempt. Do not start from or "
            "build on the most recent attempt's prompt if it scored lower than "
            "the best-scoring one -- that would carry forward whatever mistake "
            "made it score lower in the first place.",
            "Revise the PROMPT (not the code) so a code-generation model following "
            "it is more likely to produce correct Verilog, building on whichever "
            "previous approach scored highest so far rather than reverting to a "
            "worse one.",
        ]

        # if the reviser's change had no effect on the generated code
        # (code is identical to before), don't waste a round -- retry instead
        MAX_NOOP_RETRIES = 2
        noop_note = ""
        noop_attempt = 0
        for noop_attempt in range(MAX_NOOP_RETRIES + 1):
            attempt_revise_input = list(revise_input)
            if noop_note:
                attempt_revise_input += [
                    "CRITICAL: your previous edit in this same round changed the "
                    "prompt's wording but produced BYTE-IDENTICAL generated code "
                    "to the best-scoring prompt above. Nothing you changed "
                    "actually affects what the code-generation model produces. "
                    "You MUST propose a substantively different fix this time -- "
                    "e.g. a different signal/bit mapping, a different reset "
                    "scope or polarity, or a different edge direction -- not "
                    "just clearer wording of the same instruction.",
                    noop_note,
                ]

            revised_result = await reviser_agent.run(attempt_revise_input)
            new_prompt = revised_result.output.reworded_prompt
            print(f"  [{task_id}] Changes made in this round (attempt {noop_attempt}): {revised_result.output.change_made}")

            new_score_result = await score_agent.run([
                "Original prompt:", prompt,
                "Revised version:", new_prompt,
            ])
            new_score_output = new_score_result.output

            new_exec_eval = await _evaluate_prompt_async(new_prompt, eval_problem, task_id, use_gir=use_gir)
            print(f"  [{task_id}] [DEBUG] Round {round_num} execution_score (attempt {noop_attempt}): {new_exec_eval['score']}")
            print(f"  [{task_id}] [DEBUG] Round {round_num} execution_feedback (full, attempt {noop_attempt}): {new_exec_eval['feedback']}")

            is_noop = new_exec_eval["code"] == base_code_for_revision
            if is_noop and noop_attempt < MAX_NOOP_RETRIES:
                print(f"  [{task_id}] [DEBUG] Round {round_num} attempt {noop_attempt}: no-op (code identical to best-so-far) -- retrying without consuming a round")
                noop_note = (
                    f"(This was retry {noop_attempt + 1} of {MAX_NOOP_RETRIES} within the same round; still no-op so far.)"
                )
                continue
            if is_noop:
                print(f"  [{task_id}] [DEBUG] Round {round_num}: exhausted no-op retries, accepting identical result")
            break

        reward = compute_reward(exec_eval["score"], new_exec_eval["score"], pass_threshold)
        print(f"  [{task_id}] [DEBUG] Round {round_num} reward (vs previous round's score {exec_eval['score']}): {reward}")
        history.append({
            "round": round_num,
            "prompt": new_prompt,
            "code": new_exec_eval["code"],
            "circuit_ir": new_exec_eval.get("circuit_ir"),
            "vsl_text": new_exec_eval.get("vsl_text"),
            "text_quality_score": new_score_output.score,
            "execution_score": new_exec_eval["score"],
            "reward": reward,
            "text_feedback": new_score_output.feedback,
            "execution_feedback": new_exec_eval["feedback"],
            "change_made": revised_result.output.change_made,
            "noop_retries_used": noop_attempt,
        })

        current_prompt = new_prompt
        score_output = new_score_output
        exec_eval = new_exec_eval

    best = max(history, key=lambda h: h["execution_score"])
    return {
        "original_prompt": prompt,
        "original_intent": reworded_output.original_intent,
        "history": history,
        "final_prompt": best["prompt"],
        "final_execution_score": best["execution_score"],
        "final_output": best["code"],
    }



#--CLI

app = typer.Typer(help="Prompt enhancer with shaped-reward scoring across revision rounds.")

async def process_file(jsonl_file: pathlib.Path, mode: str = "enhanced", use_gir: bool = False, concurrency: int = 6):
    if not jsonl_file.exists():
        print(f"Error: File '{jsonl_file}' not found.")
        raise typer.Exit(code=1)

    print(f"Reading tasks from {jsonl_file}... (mode={mode}, concurrency={concurrency})")

    eval_file = pathlib.Path("verilog-eval/data/VerilogEval_Machine.jsonl")
    eval_problems = {}
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    eval_problems[data["task_id"]] = data

    output_dir = pathlib.Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{mode}{'_vsl' if use_gir else ''}_samples_Machine.jsonl"
    history_file = output_dir / f"{mode}{'_vsl' if use_gir else ''}_history_Machine.jsonl"

    tasks_data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks_data.append(json.loads(line))

    completed = 0
    failed = 0
    # lock so concurrent tasks don't mess up the file when writing at once
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(data: dict):
        nonlocal completed, failed
        task_id = data.get("task_id")
        simple_desc = data.get("simple_description", "")
        detail_desc = data.get("detail_description", "")

        problem = eval_problems.get(task_id)
        interface = problem["prompt"] if problem else ""
        prompt_text = (
            f"Summary: {simple_desc}\n"
            f"Detailed Description: {detail_desc}\n\n"
            f"The module interface is FIXED and must be used exactly as-is "
            f"(do not rename the module or any port):\n{interface}"
        )

        async with semaphore:
            print(f"\n Processing Task: {task_id} ")
            print(f"  [{task_id}] [DEBUG] Module interface given to gir_agent:\n{interface}\n")

            try:
                result = await enhance_prompt(
                    prompt_text,
                    mode=mode,
                    task_id=task_id,
                    eval_problem=problem,
                    use_gir=use_gir
                )
            except Exception as e:
                print(f"  ERROR on {task_id}: {type(e).__name__}: {e}")
                print("  skipping this task")
                async with write_lock:
                    failed += 1
                return

        print(f"\n  [{task_id}] Final Execution Score: {result['final_execution_score']}")
        if result.get("history"):
            print(f"  [{task_id}] Feedback: {result['history'][-1]['execution_feedback'][:150]}")

        history_record = {
            "task_id": task_id,
            "final_execution_score": result["final_execution_score"],
            "rounds": [
                {
                    "round": h["round"],
                    "execution_score": h["execution_score"],
                    "reward": h.get("reward"),
                    "change_made": h.get("change_made"),
                    "execution_feedback": h.get("execution_feedback"),
                    "prompt": h.get("prompt"),
                    "vsl_text": h.get("vsl_text"),
                }
                for h in result.get("history", [])
            ],
        }

        async with write_lock:
            with open(history_file, "a", encoding="utf-8") as hist_f:
                hist_f.write(json.dumps(history_record) + "\n")
            with open(output_file, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps({
                    "task_id": task_id,
                    "completion": result["final_output"],
                }) + "\n")
            completed += 1

    # clear the output files before the concurrent runs start
    open(output_file, "w", encoding="utf-8").close()
    open(history_file, "w", encoding="utf-8").close()

    await asyncio.gather(*(_run_one(data) for data in tasks_data))

    print(f"\nSaved {completed} {mode} samples to {output_file}")
    print(f"Saved per-round history to {history_file}")
    if failed:
        print(f"{failed} task(s) failed and were skipped.")


@app.command()
def main(
    jsonl_file: pathlib.Path = typer.Argument(...),
    mode: str = typer.Option("enhanced", help="'enhanced' or 'baseline'"),
    use_gir: bool = typer.Option(False, "--use-gir", help="Convert the prompt to VSL before generating"),
):
    asyncio.run(process_file(jsonl_file, mode=mode, use_gir=use_gir))

    """
    Reads prompts from a JSONL file, improves them with agents (using
    real iverilog feedback to drive the revision loop), and saves the results.
    """
if __name__ == "__main__":
    app()
