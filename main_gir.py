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



pass_threshold = 1
max_rounds = 3
MODEL= "google-cloud:gemini-2.5-pro"
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

class GIRPort(BaseModel):
    name: str
    direction: str = Field(..., description="'input' or 'output'")
    width: str = Field(..., description="Bit range exactly as declared, e.g. '[3:0]' or '' for 1-bit")

class GIROperation(BaseModel):
    target: str = Field(..., description="Output signal or bit being driven, e.g. 'out_both[0]'")
    expression: str = Field(..., description="The logic driving it, e.g. 'in[1] & in[0]'")
    kind: str = Field(..., description="'combinational' or 'sequential'")

class GIROutput(BaseModel):
    module_name: str
    function: str
    ports: list[GIRPort] = Field(default_factory=list)
    clocking: str | None = Field("none")
    reset: str | None = Field("none")
    operations: list[GIROperation] = Field(default_factory=list)

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
    model_settings={"temperature": 0.2},
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

gir_agent = Agent(
    MODEL,
    name="GIR Agent",
    output_type=GIROutput,
    model_settings={"temperature": 0},
    system_prompt=(
        "When an operation is sequential and its target is an output port, do not "
"assume the port can be assigned procedurally. Express the intent as an "
"internal state element that the output is derived from, so the "
"implementation can respect however the fixed interface declares that port."),
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

def _read_result_field(test_result, *field_names, default=None):
    for name in field_names:
        if isinstance(test_result, dict):
            if name in test_result:
                return test_result[name]
        else:
            if hasattr(test_result, name):
                return getattr(test_result, name)
    return default

def parse_pass_fraction(error_message: str) -> float:
    """Turn a real iverilog test result into a proportional 0-1 score,
    instead of a fixed level - e.g. 'failed: 186 out of 213 samples' becomes
    (213-186)/213, so an improvement from 195->186 mismatches is visible
    in the score, not flattened to the same fixed value."""
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
 

def gir_to_description(gir) -> str:
    def s(v):
        return "" if v is None else str(v)

    lines = [f"Function: {s(gir.function)}", ""]

    if s(gir.clocking) not in ("", "none"):
        lines.append(f"Clocking: {s(gir.clocking)}")
    if s(gir.reset) not in ("", "none"):
        lines.append(f"Reset: {s(gir.reset)}")

    lines.append("\nImplement exactly these assignments:")
    for op in gir.operations or []:
        lines.append(f"  {s(op.target)} = {s(op.expression)}   [{s(op.kind)}]")

    lines.append(
        "\nAll ports are declared by the fixed interface. Do not re-declare them. "
        "If a value must be held across clock edges, introduce an internal signal "
        "and drive the output from it."
    )
    return "\n".join(lines)


async def _evaluate_prompt_async(prompt_text, eval_problem, task_id, use_gir=True):
    if use_gir:
        gir = (await gir_agent.run(prompt_text)).output
        exec_input = (
            f"{prompt_text}\n\n"
            f"--- Structured specification (authoritative) ---\n"
            f"{gir_to_description(gir)}"
        )
    else:
        exec_input = prompt_text

    execution_result = await execution_agent.run(exec_input)
    code = strip_trailing_endmodule(execution_result.output.internal_logic) + "\n\nendmodule\n"
 
    if eval_problem is None:
        print(f"WARNING: no reference problem found for task_id={task_id!r}; skipping real grading.")
        return {
            "code": code,
            "score": 0.0,
            "feedback": f"No reference problem/testbench available for task_id={task_id!r}; cannot grade.",
            "gradeable": False, 
        }
 
    from verilog_eval.execution import check_correctness

    try:
        test_result = check_correctness(problem=eval_problem, completion=code, timeout=10.0, completion_id=0)
    except Exception as e:
        return {"code": code, "score": 0.0, "feedback": f"Execution harness raised an exception: {e}", "gradeable": True}
    
    passed = bool(_read_result_field(test_result, "passed", default=False))
    if passed:
        return {"code": code, "score": 1.0, "feedback": "All testbench vectors passed.", "gradeable": True}

    error_message = _read_result_field(test_result, "result", "error", "message", default="Compilation or simulation failed.")
    score = parse_pass_fraction(str(error_message))

 
    return {"code": code, "score": score, "feedback": str(error_message), "gradeable": True}

 

async def enhance_prompt(prompt: str, mode: str = "enhanced", max_rounds: int = max_rounds, pass_threshold: float = pass_threshold, task_id: str | None = None, eval_problem: dict | None = None, use_gir: bool = False):

    if mode == "baseline":
        execution_result = await execution_agent.run(prompt)
        final_code = strip_trailing_endmodule(execution_result.output.internal_logic) + "\n\nendmodule\n"        
        return {
            "original_prompt": prompt,
            "final_output": final_code,
            
        }

    
    base_eval = await _evaluate_prompt_async(prompt, eval_problem, task_id, use_gir=use_gir)
    if base_eval["score"] >= pass_threshold:
        print(f"  raw prompt already passes (score={base_eval['score']}), skipping enhancement")
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
        "text_quality_score": None,
        "execution_score": base_eval["score"],
        "reward": None,
        "text_feedback": None,
        "execution_feedback": base_eval["feedback"],
        "change_made": "Raw prompt, no enhancement",
    }]
    reworded_result = await reworded_agent.run(prompt)
    reworded_output = reworded_result.output
    current_prompt = reworded_output.reworded_prompt

    score_result = await score_agent.run([
        "Original prompt:", prompt,
        "Revised version:", current_prompt,
    ])
    score_output = score_result.output

    exec_eval = await _evaluate_prompt_async(current_prompt, eval_problem, task_id, use_gir=use_gir)
 
    reward = compute_reward(base_eval["score"], exec_eval["score"], pass_threshold) #to compute the improvement
    history.append({
        "round": 0,
        "prompt": current_prompt,
        "code": exec_eval["code"],
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
            print(f"  round {round_num}: compile error, regenerating from same prompt")
            retry_eval = await _evaluate_prompt_async(current_prompt, eval_problem, task_id, use_gir=use_gir)
            history.append({
                "round": round_num,
                "prompt": current_prompt,
                "code": retry_eval["code"],
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
 
        revise_input = [
            "Original prompt:", prompt,
            "Current Revised version:", current_prompt,
            "Text-quality missing elements:", str(score_output.missing_elements),
            "Text-quality ambiguities:", str(score_output.ambiguities),
            "Text-quality feedback:", score_output.feedback,
            "Code that was generated from the current prompt:", exec_eval["code"],
            "Real execution result: the code generated from the current prompt "
            "FAILED verification against the reference testbench.",
            "Execution/compiler feedback:", exec_eval["feedback"],
            "Full history of every previous attempt in this session, with their "
            "actual execution scores (higher is better) - do NOT revert to an "
            "approach that already scored lower than a later attempt:",
            history_summary,
            "Revise the PROMPT (not the code) so a code-generation model following "
            "it is more likely to produce correct Verilog, building on whichever "
            "previous approach scored highest so far rather than reverting to a "
            "worse one.",
        ]
 
        revised_result = await reviser_agent.run(revise_input)
        new_prompt = revised_result.output.reworded_prompt
 
        print(f"Changes made in this round: {revised_result.output.change_made}")
 

        new_score_result = await score_agent.run([
            "Original prompt:", prompt,
            "Revised version:", new_prompt,
        ])
        new_score_output = new_score_result.output

        new_exec_eval = await _evaluate_prompt_async(new_prompt, eval_problem, task_id, use_gir=use_gir)

        reward = compute_reward(exec_eval["score"], new_exec_eval["score"], pass_threshold)
        history.append({
            "round": round_num,
            "prompt": new_prompt,
            "code": new_exec_eval["code"],
            "text_quality_score": new_score_output.score,
            "execution_score": new_exec_eval["score"],
            "reward": reward,
            "text_feedback": new_score_output.feedback,
            "execution_feedback": new_exec_eval["feedback"],
            "change_made": revised_result.output.change_made,
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

async def process_file(jsonl_file: pathlib.Path, mode: str = "enhanced", use_gir: bool = False):
    if not jsonl_file.exists():
        print(f"Error: File '{jsonl_file}' not found.")
        raise typer.Exit(code=1)

    print(f"Reading tasks from {jsonl_file}... (mode={mode})")

# Load the fixed module interface for each task_id
    
    eval_file = pathlib.Path("verilog-eval/data/VerilogEval_Machine.jsonl")
    eval_problems = {}  
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    eval_problems[data["task_id"]] = data
                    

#temporary change to save output incrementally to disk instead of all at once at the end.
    output_dir = pathlib.Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{mode}{'_gir' if use_gir else ''}_samples.jsonl"

    completed = 0
    failed = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

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

                print(f"\n Processing Task: {task_id} ")

                # A timeout or server error skips only this task instead of
                # aborting the entire run.
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
                    failed += 1
                    continue

                if mode != "baseline":
                    print(f"  Final Execution Score: {result['final_execution_score']}")
                    if result.get("history"):
                        print(f"  Feedback: {result['history'][-1]['execution_feedback'][:150]}")

                out_f.write(json.dumps({
                    "task_id": task_id,
                    "completion": result["final_output"],
                }) + "\n")
                out_f.flush()
                completed += 1

    print(f"\nSaved {completed} {mode} samples to {output_file}")
    if failed:
        print(f"{failed} task(s) failed and were skipped.")
                 
 
@app.command()
def main(
    jsonl_file: pathlib.Path = typer.Argument(...),
    mode: str = typer.Option("enhanced", help="'enhanced' or 'baseline'"),
    use_gir: bool = typer.Option(False, "--use-gir", help="Convert the prompt to GIR before generating"),
):
    asyncio.run(process_file(jsonl_file, mode=mode, use_gir=use_gir))

    """
    Read prompts from a VerilogEval JSONL file, enhance them using agents
    (with real iverilog-based feedback driving the revision loop), and save results.
    """
if __name__ == "__main__":
    app()

#later change to this:eval_file: pathlib.Path = typer.Option("verilog-eval/data/VerilogEval_Machine.jsonl",help="VerilogEval eval JSONL (interfaces + testbenches)")
