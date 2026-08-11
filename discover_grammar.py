"""
Grammar-discovery loop for VSL.

Instead of a human hand-writing the VSL_GRAMMAR_AND_EXAMPLES system prompt
(the current approach in vsl_core.py), this script lets an LLM propose the
grammar itself: it looks at many real (description, successful VSL) pairs
from past pipeline runs, proposes a grammar, tests that grammar by running
gir_agent against it on a sample of descriptions, sees how well it did, and
refines the grammar again. This repeats for a fixed number of rounds.

This is a ONE-TIME (offline) process, not something that runs on every
pipeline call. The output is a single grammar string that can replace
VSL_GRAMMAR_AND_EXAMPLES in vsl_core.py.

Usage:
    python discover_grammar.py --history enhanced_vsl_history.jsonl --rounds 8 --sample-size 150
"""

import asyncio
import json
import random
import re
from pathlib import Path

import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from vsl_core import MODEL, parse_vsl, validate_circuit, render_verilog, VSLParseError, ValidationError

app = typer.Typer()


# --- output types -----------------------------------------------------------

class GrammarProposal(BaseModel):
    grammar_text: str = Field(
        ...,
        description=(
            "The full VSL grammar and worked examples, written as a system "
            "prompt for a model that will translate English descriptions "
            "into VSL. Must be self-contained: a model given ONLY this text "
            "plus a description should be able to produce correct VSL."
        ),
    )
    reasoning: str = Field(
        ...,
        description="What changed from the previous version and why, based on the test feedback.",
    )


class DiscoveryVSLOutput(BaseModel):
    """Same shape as VSLOutput in vsl_core.py, but kept local so this script
    can run gir-style calls against a *candidate* grammar instead of the
    fixed one baked into vsl_core.gir_agent."""
    vsl_code: str = Field(..., description="Circuit logic written in VSL, following the grammar in the system prompt.")


discovery_agent = Agent(
    MODEL,
    name="Grammar Discovery Agent",
    output_type=GrammarProposal,
    model_settings={"temperature": 0.3},
    system_prompt=(
        "Design a compact grammar (VSL) that lets a language model translate "
        "English Verilog descriptions into an unambiguous intermediate form, "
        "later parsed and rendered into Verilog by fixed Python code (no "
        "model involved in that step). You design the grammar; you do not "
        "write Verilog yourself.\n\n"
        "You'll see real (description, VSL) pairs that previously scored a "
        "perfect execution score. Find the recurring patterns -- especially "
        "English ambiguities that needed an explicit VSL construct (shift "
        "vs. rotate, sync vs. async reset, condition priority) -- and "
        "propose a grammar that generalizes them.\n\n"
        "On later rounds you'll also see how your previous grammar performed "
        "on a fresh sample: what passed, what failed, and why. Fix the "
        "specific gaps rather than rewriting from scratch.\n\n"
        "Output a complete, self-contained grammar: another model given only "
        "your grammar_text plus a new description must produce correct VSL."
    ),
)


# --- data loading ------------------------------------------------------------

def load_successful_examples(history_path: Path) -> list[dict]:
    """Pull (description, vsl_text) pairs from a history JSONL file where a
    round both has vsl_text stored and hit a perfect execution_score.
    Assumes main_vsl.py has been updated to record vsl_text per round
    (see the main_vsl.py changes that added this field)."""
    examples = []
    with open(history_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for round_ in rec.get("rounds", []):
                if (
                    round_.get("execution_score") == 1.0
                    and round_.get("vsl_text")
                    and round_.get("prompt")
                ):
                    examples.append({
                        "task_id": rec["task_id"],
                        "description": round_["prompt"],
                        "vsl_text": round_["vsl_text"],
                    })
                    break  # one example per task_id is enough
    return examples


def load_eval_problems(problem_file: Path) -> dict[str, dict]:
    """Load the original VerilogEval-style problem records (task_id -> record
    with the fixed module interface under 'prompt' and reference test info),
    needed to actually grade candidate VSL during testing rounds."""
    problems = {}
    with open(problem_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            problems[rec["task_id"]] = rec
    return problems


# --- testing a candidate grammar ---------------------------------------------

async def _try_one(task_id, description, module_interface, grammar_text) -> dict:
    """Run a single description through a throwaway gir-style agent that uses
    the CANDIDATE grammar (not the fixed one in vsl_core.py), then parse,
    validate, and render. Does not run iverilog -- this stage only checks
    that valid VSL was produced, which is a fast, cheap proxy signal for the
    discovery loop. (Full execution scoring is left to main_vsl.py once a
    grammar is promoted.)"""
    candidate_agent = Agent(
        MODEL,
        name="Candidate GIR Agent",
        output_type=DiscoveryVSLOutput,
        model_settings={"temperature": 0},
        system_prompt=grammar_text,
    )
    try:
        result = await candidate_agent.run(description)
        vsl_text = result.output.vsl_code
    except Exception as e:
        return {"task_id": task_id, "ok": False, "stage": "agent_call", "error": str(e), "vsl_text": None}

    try:
        ir = parse_vsl(vsl_text, module_interface=module_interface)
    except VSLParseError as e:
        return {"task_id": task_id, "ok": False, "stage": "parse", "error": str(e), "vsl_text": vsl_text}

    problems = validate_circuit(ir)
    if problems:
        return {"task_id": task_id, "ok": False, "stage": "validate", "error": "; ".join(problems), "vsl_text": vsl_text}

    try:
        render_verilog(ir)
    except ValidationError as e:
        return {"task_id": task_id, "ok": False, "stage": "render", "error": str(e), "vsl_text": vsl_text}

    return {"task_id": task_id, "ok": True, "stage": "done", "error": None, "vsl_text": vsl_text}


async def test_grammar(grammar_text: str, sample: list[dict], eval_problems: dict[str, dict], concurrency: int = 6) -> dict:
    """Run a candidate grammar against a sample of descriptions and summarize
    pass/fail counts plus concrete failure examples for the next round's
    feedback."""
    sem = asyncio.Semaphore(concurrency)

    async def bound_try(ex):
        async with sem:
            problem = eval_problems.get(ex["task_id"], {})
            module_interface = problem.get("prompt", "")
            return await _try_one(ex["task_id"], ex["description"], module_interface, grammar_text)

    results = await asyncio.gather(*(bound_try(ex) for ex in sample))
    passed = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    return {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": len(passed) / len(results) if results else 0.0,
        "failures": failed,
    }


def summarize_test_results(test_results: dict, sample: list[dict], max_failures_shown: int = 15) -> str:
    """Build a human/model-readable summary of a test round: overall
    pass rate plus concrete failure cases (description + error), so the
    discovery agent can see both the full picture (per user's choice) and
    specific gaps to fix."""
    by_id = {ex["task_id"]: ex for ex in sample}
    lines = [
        f"Pass rate this round: {test_results['passed']}/{test_results['total']} "
        f"({test_results['pass_rate']:.1%})",
        "",
        "Failures (up to the first "
        f"{max_failures_shown} of {test_results['failed']}):",
    ]
    for f in test_results["failures"][:max_failures_shown]:
        ex = by_id.get(f["task_id"], {})
        desc = (ex.get("description") or "")[:300]
        lines.append(
            f"- task {f['task_id']} failed at stage '{f['stage']}': {f['error']}\n"
            f"  description: {desc}"
        )
    return "\n".join(lines)


# --- main discovery loop ------------------------------------------------------

async def run_discovery(
    history_path: Path,
    rounds: int,
    sample_size: int,
    problem_file: Path,
    output_path: Path,
    test_history_path: Path | None = None,
    test_problem_file: Path | None = None,
    seed: int = 0,
):
    examples = load_successful_examples(history_path)
    print(f"Loaded {len(examples)} successful (description, VSL) examples from {history_path}")
    if not examples:
        raise SystemExit(
            "No examples with vsl_text found. Make sure main_vsl.py has been "
            "run with the updated code that stores vsl_text per round."
        )

    # Test set: by default, test on the same source as the reference examples
    # (risk of the grammar overfitting to what it already saw). If a separate
    # test_history_path is given (e.g. a run against VerilogDescription_Human,
    # a different, harder style of description than the Machine-derived
    # reference set), test on that instead -- a much better check of whether
    # the grammar actually generalizes rather than just memorizing patterns.
    if test_history_path is not None:
        test_examples = load_successful_examples(test_history_path)
        print(f"Loaded {len(test_examples)} examples from {test_history_path} for testing "
              f"(separate from the {len(examples)} reference examples)")
        if not test_examples:
            raise SystemExit(
                f"No examples with vsl_text found in {test_history_path}. Make sure "
                "main_vsl.py has been run against that dataset with the updated code."
            )
    else:
        test_examples = examples


    eval_problems = load_eval_problems(problem_file)
    if test_problem_file is not None:
        eval_problems = {**eval_problems, **load_eval_problems(test_problem_file)}
    elif test_history_path is not None:
        missing = {ex["task_id"] for ex in test_examples} - eval_problems.keys()
        if missing:
            print(f"WARNING: {len(missing)} test task_id(s) not found in {problem_file} "
                  f"and no --test-problems given -- their module_interface will be empty "
                  f"(e.g. {sorted(missing)[:5]}). Pass --test-problems if the test set "
                  f"uses a different problem file.")

    rng = random.Random(seed)

    reference_set = rng.sample(examples, min(40, len(examples)))

    grammar_text = None
    reasoning = "Initial proposal, no prior feedback yet."
    test_summary_text = "No test has been run yet -- this is the first round."
    best_round = None  

    for round_num in range(1, rounds + 1):
        print(f"\n=== Discovery round {round_num}/{rounds} ===")

        reference_block = "\n\n".join(
            f"Description:\n{ex['description']}\n\nVSL:\n{ex['vsl_text']}"
            for ex in reference_set
        )

        prompt_parts = [
            "Reference examples (description -> VSL) that scored a perfect "
            "execution score with a previous grammar:",
            reference_block,
            "Feedback from testing your most recent grammar proposal on a "
            "fresh sample of descriptions:",
            test_summary_text,
        ]
        if grammar_text:
            prompt_parts += ["Your previous grammar proposal:", grammar_text]

        proposal_result = await discovery_agent.run(prompt_parts)
        grammar_text = proposal_result.output.grammar_text
        reasoning = proposal_result.output.reasoning
        print(f"Discovery agent reasoning: {reasoning[:300]}")

        
        test_sample = rng.sample(test_examples, min(sample_size, len(test_examples)))
        test_results = await test_grammar(grammar_text, test_sample, eval_problems)
        test_summary_text = summarize_test_results(test_results, test_sample)
        print(f"Round {round_num} pass rate: {test_results['pass_rate']:.1%} "
              f"({test_results['passed']}/{test_results['total']})")


        round_record = {
            "round": round_num,
            "pass_rate": test_results["pass_rate"],
            "passed": test_results["passed"],
            "total": test_results["total"],
            "reasoning": reasoning,
            "grammar_text": grammar_text,
        }
        with open(output_path.with_suffix(".rounds.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(round_record) + "\n")


        if best_round is None or round_record["pass_rate"] > best_round["pass_rate"]:
            best_round = round_record

    output_path.write_text(best_round["grammar_text"], encoding="utf-8")
    print(f"\nBest round was {best_round['round']}/{rounds} "
          f"(pass_rate={best_round['pass_rate']:.1%}). Final grammar written to {output_path}")
    print("Review it, then paste it in as VSL_GRAMMAR_AND_EXAMPLES in vsl_core.py to use it.")


@app.command()
def main(
    history: Path = typer.Option(..., help="Path to a history JSONL file (Machine-derived) that includes vsl_text per round -- used as reference examples."),
    problems: Path = typer.Option(..., help="Path to the VerilogEval-style problem JSONL matching --history (task_id -> module interface / reference test)."),
    test_history: Path = typer.Option(None, help="Optional separate history JSONL (e.g. from a run against VerilogDescription_Human) used to TEST each candidate grammar. If omitted, tests on the same file as --history."),
    test_problems: Path = typer.Option(None, help="Problem JSONL matching --test-history (needed if the test set's task_ids come from a different problem file, e.g. VerilogEval_Human.jsonl vs VerilogEval_Machine.jsonl). If omitted, --problems is reused."),
    rounds: int = typer.Option(8, help="Number of discover -> test -> refine rounds."),
    sample_size: int = typer.Option(150, help="Number of descriptions to test each candidate grammar against, per round."),
    output: Path = typer.Option(Path("discovered_grammar.txt"), help="Where to write the final grammar text."),
    seed: int = typer.Option(0, help="Random seed for sampling, for reproducibility."),
):
    asyncio.run(run_discovery(
        history, rounds, sample_size, problems, output,
        test_history_path=test_history, test_problem_file=test_problems, seed=seed,
    ))


if __name__ == "__main__":
    app()