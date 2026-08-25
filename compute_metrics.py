"""
Computes pass@1 and resolve-rate metrics for any dataset's force-VSL run,
using the pipeline's own internal execution scores (from vsl_core.py's iverilog
grading), following the same pass@1 / resolve-rate definitions used in the
VSL_Description.docx methodology doc:

  - pass@1: score of the very first attempt (round -1), before any rewording
    or revision-loop rounds.
  - resolve rate: best score reached after the full pipeline (raw attempt,
    reworded retry, up to 3 revision-loop rounds) -- an oracle-guided best-of-4,
    since the revision loop sees the real iverilog error each round.

Usage:
    python3 compute_metrics.py <history_file.jsonl> <dataset_label> [output_file.json]

Example:
    python3 compute_metrics.py outputs/machine_143_FINAL_history.jsonl Machine outputs/machine_143_metrics.json
    python3 compute_metrics.py outputs/human_156_FINAL_history.jsonl Human outputs/human_156_metrics.json
"""
import json
import sys


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 compute_metrics.py <history_file.jsonl> <dataset_label> [output_file.json]')
        sys.exit(1)

    history_file = sys.argv[1]
    dataset_label = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    history = {}
    with open(history_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            history[d['task_id']] = d

    total = len(history)

    pass1_scores = []
    resolve_scores = []
    for task_id, d in history.items():
        rounds = d.get('rounds', [])
        first_round = next((r for r in rounds if r.get('round') == -1), rounds[0] if rounds else None)
        pass1_score = first_round.get('execution_score', 0) if first_round else 0
        pass1_scores.append(pass1_score)
        resolve_scores.append(d.get('final_execution_score', 0))

    pass1_count = sum(1 for s in pass1_scores if s >= 1.0)
    resolve_count = sum(1 for s in resolve_scores if s >= 1.0)

    result = {
        'dataset': dataset_label,
        'total_tasks': total,
        'pass_at_1': {
            'passed': pass1_count,
            'pass@1': round(pass1_count / total * 100, 2),
            'avg_score_pct': round(sum(pass1_scores) / total * 100, 2),
        },
        'resolve_rate': {
            'passed': resolve_count,
            'resolve_rate': round(resolve_count / total * 100, 2),
            'avg_score_pct': round(sum(resolve_scores) / total * 100, 2),
        },
    }

    print(json.dumps(result, indent=2))

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print('Saved to {}'.format(output_file))


if __name__ == '__main__':
    main()
