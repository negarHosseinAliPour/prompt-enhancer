import json
import sys

def main():
    history_file = sys.argv[1] if len(sys.argv) > 1 else 'outputs/verilogeval_v2/enhanced_vsl_forcevsl_history_verilogeval_v2_forcevsl_v8_FINAL.jsonl'

    history = {}
    with open(history_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            history[d['task_id']] = d

    improved = []
    same = []
    for task_id, d in history.items():
        rounds = d.get('rounds', [])
        first_round = next((r for r in rounds if r.get('round') == -1), rounds[0] if rounds else None)
        p1 = first_round.get('execution_score', 0) if first_round else 0
        final = d.get('final_execution_score', 0)
        if final > p1:
            improved.append((task_id, p1, final))
        else:
            same.append((task_id, p1, final))

    print('Total tasks: {}'.format(len(history)))
    print('Improved by revision loop: {}'.format(len(improved)))
    print('Not improved (same or worse): {}'.format(len(same)))
    print()
    print('--- Improved tasks (task_id: pass@1 -> resolved) ---')
    for t, p1, f in sorted(improved, key=lambda x: x[2] - x[1], reverse=True):
        print('  {}: {:.4f} -> {:.4f}  (delta +{:.4f})'.format(t, p1, f, f - p1))

    zero_to_full = sum(1 for _, p1, f in improved if p1 == 0 and f >= 1.0)
    zero_to_partial = sum(1 for _, p1, f in improved if p1 == 0 and 0 < f < 1.0)
    partial_to_full = sum(1 for _, p1, f in improved if 0 < p1 < 1.0 and f >= 1.0)
    partial_to_partial = sum(1 for _, p1, f in improved if 0 < p1 < 1.0 and 0 < f < 1.0 and f > p1)

    print()
    print('--- Breakdown of improvements ---')
    print('0 -> full pass (1.0): {}'.format(zero_to_full))
    print('0 -> partial: {}'.format(zero_to_partial))
    print('partial -> full pass: {}'.format(partial_to_full))
    print('partial -> higher partial: {}'.format(partial_to_partial))

    print()
    print('--- Not improved tasks (still failing/partial after full pipeline) ---')
    for t, p1, f in sorted(same, key=lambda x: x[2]):
        if f < 1.0:
            print('  {}: pass@1={:.4f}, final={:.4f}'.format(t, p1, f))

if __name__ == '__main__':
    main()
