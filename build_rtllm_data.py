import json
import os
import re

base = 'RTLLM'
skip_dirs = {'_chatgpt35', '_chatgpt4', '_pic', '__pycache__', '.git'}


def find_matching_paren(text, open_idx):
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return -1


def find_instantiation(text, module_name):
    for m in re.finditer(r'\b' + re.escape(module_name) + r'\b', text):
        pos = m.end()
        while pos < len(text) and text[pos] in ' \t\r\n':
            pos += 1
        param_conns = []
        if pos < len(text) and text[pos] == '#':
            popen = text.find('(', pos)
            if popen == -1:
                continue
            pclose = find_matching_paren(text, popen)
            if pclose == -1:
                continue
            param_text = text[popen + 1:pclose]
            param_conns = re.findall(r'\.(\w+)\s*\(\s*([^,()]+?)\s*\)', param_text)
            pos = pclose + 1
            while pos < len(text) and text[pos] in ' \t\r\n':
                pos += 1
        im = re.match(r'(\w+)\s*\(', text[pos:])
        if not im:
            continue
        popen2 = pos + im.end() - 1
        pclose2 = find_matching_paren(text, popen2)
        if pclose2 == -1:
            continue
        after = text[pclose2 + 1:pclose2 + 5]
        if ';' not in after:
            continue
        port_text = text[popen2 + 1:pclose2]
        port_conns = re.findall(r'\.(\w+)\s*\(\s*([^,()]*?)\s*\)', port_text)
        if port_conns:
            return param_conns, port_conns
        # also handle positional (non-named) connections as a fallback signal
        positional = [s.strip() for s in port_text.split(',') if s.strip()]
        if positional:
            return param_conns, [('__pos{}__'.format(i), s) for i, s in enumerate(positional)]
    return None


def build_decl_index(text):
    """Map signal_name -> (kind, width) for every reg/wire declaration in text."""
    index = {}
    # strip line comments to avoid false matches
    clean = re.sub(r'//.*', '', text)
    for m in re.finditer(r'\b(reg|wire)\s*(\[[^\]]*\])?\s*([^;]+);', clean):
        kind, width, rest = m.group(1), m.group(2) or '', m.group(3)
        # skip if this looks like a port declaration list (contains 'input'/'output') - rare, but guard
        for item in rest.split(','):
            item = item.strip()
            if not item:
                continue
            # drop initializer
            item = item.split('=')[0].strip()
            # drop array subscript like foo[0:15]
            name_match = re.match(r'(\w+)', item)
            if name_match:
                name = name_match.group(1)
                if name not in index:
                    index[name] = (kind, width)
    return index


def get_param_default(text, pname):
    m = re.search(r'\bparameter\s+' + re.escape(pname) + r'\s*=\s*([^;,]+)', text)
    if m:
        return m.group(1).strip()
    return None


desc_entries = []
eval_entries = []
issues = []

for root, dirs, files in os.walk(base):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    if 'design_description.txt' in files and 'testbench.v' in files:
        task_id = os.path.basename(root)
        with open(os.path.join(root, 'design_description.txt'), encoding='utf-8', errors='replace') as f:
            description = f.read().strip()
        with open(os.path.join(root, 'testbench.v'), encoding='utf-8', errors='replace') as f:
            testbench = f.read()
        canonical = ''
        for fn in files:
            if fn.startswith('verified_') and fn.endswith('.v'):
                with open(os.path.join(root, fn), encoding='utf-8', errors='replace') as f:
                    canonical = f.read()
                break

        inst = find_instantiation(testbench, task_id)
        header = ''
        if inst is None:
            issues.append((task_id, 'no instantiation found'))
        else:
            param_conns, port_conns = inst
            decl_index = build_decl_index(testbench)
            param_lines = []
            for pname, pval in param_conns:
                default = get_param_default(testbench, pval) or get_param_default(testbench, pname) or '1'
                param_lines.append('parameter {} = {}'.format(pname, default))
            port_lines = []
            unresolved = []
            positional_flag = any(p.startswith('__pos') for p, _ in port_conns)
            for pname, sig in port_conns:
                decl = decl_index.get(sig)
                if decl is None:
                    unresolved.append((pname, sig))
                    label = sig if not pname.startswith('__pos') else 'port{}'.format(pname)
                    port_lines.append('input {} /* WARNING: could not resolve signal {} */'.format(label, sig))
                    continue
                kind, width = decl
                direction = 'input' if kind == 'reg' else 'output'
                width_str = (width + ' ') if width else ''
                label = pname if not pname.startswith('__pos') else sig
                port_lines.append('{} {}{}'.format(direction, width_str, label))
            if unresolved:
                issues.append((task_id, 'unresolved: {}'.format(unresolved)))
            if positional_flag:
                issues.append((task_id, 'POSITIONAL connections (names guessed from signal names) - please review'))
            if param_lines:
                header = 'module {} #(\n    {}\n)(\n    {}\n);\n'.format(
                    task_id, ',\n    '.join(param_lines), ',\n    '.join(port_lines))
            else:
                header = 'module {} (\n    {}\n);\n'.format(
                    task_id, ',\n    '.join(port_lines))

        desc_entries.append({
            'task_id': task_id,
            'simple_description': description,
            'detail_description': description,
        })
        eval_entries.append({
            'task_id': task_id,
            'prompt': header,
            'canonical_solution': canonical,
            'test': testbench,
        })

print('Found {} RTLLM tasks'.format(len(desc_entries)))
print('Issues ({}):'.format(len(issues)))
for t, i in issues:
    print('  {}: {}'.format(t, i))

os.makedirs('outputs/rtllm', exist_ok=True)
with open('outputs/rtllm/rtllm_desc.jsonl', 'w') as f:
    for d in desc_entries:
        f.write(json.dumps(d) + '\n')
with open('outputs/rtllm/rtllm_eval.jsonl', 'w') as f:
    for d in eval_entries:
        f.write(json.dumps(d) + '\n')

print('Saved to outputs/rtllm/rtllm_desc.jsonl and rtllm_eval.jsonl')
