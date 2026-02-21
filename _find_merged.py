import re, sys

out = []
for fname in ['trading/engine.py', 'trading/strategy.py', 'trading/runner.py', 'trading/config.py']:
    lines = open(fname, encoding='utf-8').read().splitlines()
    bad = []
    for i, l in enumerate(lines):
        # Detect: non-whitespace content, then 4+ spaces, then more content
        # This catches lines like `flush=True)    print(` or `"""    def `
        stripped = l.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if re.search(r'\)\s{4,}(print|def |self\.|raise |if |for |return )', l):
            bad.append((i+1, l))
        if re.search(r'"""\s{4,}def ', l):
            bad.append((i+1, l))
    if bad:
        out.append(f'=== {fname} ===')
        for n, l in bad:
            out.append(f'  L{n}: {repr(l[:150])}')

# Syntax check
for fname in ['trading/engine.py', 'trading/strategy.py', 'trading/runner.py']:
    try:
        compile(open(fname, encoding='utf-8').read(), fname, 'exec')
        out.append(f'{fname}: syntax OK')
    except SyntaxError as e:
        out.append(f'{fname}: SyntaxError L{e.lineno}: {e.msg}')
        out.append(f'  text: {repr(e.text)}')

result = '\n'.join(out)
open('_merged_lines.txt', 'w', encoding='utf-8').write(result)
print(result)
