"""Scan for concatenated statements on same line in master_pipeline.py."""
import re

with open(r'C:\Users\Administrator\ibkr8\scripts\master_pipeline.py', 'r') as f:
    lines = f.readlines()

keywords = ['if ', 'for ', 'while ', 'try:', 'except ', 'return ', 'def ', 'class ',
            'import ', 'from ', 'raise ', 'with ', 'elif ', 'else:', 'pass', 'break', 'continue']

found = 0
for i, line in enumerate(lines, 1):
    s = line.rstrip()
    if not s or s.lstrip().startswith('#'):
        continue
    # Look for code followed by 4+ spaces then a keyword
    for kw in keywords:
        pattern = r'[)}\]\w\d\'\"]\s{4,}' + re.escape(kw)
        if re.search(pattern, s):
            print(f'L{i}: {s[:160]}')
            found += 1
            break

print(f'\nTotal suspicious: {found}')
