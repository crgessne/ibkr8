"""Scan for merged lines in master_pipeline.py."""
import re

with open(r'C:\Users\Administrator\ibkr8\scripts\master_pipeline.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    stripped = line.rstrip()
    if not stripped or stripped.lstrip().startswith('#'):
        continue
    content = stripped.lstrip()
    # Look for 4+ spaces in the middle of a line (between two code tokens)
    m = re.search(r'[^\s]\s{4,}[^\s]', content)
    if not m:
        continue
    # Skip common false positives
    skip_patterns = [
        'np.where', 'bins=', 'help=', 'labels=', 'lambda',
        'def ', 'class ', "' '", '" "', '"""', "'''",
        'choices=', 'default=', 'action=',
    ]
    if any(p in content for p in skip_patterns):
        continue
    # Skip lines that are just long argument lists or string literals
    if content.count('(') > 0 and content.count(')') > 0:
        continue
    print(f'L{i}: {stripped.strip()[:140]}')
