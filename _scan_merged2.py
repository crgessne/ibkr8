"""Scan master_pipeline.py for merged lines (two statements on one line)."""
import re

with open(r'C:\Users\Administrator\ibkr8\scripts\master_pipeline.py', encoding='utf-8') as f:
    lines = f.readlines()

merged = []
for i, line in enumerate(lines, 1):
    stripped = line.rstrip('\n')
    # Pattern: something that closes a statement (closing paren, quote, word)
    # followed by 2+ spaces, then a new statement starting with keyword/identifier
    if re.search(r'[)\'\"0-9a-zA-Z_]\s{2,}(?:print|all_results|for |if |df |result|parser|args|trained)', stripped):
        merged.append((i, stripped[:140]))

for ln, content in merged:
    print(f'Line {ln:5d}: {content}')
print(f'\nTotal suspect merged lines: {len(merged)}')
