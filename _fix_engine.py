"""Find merged lines in trading/engine.py, print them, then fix them."""
import re

path = "trading/engine.py"
content = open(path, encoding="utf-8").read()
lines = content.splitlines(keepends=True)

out = open("_fix_engine_out.txt", "w", encoding="utf-8")

# Check syntax first
try:
    compile(content, path, "exec")
    out.write("Syntax OK!\n")
except SyntaxError as e:
    out.write(f"SyntaxError at line {e.lineno}: {e.msg}\n")
    out.write(f"Text: {repr(e.text)}\n")
    start = max(0, e.lineno - 4)
    end = min(len(lines), e.lineno + 3)
    for j in range(start, end):
        marker = ">>>" if j + 1 == e.lineno else "   "
        out.write(f"{marker} L{j+1}: {repr(lines[j])}\n")

# Find all merged lines
merged = []
for i, line in enumerate(lines):
    s = line.rstrip("\n")
    if re.search(r'\)\s{4,}(print|def |self\.|raise )', s):
        merged.append(i)
    if re.search(r'"""\s{4,}def ', s):
        merged.append(i)

out.write(f"\nFound {len(merged)} merged lines:\n")
for i in merged:
    out.write(f"  L{i+1}: {repr(lines[i][:160])}\n")

out.close()
