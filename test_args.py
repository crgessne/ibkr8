import argparse

p = argparse.ArgumentParser()
p.add_argument("--concurrent", action="store_true")
args = p.parse_args()

print(f"Concurrent flag: {args.concurrent}")
print(f"Type: {type(args.concurrent)}")
