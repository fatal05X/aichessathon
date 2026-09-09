SHELL := /bin/bash

# Directories whose contents should be copied into the project root.
PRE_DIRS := tablebases/3-4/wdl tablebases/3-4/dtz-nr

.PHONY: setup play arena zip gate pre clear

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/numba

zip:
	@uv run python -c "import os, subprocess; p='pre/pre.txt'; files=['book.bin', 'weights.npy']; files += open(p).read().splitlines() if os.path.isfile(p) else []; subprocess.run(['uv', 'run', 'python', '-m', 'harness.package'] + [x for f in files for x in ('--include', f)], check=True)"


gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000

pre:
	@uv run python -c "import os, shutil; dirs='$(PRE_DIRS)'.split(); os.makedirs('pre', exist_ok=True); open('pre/pre.txt', 'a').close(); [(print(f'Skipping existing: {d}'), None) if os.path.exists((dest:=os.path.relpath(os.path.join(root,f),d))) else (os.makedirs(os.path.dirname(dest) or '.', exist_ok=True), shutil.copy2(os.path.join(root,f),dest), open('pre/pre.txt','a').write(dest+'\\n'), print(f'Copied: {os.path.join(root,f)} -> {dest}')) for d in dirs for root,_,files in os.walk(d) for f in files]"

clear:
	@uv run python -c "import os; p='pre/pre.txt'; [ (os.remove(f), print(f'Removed: {f}')) for f in open(p).read().splitlines() if os.path.isfile(f) ] if os.path.isfile(p) else None; os.path.exists(p) and os.remove(p)"