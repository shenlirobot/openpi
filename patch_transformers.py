import shutil, sys
from pathlib import Path
import transformers

site_dir = Path(transformers.__file__).parent
src = Path('/data/scratch/shenli/openpi/src/openpi/models_pytorch/transformers_replace')

print(f"[patch] transformers dir: {site_dir}", flush=True)
print(f"[patch] source dir:       {src}", flush=True)

if not src.exists():
    print(f"[patch] ERROR: missing source {src}", file=sys.stderr, flush=True)
    sys.exit(1)

count = 0
for p in src.rglob('*'):
    dst = site_dir / p.relative_to(src)
    if p.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        print(f"[patch] copied {p} -> {dst}", flush=True)
        count += 1

print(f"[patch] done, copied {count} files.", flush=True)
