"""Check the runtime can reproduce K-Bound (numpy/sklearn required; torch only for CIFAR/TTA)."""
import importlib, sys
req = ["numpy", "scipy", "sklearn", "matplotlib"]
opt = ["torch", "torchvision"]
ok = True
for m in req:
    try: importlib.import_module(m); print(f"[ok]  {m}")
    except Exception as e: ok = False; print(f"[MISSING] {m}: {e}")
for m in opt:
    try: importlib.import_module(m); print(f"[ok]  {m} (for CIFAR/TTA)")
    except Exception: print(f"[warn] {m} not installed (needed only for CIFAR/ImageNet TTA)")
print("python", sys.version.split()[0]); sys.exit(0 if ok else 1)
