# --- defect D8: portable roots (docs/research/kbound/EXTERNAL_STORAGE_POLICY.md bans
# --- machine-local absolute paths in tracked code). KB_REPO_ROOT is discovered from this
# --- file's own location; override with $KBOUND_REPO_ROOT.
import os as _kb_os
from pathlib import Path as _KbPath


def _kb_repo_root() -> str:
    override = _kb_os.environ.get("KBOUND_REPO_ROOT", "").strip()
    if override:
        return str(_KbPath(override).expanduser().resolve())
    here = _KbPath(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return str(candidate)
    raise RuntimeError(f"repository root not found above {here}; set KBOUND_REPO_ROOT")


KB_REPO_ROOT = _kb_repo_root()

# --- external (git-excluded) data volume: ONE documented variable, no default.
def _kb_external_root() -> str:
    value = _kb_os.environ.get("KBOUND_EXTERNAL_ROOT", "").strip()
    if not value:
        raise RuntimeError(
            "KBOUND_EXTERNAL_ROOT is not set. This script needs data that is deliberately "
            "not in the git release (raw datasets, checkpoints, caches). Point "
            "KBOUND_EXTERNAL_ROOT at the volume holding them; the expected layout is "
            "documented in docs/research/kbound/kbound_repro/paths.py (EXTERNAL_LAYOUT) "
            "and acquisition is in DATA.md. There is no default on purpose: this used to "
            "be one author's external SSD, and defaulting to $HOME would write gigabytes "
            "somewhere you did not choose."
        )
    return str(_KbPath(value).expanduser().resolve())


KB_EXTERNAL_ROOT = _kb_external_root()

import time, numpy as np, torch
import torchvision.models as M
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from wilds import get_dataset

CK = KB_EXTERNAL_ROOT + "/kbound_rxrx1_ckpt/rxrx1_seed:0_epoch:best_model.pth"
ROOT = KB_REPO_ROOT + "/experiments/kbound/data/wilds"

def standardize(x):
    mean = x.mean(dim=(1, 2)); std = x.std(dim=(1, 2)); std[std == 0.] = 1.
    return TF.normalize(x, mean, std)

tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(standardize)])

sd = torch.load(CK, map_location="cpu", weights_only=False)["algorithm"]
new = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
m = M.resnet50(weights=None, num_classes=1139)
res = m.load_state_dict(new, strict=False)
print("LOAD missing=%d unexpected=%d" % (len(res.missing_keys), len(res.unexpected_keys)), flush=True)
dev = torch.device("mps"); m.to(dev).eval()

ds = get_dataset(dataset="rxrx1", download=False, root_dir=ROOT)
for split, n in [("id_test", 256), ("test", 256)]:
    sub = ds.get_subset(split, transform=tf)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(sub), n, replace=False)
    t = time.time()
    items = [sub[int(i)] for i in idx]
    xs = torch.stack([it[0] for it in items]).to(dev)
    ys = np.array([int(it[1]) for it in items])
    with torch.no_grad():
        pr = m(xs).argmax(1).cpu().numpy()
    print("%s n=%d acc=%.4f load_infer_sec=%.1f" % (split, n, float((pr == ys).mean()), time.time() - t), flush=True)
print("VALIDATION_DONE", flush=True)
