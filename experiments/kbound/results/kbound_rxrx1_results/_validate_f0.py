import time, numpy as np, torch
import torchvision.models as M
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from wilds import get_dataset

CK = "/Users/pratik_n/kbound_rxrx1_ckpt/rxrx1_seed:0_epoch:best_model.pth"
ROOT = "/Volumes/T9/uav/AutoML_Flagship_V8/experiments/kbound/data/wilds"

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
