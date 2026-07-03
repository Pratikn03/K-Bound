#!/usr/bin/env python3
"""fetch_cifar_c.py -- auto-download + verify + extract the Hendrycks CIFAR-C
corruption benchmarks used by the K-Bound decisive deep-TTA experiment.

Pure-stdlib (urllib + tarfile + hashlib): no torch, no extra pip deps.
Resumable download (HTTP Range), md5 verification, idempotent (skips if present).

Datasets (verified against the Zenodo record on 2026-06-05):
  CIFAR-10-C  : record 2535967  CIFAR-10-C.tar   2.9 GB  md5 56bf5dcef84df0e2308c6dcbcbbd8499
  CIFAR-100-C : record 3555552  CIFAR-100-C.tar  2.9 GB  (md5 verified at runtime if --md5 given)

Layout produced (matches cifar_tent_mps_v2.load_cifar_c):
  <data-root>/CIFAR-10-C/gaussian_noise.npy ... labels.npy

Usage:
  python3 fetch_cifar_c.py --which 10                 # download+extract CIFAR-10-C
  python3 fetch_cifar_c.py --which 10 --check         # report status only, no download
  python3 fetch_cifar_c.py --which 100 --keep-tar     # also keep the .tar after extract
"""
from __future__ import annotations
import argparse, hashlib, os, sys, tarfile, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/ -> kbound/ -> research/ -> docs/ -> AutoML_Flagship_V8/
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "experiments", "kbound", "cifar")

SOURCES = {
    "10": {
        "dir": "CIFAR-10-C",
        "url": "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1",
        "md5": "56bf5dcef84df0e2308c6dcbcbbd8499",
        "size_gb": 2.9,
    },
    "100": {
        "dir": "CIFAR-100-C",
        "url": "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1",
        "md5": None,  # pass --md5 <hash> to enforce; left None so we don't assert a wrong one
        "size_gb": 2.9,
    },
}
# A representative set the runner needs (full set is 15 + 4 extra; labels.npy is shared).
KEY_FILES = ["labels.npy", "gaussian_noise.npy"]


def have_dataset(target_dir: str) -> bool:
    return all(os.path.exists(os.path.join(target_dir, f)) for f in KEY_FILES)


def md5sum(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    """Resumable GET with a simple progress meter."""
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"  resuming at {existing/1e9:.2f} GB")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable -> already complete
            print("  server says already complete"); return
        raise
    total = existing + int(resp.headers.get("Content-Length", 0))
    mode = "ab" if existing else "wb"
    done = existing
    with open(dest, mode) as f:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block); done += len(block)
            if total:
                pct = 100 * done / total
                sys.stdout.write(f"\r  {done/1e9:5.2f}/{total/1e9:5.2f} GB ({pct:5.1f}%)")
                sys.stdout.flush()
    sys.stdout.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["10", "100"], default="10")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--check", action="store_true", help="report status only; no download")
    ap.add_argument("--keep-tar", action="store_true", help="keep the .tar after extraction")
    ap.add_argument("--md5", default=None, help="override/enforce expected md5")
    ap.add_argument("--no-verify", action="store_true", help="skip md5 verification")
    args = ap.parse_args()

    src = SOURCES[args.which]
    target_dir = os.path.join(args.data_root, src["dir"])
    tar_path = os.path.join(args.data_root, src["dir"] + ".tar")
    os.makedirs(args.data_root, exist_ok=True)

    print(f"CIFAR-{args.which}-C")
    print(f"  data-root : {args.data_root}")
    print(f"  target    : {target_dir}")
    print(f"  source    : {src['url']}  (~{src['size_gb']} GB)")

    if have_dataset(target_dir):
        print("  status    : PRESENT (key files found) -> nothing to do.")
        return 0
    print("  status    : MISSING")
    if args.check:
        print("  (--check) would download + extract. Re-run without --check to fetch.")
        return 0

    # Download (resumable).
    print("Downloading ...")
    download(src["url"], tar_path)

    # Verify.
    expected = args.md5 or src["md5"]
    if expected and not args.no_verify:
        print("Verifying md5 ...")
        got = md5sum(tar_path)
        if got.lower() != expected.lower():
            print(f"  md5 MISMATCH: got {got}, expected {expected}", file=sys.stderr)
            print("  delete the .tar and retry (download may be truncated).", file=sys.stderr)
            return 2
        print(f"  md5 OK ({got})")
    else:
        print("  md5 check skipped (no expected hash).")

    # Extract.
    print("Extracting ...")
    with tarfile.open(tar_path) as t:
        t.extractall(args.data_root)  # tar contains the CIFAR-XX-C/ directory
    if not have_dataset(target_dir):
        print("  extraction did not produce the expected layout.", file=sys.stderr)
        return 3
    if not args.keep_tar:
        os.remove(tar_path); print("  removed .tar")
    print(f"DONE -> {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
