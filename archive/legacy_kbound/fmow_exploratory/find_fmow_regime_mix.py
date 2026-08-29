"""
find_fmow_regime_mix.py

Evaluates the source model f0 and adapted models (Tent/EATA) across different
temporal/geographic splits in FMoW to identify a mixed regime (where adaptation
is helpful in some cells and harmful in others).
"""
import os, sys, argparse, json
from pathlib import Path
import torch
import torchvision.models as tvm
from torch.utils.data import DataLoader
from torchvision import transforms
from wilds import get_dataset

NUM_CLASSES = 62

def build_model(device):
    m = tvm.densenet121(weights=None)
    m.classifier = torch.nn.Linear(m.classifier.in_features, NUM_CLASSES)
    return m.to(device)

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total if total > 0 else 0.0

def main(args):
    device = torch.device(args.device)
    repo_root = Path(__file__).resolve().parents[3]
    default_data_root = repo_root / "data"
    data_dir = Path(os.environ.get("KBOUND_DATA_ROOT", default_data_root))

    dataset = get_dataset(dataset="fmow", root_dir=data_dir, download=False)

    # Load f0
    model = build_model(device)
    if args.f0_path:
        model.load_state_dict(torch.load(args.f0_path, map_location=device))
        print(f"Loaded f0 from {args.f0_path}")
    else:
        print("WARNING: No f0 path provided, using untrained weights for diagnostic.")

    val_data = dataset.get_subset("val", transform=get_transforms())
    test_data = dataset.get_subset("test", transform=get_transforms())

    # We want to break it down by year or region.
    # FMoW metadata fields usually include region (idx 0) and year (idx 1).
    # Let's inspect the metadata
    metadata_fields = dataset.metadata_fields
    print(f"Metadata fields: {metadata_fields}")

    def evaluate_split(split_name, split_data):
        print(f"\n--- Evaluating {split_name} ---")
        loader = DataLoader(split_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        acc = evaluate(model, loader, device)
        print(f"Overall {split_name} Accuracy: {acc:.4f}")

        # Breakdown by region (idx 0)
        region_idx = metadata_fields.index('region')
        regions = torch.unique(split_data.dataset.metadata_array[split_data.indices, region_idx])
        for r in regions:
            r_val = r.item()
            # FMoW region names are mapped in dataset.metadata_map
            r_name = dataset.metadata_map['region'][r_val]
            idx_r = (split_data.dataset.metadata_array[split_data.indices, region_idx] == r).nonzero(as_tuple=True)[0]
            if len(idx_r) == 0:
                continue
            subset = torch.utils.data.Subset(split_data, idx_r)
            subset_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            r_acc = evaluate(model, subset_loader, device)
            print(f"  Region {r_name} (n={len(idx_r)}): Acc = {r_acc:.4f}")

    evaluate_split("val", val_data)
    evaluate_split("test", test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
