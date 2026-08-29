import os
from pathlib import Path
from wilds import get_dataset

def main():
    repo_root = Path(__file__).resolve().parents[3]
    default_data_root = repo_root / "data"
    data_dir = Path(os.environ.get("KBOUND_DATA_ROOT", default_data_root))

    print(f"Downloading FMoW to {data_dir}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # download=True will download and extract the dataset
    dataset = get_dataset(dataset="fmow", root_dir=data_dir, download=True)
    print(f"Successfully downloaded/verified FMoW to {data_dir}")

if __name__ == "__main__":
    main()
