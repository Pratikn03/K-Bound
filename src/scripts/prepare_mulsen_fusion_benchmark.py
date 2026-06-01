"""Prepare naturally paired MulSen-AD RGB + infrared fusion inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from prepare_mvtec3d_fusion_benchmark import (
    IMAGE_SUFFIXES,
    PairedObservation,
    _is_visible_dir,
    build_fusion_frame_from_pairs,
)


@dataclass(frozen=True)
class MulSenPairedObservation(PairedObservation):
    @property
    def sample_id(self) -> str:
        safe = "_".join([self.category, self.split, self.defect_type, self.stem])
        return f"mulsen_{safe}".replace("-", "_")


@dataclass(frozen=True)
class MulSenPair:
    category: str
    split: str
    defect_type: str
    stem: str
    rgb_path: Path
    infrared_path: Path

    @property
    def label(self) -> int:
        return 0 if self.defect_type == "good" else 1

    def to_paired_observation(self) -> MulSenPairedObservation:
        return MulSenPairedObservation(
            category=self.category,
            split=self.split,
            defect_type=self.defect_type,
            stem=self.stem,
            rgb_path=self.rgb_path,
            depth_path=self.infrared_path,
        )


def _image_files(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    return {
        file.stem: file
        for file in sorted(path.iterdir())
        if file.is_file()
        and not file.name.startswith(".")
        and not file.name.startswith("._")
        and file.suffix.lower() in IMAGE_SUFFIXES
    }


def discover_mulsen_rgb_infrared_pairs(
    dataset_root: Path,
    categories: list[str] | None = None,
) -> list[MulSenPair]:
    """Discover RGB/Infrared pairs under MulSen_AD/<category>/{RGB,Infrared}/."""
    root = Path(dataset_root)
    allowed = set(categories) if categories else None
    pairs: list[MulSenPair] = []
    for category_dir in sorted(p for p in root.iterdir() if _is_visible_dir(p)):
        if allowed is not None and category_dir.name not in allowed:
            continue
        rgb_root = category_dir / "RGB"
        ir_root = category_dir / "Infrared"
        if not rgb_root.is_dir() or not ir_root.is_dir():
            continue
        for split in ("train", "test"):
            rgb_split = rgb_root / split
            ir_split = ir_root / split
            if not rgb_split.is_dir() or not ir_split.is_dir():
                continue
            for defect_dir in sorted(p for p in rgb_split.iterdir() if _is_visible_dir(p)):
                ir_defect = ir_split / defect_dir.name
                if not ir_defect.is_dir():
                    continue
                rgb_files = _image_files(defect_dir)
                ir_files = _image_files(ir_defect)
                for stem in sorted(set(rgb_files) & set(ir_files)):
                    pairs.append(
                        MulSenPair(
                            category=category_dir.name,
                            split=split,
                            defect_type=defect_dir.name,
                            stem=stem,
                            rgb_path=rgb_files[stem],
                            infrared_path=ir_files[stem],
                        )
                    )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/mulsen_ad"))
    parser.add_argument("--output", type=Path, default=Path("experiments/fusion/m2_external_mulsen_sealed_inputs.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("experiments/fusion/m2_external_mulsen_sealed_metadata.json"))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--train-categories", nargs="*", required=True)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--feature-mode", default="patchcore", choices=["patchcore", "m3dm", "image_statistics"])
    parser.add_argument("--patchcore-k", type=int, default=5)
    parser.add_argument("--patchcore-coreset-size", type=int, default=2048)
    parser.add_argument("--heldout-val-fraction", type=float, default=0.15)
    parser.add_argument("--heldout-val-seed", type=int, default=20260528)
    args = parser.parse_args()

    mulsen_pairs = discover_mulsen_rgb_infrared_pairs(args.dataset_root, categories=args.categories)
    if not mulsen_pairs:
        raise SystemExit(f"No MulSen RGB/Infrared pairs under {args.dataset_root}")

    pairs = [p.to_paired_observation() for p in mulsen_pairs]
    frame, metadata = build_fusion_frame_from_pairs(
        pairs,
        embedding_dim=args.embedding_dim,
        feature_mode=args.feature_mode,
        train_categories=args.train_categories,
        patchcore_k=args.patchcore_k,
        patchcore_coreset_size=args.patchcore_coreset_size,
        heldout_val_fraction=args.heldout_val_fraction,
        heldout_val_seed=args.heldout_val_seed,
        second_domain_name="infrared",
        benchmark_type="naturally_paired_mulsen_rgb_infrared_score_fusion",
        pairing_unit="MulSen-AD synchronized RGB + lock-in infrared capture",
    )
    metadata["dataset_layout"] = "MulSen_AD/<category>/{RGB,Infrared}/<train|test>/<defect>/"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata["output"] = str(args.output)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
