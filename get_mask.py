import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LABEL_MAP = {
    "impurity|notSure": 244,
    "graphite-edgeCrinkle": 180,
}


@dataclass(frozen=True)
class AppConfig:
    root: Path
    data_dir_name: str
    mask_dir_name: str
    label_map: dict[str, int]
    strict: bool


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations into grayscale mask images."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to search for JSON files. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--data-dir-name",
        default="label",
        help="Directory name segment replaced when resolving output paths.",
    )
    parser.add_argument(
        "--mask-dir-name",
        default="mask",
        help="Directory name segment used for generated mask directories.",
    )
    parser.add_argument(
        "--label-map-file",
        type=Path,
        help="Optional JSON file containing a label-to-pixel-value mapping.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately when a file cannot be processed.",
    )
    return parser.parse_args()


def load_label_map(label_map_file: Path | None) -> dict[str, int]:
    if label_map_file is None:
        return DEFAULT_LABEL_MAP.copy()

    with label_map_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Label map file must contain a JSON object.")

    label_map: dict[str, int] = {}
    for label, value in payload.items():
        if not isinstance(label, str):
            raise ValueError("All label map keys must be strings.")
        if not isinstance(value, int):
            raise ValueError(f"Label '{label}' must map to an integer value.")
        if not 0 <= value <= 255:
            raise ValueError(f"Label '{label}' value must be in [0, 255].")
        label_map[label] = value

    return label_map


def build_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        root=args.root.resolve(),
        data_dir_name=args.data_dir_name,
        mask_dir_name=args.mask_dir_name,
        label_map=load_label_map(args.label_map_file),
        strict=args.strict,
    )


def load_labelme_shapes(json_path: Path) -> tuple[int, int, list[dict]]:
    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    image_height = payload["imageHeight"]
    image_width = payload["imageWidth"]
    shapes = payload.get("shapes", [])

    if not isinstance(image_height, int) or not isinstance(image_width, int):
        raise ValueError("imageHeight and imageWidth must be integers.")
    if not isinstance(shapes, list):
        raise ValueError("shapes must be a list.")

    return image_height, image_width, shapes


def build_mask(
    image_height: int,
    image_width: int,
    shapes: list[dict],
    label_map: dict[str, int],
) -> Any:
    cv2 = require_cv2()
    np = require_numpy()
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for shape in shapes:
        label = shape["label"]
        if label not in label_map:
            raise KeyError(f"Unknown label: {label}")

        points = np.array(shape["points"], dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Invalid polygon points for label: {label}")

        cv2.fillConvexPoly(mask, points, label_map[label])

    return mask


def resolve_output_dir(json_path: Path, data_dir_name: str, mask_dir_name: str) -> Path:
    try:
        relative_parts = json_path.parent.relative_to(json_path.anchor).parts
    except ValueError:
        relative_parts = json_path.parent.parts

    replaced = [mask_dir_name if part == data_dir_name else part for part in relative_parts]
    if data_dir_name not in relative_parts:
        return json_path.parent / mask_dir_name
    return Path(json_path.anchor, *replaced)


def write_mask(mask: Any, output_path: Path) -> None:
    cv2 = require_cv2()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), mask)
    if not success:
        raise OSError(f"Failed to write mask image: {output_path}")


def json_to_mask(json_path: Path, output_dir: Path, label_map: dict[str, int]) -> Path:
    image_height, image_width, shapes = load_labelme_shapes(json_path)
    mask = build_mask(image_height, image_width, shapes, label_map)

    output_path = output_dir / f"{json_path.stem}_mask.png"
    write_mask(mask, output_path)
    return output_path


def iter_json_files(root: Path) -> Iterable[Path]:
    return root.rglob("*.json")


def require_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required to generate mask images. Install 'opencv-python' first."
        ) from exc
    return cv2


def require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "NumPy is required to generate mask images. Install 'numpy' first."
        ) from exc
    return np


def process_all(config: AppConfig) -> tuple[int, int]:
    processed = 0
    failed = 0

    for json_path in iter_json_files(config.root):
        output_dir = resolve_output_dir(
            json_path,
            data_dir_name=config.data_dir_name,
            mask_dir_name=config.mask_dir_name,
        )
        try:
            output_path = json_to_mask(json_path, output_dir, config.label_map)
        except Exception as exc:
            failed += 1
            logging.error("Failed: %s (%s)", json_path, exc)
            if config.strict:
                raise
            continue

        processed += 1
        logging.info("input:  %s", json_path)
        logging.info("output: %s", output_path)
        logging.info("count:  %s", processed)

    return processed, failed


def main() -> int:
    setup_logging()
    args = parse_args()
    config = build_config(args)

    processed, failed = process_all(config)
    logging.info("json2mask finished. processed=%s failed=%s", processed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
