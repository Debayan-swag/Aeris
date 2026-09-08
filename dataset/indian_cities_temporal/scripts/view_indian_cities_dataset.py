"""Display actual image files recorded by the local temporal-dataset CSV."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
CSV_PATH = REPO / "dataset" / "indian_cities_temporal" / "metadata" / "indian_cities_images.csv"

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--city", help="City name (case-insensitive)"); args = parser.parse_args()
    with CSV_PATH.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    cities = sorted({r["city"] for r in rows})
    if not args.city:
        print("Available cities: " + (", ".join(cities) or "none")); return
    selected = [r for r in rows if r["city"].casefold() == args.city.casefold()]
    if not selected: raise SystemExit(f"No records for {args.city!r}. Available cities: {', '.join(cities) or 'none'}")
    fig, axes = plt.subplots(1, len(selected), squeeze=False, figsize=(5 * len(selected), 5))
    fig.suptitle(selected[0]["city"])
    for number, (axis, row) in enumerate(zip(axes[0], selected), start=1):
        with Image.open(REPO / row["image_path"]) as image: axis.imshow(image.copy())
        axis.set_title(f"Image {number}\nAcquisition Date: {row['acquisition_date']}"); axis.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__": main()
