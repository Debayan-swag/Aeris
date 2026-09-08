"""
TerraWatch AI - Extract Images from Sentinel-2 Parquet

Reads the Sentinel-2 Parquet dataset in batches and extracts image files
without loading the entire Parquet file into memory.

Expected project structure:

TerraWatch/
│
├── data/
│   ├── raw/
│   │   └── sentinel2.parquet
│   │
│   └── processed/
│       ├── images/
│       └── metadata/
│
└── scripts/
    └── extract_images.py
"""

from pathlib import Path
import io
import csv
import argparse

import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PARQUET_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "sentinel2.parquet"
)

IMAGES_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "images"
)

METADATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "metadata"
)

METADATA_FILE = (
    METADATA_DIR
    / "metadata.csv"
)

ERROR_FILE = (
    METADATA_DIR
    / "extraction_errors.csv"
)


# ============================================================
# EXTRACT COORDINATES
# ============================================================

def extract_coordinates(image_name: str):
    """
    Extract longitude and latitude from image name.

    Expected format:

        -113.917243,51.101323.jpg

    Returns:

        longitude, latitude

    If coordinates cannot be extracted:

        None, None
    """

    try:

        filename = Path(image_name).stem

        longitude, latitude = filename.split(",")

        return (
            float(longitude),
            float(latitude)
        )

    except (ValueError, AttributeError):

        return None, None


# ============================================================
# DETECT IMAGE EXTENSION
# ============================================================

def get_extension(image_format: str | None) -> str:
    """
    Convert PIL image format to a file extension.
    """

    extensions = {

        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "TIFF": ".tif",
        "TIF": ".tif",
        "WEBP": ".webp",

    }

    if image_format:

        return extensions.get(
            image_format.upper(),
            ".png"
        )

    return ".bin"


# ============================================================
# EXTRACT IMAGES
# ============================================================

def extract_images(
    limit: int | None = 1000,
    batch_size: int = 100,
):
    """
    Extract images from Parquet.

    Parameters
    ----------

    limit:
        Maximum number of images to extract.

        Example:
            1000

        Use None to extract all images.

    batch_size:
        Number of Parquet rows to process at once.
    """


    # --------------------------------------------------------
    # CHECK PARQUET FILE
    # --------------------------------------------------------

    if not PARQUET_PATH.exists():

        print("\nERROR: Parquet file not found!")

        print(f"\nExpected location:")

        print(PARQUET_PATH)

        return


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # OPEN PARQUET
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "TERRAWATCH AI - SENTINEL-2 IMAGE EXTRACTION"
    )

    print("=" * 70)

    print("\nOpening Parquet file...")

    parquet_file = pq.ParquetFile(
        PARQUET_PATH
    )

    total_dataset_rows = (
        parquet_file.metadata.num_rows
    )

    total_to_process = (
        total_dataset_rows
        if limit is None
        else min(
            total_dataset_rows,
            limit
        )
    )


    print(
        f"\nTotal images in dataset: "
        f"{total_dataset_rows:,}"
    )

    print(
        f"Images to extract: "
        f"{total_to_process:,}"
    )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"\nOutput images directory:"
    )

    print(IMAGES_DIR)

    print("\nStarting extraction...\n")


    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    processed = 0
    successful = 0
    failed = 0
    skipped = 0

    metadata_rows = []
    error_rows = []


    # --------------------------------------------------------
    # PROGRESS BAR
    # --------------------------------------------------------

    progress_bar = tqdm(

        total=total_to_process,

        desc="Extracting images",

        unit="image"

    )


    # --------------------------------------------------------
    # READ PARQUET IN BATCHES
    # --------------------------------------------------------

    required_columns = [

        "image_no",

        "image_name",

        "image"

    ]


    try:

        for batch in parquet_file.iter_batches(

            batch_size=batch_size,

            columns=required_columns

        ):


            # Convert ONLY CURRENT BATCH to Pandas

            df = batch.to_pandas()


            # ------------------------------------------------
            # PROCESS ROWS
            # ------------------------------------------------

            for _, row in df.iterrows():


                # Stop at requested limit

                if limit is not None and processed >= limit:

                    break


                processed += 1


                # --------------------------------------------
                # READ VALUES
                # --------------------------------------------

                try:

                    image_no = int(

                        row["image_no"]

                    )

                    image_name = str(

                        row["image_name"]

                    )

                    image_bytes = (

                        row["image"]

                    )

                except Exception as error:

                    failed += 1

                    error_rows.append({

                        "image_no": None,

                        "image_name": None,

                        "error": str(error)

                    })

                    progress_bar.update(1)

                    continue


                # --------------------------------------------
                # EXTRACT COORDINATES
                # --------------------------------------------

                longitude, latitude = (

                    extract_coordinates(

                        image_name

                    )

                )


                # --------------------------------------------
                # VALIDATE IMAGE
                # --------------------------------------------

                try:

                    image = Image.open(

                        io.BytesIO(

                            image_bytes

                        )

                    )

                    image.load()


                    width, height = (

                        image.size

                    )

                    image_format = (

                        image.format
                        or "UNKNOWN"

                    )

                    extension = (

                        get_extension(

                            image_format

                        )

                    )


                except (

                    UnidentifiedImageError,

                    OSError,

                    ValueError

                ) as error:


                    failed += 1


                    error_rows.append({

                        "image_no":

                            image_no,

                        "image_name":

                            image_name,

                        "error":

                            f"Invalid image: {error}"

                    })


                    progress_bar.update(1)

                    continue


                # --------------------------------------------
                # CREATE STANDARDIZED FILENAME
                # --------------------------------------------

                image_id = (

                    f"S2_{image_no:06d}"

                )

                output_filename = (

                    image_id

                    +

                    extension

                )

                output_path = (

                    IMAGES_DIR

                    /

                    output_filename

                )


                # --------------------------------------------
                # SAVE IMAGE
                # --------------------------------------------

                try:

                    if output_path.exists():

                        skipped += 1


                    else:

                        with open(

                            output_path,

                            "wb"

                        ) as file:

                            file.write(

                                image_bytes

                            )


                        successful += 1


                    # ----------------------------------------
                    # STORE METADATA
                    # ----------------------------------------

                    metadata_rows.append({

                        "image_id":

                            image_id,

                        "image_no":

                            image_no,

                        "image_name":

                            image_name,

                        "image_path":

                            str(

                                output_path.relative_to(

                                    BASE_DIR

                                )

                            ),

                        "longitude":

                            longitude,

                        "latitude":

                            latitude,

                        "width":

                            width,

                        "height":

                            height,

                        "format":

                            image_format,

                        "file_size_bytes":

                            len(image_bytes)

                    })


                except Exception as error:


                    failed += 1


                    error_rows.append({

                        "image_no":

                            image_no,

                        "image_name":

                            image_name,

                        "error":

                            str(error)

                    })


                progress_bar.update(1)


            # Stop outer loop if limit reached

            if (

                limit is not None

                and processed >= limit

            ):

                break


    finally:

        progress_bar.close()


    # ========================================================
    # SAVE METADATA CSV
    # ========================================================

    print("\nSaving metadata...")


    metadata_columns = [

        "image_id",

        "image_no",

        "image_name",

        "image_path",

        "longitude",

        "latitude",

        "width",

        "height",

        "format",

        "file_size_bytes"

    ]


    with open(

        METADATA_FILE,

        "w",

        newline="",

        encoding="utf-8"

    ) as file:


        writer = csv.DictWriter(

            file,

            fieldnames=

                metadata_columns

        )


        writer.writeheader()


        writer.writerows(

            metadata_rows

        )


    # ========================================================
    # SAVE ERROR LOG
    # ========================================================

    if error_rows:


        with open(

            ERROR_FILE,

            "w",

            newline="",

            encoding="utf-8"

        ) as file:


            writer = csv.DictWriter(

                file,

                fieldnames=[

                    "image_no",

                    "image_name",

                    "error"

                ]

            )


            writer.writeheader()


            writer.writerows(

                error_rows

            )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)

    print(

        "EXTRACTION COMPLETE"

    )

    print("=" * 70)


    print(

        f"\nProcessed: "

        f"{processed:,}"

    )


    print(

        f"Successfully extracted: "

        f"{successful:,}"

    )


    print(

        f"Skipped existing: "

        f"{skipped:,}"

    )


    print(

        f"Failed: "

        f"{failed:,}"

    )


    print(

        f"\nImages saved to:"

    )

    print(

        IMAGES_DIR.resolve()

    )


    print(

        f"\nMetadata saved to:"

    )

    print(

        METADATA_FILE.resolve()

    )


    if error_rows:


        print(

            f"\nErrors saved to:"

        )

        print(

            ERROR_FILE.resolve()

        )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":


    parser = argparse.ArgumentParser(

        description=

        "Extract Sentinel-2 images from Parquet"

    )


    parser.add_argument(

        "--limit",

        type=int,

        default=1000,

        help=

        "Maximum images to extract. "

        "Use --limit 0 to extract all."

    )


    parser.add_argument(

        "--batch-size",

        type=int,

        default=100,

        help=

        "Number of rows processed per batch."

    )


    args = parser.parse_args()


    # Convert 0 to None

    extraction_limit = (

        None

        if args.limit == 0

        else args.limit

    )


    extract_images(

        limit=extraction_limit,

        batch_size=args.batch_size

    )