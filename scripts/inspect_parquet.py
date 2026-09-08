"""
inspect_parquet.py

Inspect the Sentinel-2 Parquet dataset safely without loading
the entire dataset into memory.
"""

from pathlib import Path
import sys
import pyarrow.parquet as pq


# Automatically finds the project root structure
RAW_PARQUET_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "sentinel2.parquet"
)


def inspect_value(value):
    """
    Inspect a single value and return useful information.
    """

    if value is None:
        return "None"

    # Image or binary data
    if isinstance(value, bytes):
        return (
            f"BINARY DATA\n"
            f"Size: {len(value):,} bytes\n"
            f"First bytes: {value[:20]}"
        )

    # String
    if isinstance(value, str):
        sample = value[:200]
        return f"STRING\nSample: {sample}"

    # List
    if isinstance(value, list):
        return (
            f"LIST\n"
            f"Length: {len(value)}\n"
            f"Sample: {value[:5]}"
        )

    # Dictionary
    if isinstance(value, dict):
        return (
            f"DICTIONARY\n"
            f"Keys: {list(value.keys())[:10]}"
        )

    # Other values
    return f"VALUE\nSample: {value}"


def inspect_parquet(file_path: Path = RAW_PARQUET_PATH) -> None:

    print("\n" + "=" * 70)
    print("TERRAWATCH AI - SENTINEL-2 PARQUET INSPECTION")
    print("=" * 70)

    # --------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------

    if not file_path.exists():
        print(f"\nERROR: Parquet file not found!")
        print(f"Expected location:\n{file_path}")
        sys.exit(1)

    print(f"\nFILE PATH:")
    print(file_path)

    print(f"\nFILE SIZE:")
    print(f"{file_path.stat().st_size / (1024 ** 2):.2f} MB")

    try:

        # --------------------------------------------------
        # OPEN PARQUET WITHOUT LOADING EVERYTHING
        # --------------------------------------------------

        parquet_file = pq.ParquetFile(file_path)

        metadata = parquet_file.metadata

        print("\n" + "=" * 70)
        print("PARQUET METADATA")
        print("=" * 70)

        print(f"\nTOTAL ROWS: {metadata.num_rows:,}")
        print(f"TOTAL COLUMNS: {metadata.num_columns}")
        print(f"ROW GROUPS: {metadata.num_row_groups}")

        # --------------------------------------------------
        # SCHEMA
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("SCHEMA")
        print("=" * 70)

        print(parquet_file.schema)

        # --------------------------------------------------
        # COLUMN NAMES
        # --------------------------------------------------

        column_names = parquet_file.schema.names

        print("\n" + "=" * 70)
        print("COLUMN NAMES")
        print("=" * 70)

        for i, column in enumerate(column_names, start=1):
            print(f"{i}. {column}")

        # --------------------------------------------------
        # READ ONLY FIRST 5 ROWS
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("SAMPLE DATA")
        print("=" * 70)

        batch = next(
            parquet_file.iter_batches(batch_size=5)
        )

        df = batch.to_pandas()

        print(df.head())

        # --------------------------------------------------
        # DATA TYPES
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("PANDAS DATA TYPES")
        print("=" * 70)

        print(df.dtypes)

        # --------------------------------------------------
        # DETAILED COLUMN ANALYSIS
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("DETAILED COLUMN ANALYSIS")
        print("=" * 70)

        for column in df.columns:

            print("\n" + "-" * 70)
            print(f"COLUMN: {column}")
            print(f"PANDAS TYPE: {df[column].dtype}")

            try:

                # Find first non-null value
                non_null = df[column].dropna()

                if len(non_null) == 0:
                    print("All sampled values are NULL.")
                    continue

                value = non_null.iloc[0]

                print(f"PYTHON TYPE: {type(value)}")

                result = inspect_value(value)

                print(result)

            except Exception as error:

                print(
                    f"Could not inspect column.\n"
                    f"Error: {error}"
                )

        # --------------------------------------------------
        # FINISH
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("INSPECTION COMPLETE")
        print("=" * 70)

    except Exception as error:

        print("\nERROR WHILE READING PARQUET FILE")
        print(error)

        sys.exit(1)


if __name__ == "__main__":
    inspect_parquet()