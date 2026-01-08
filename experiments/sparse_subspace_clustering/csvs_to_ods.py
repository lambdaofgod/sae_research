import fire
import pandas as pd
from pathlib import Path


def main(path: str, prefix: str = "goodfire_sae_labels"):
    dir = Path(path)
    csvs = list(dir.glob(f"{prefix}*.csv"))
    output_path = dir / f"{prefix}.ods"
    with pd.ExcelWriter(output_path, engine="odf") as writer:
        for csv in csvs:
            suffix = csv.stem.removeprefix(prefix + "_") if csv.stem != prefix else "base"
            pd.read_csv(csv).to_excel(writer, sheet_name=suffix, index=False)
    print("Created:", output_path)
    print("Sheets:", [csv.stem.removeprefix(prefix + "_") if csv.stem != prefix else "base" for csv in csvs])


if __name__ == "__main__":
    fire.Fire(main)
