import json
import re
from pathlib import Path


def convert_global2imgpath(input_file: str, output_file: str):
    # Load the JSON
    with open(input_file, "r") as f:
        data = json.load(f)

    result = {}
    for k, v in data.items():
        path = Path(v)

        # Extract Lxx and Vxxx
        match_l = re.search(r"L(\d+)", str(path))
        match_v = re.search(r"V(\d+)", str(path))
        frame_num = int(path.stem)  # filename without extension, as integer

        if not match_l or not match_v:
            raise ValueError(f"Unexpected path format: {v}")

        l_num = int(match_l.group(1))  # e.g. L01 -> 1
        v_num = int(match_v.group(1))  # e.g. V001 -> 1

        result[k] = f"{l_num}/{v_num}/{frame_num}"

    # Save to new file
    file_path = Path(output_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    convert_global2imgpath(
        "migration_data/global2imgpath.json",
        "data_collection/converter/id2index.json"
    )
