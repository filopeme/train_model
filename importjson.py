import json
import os


def extract_layoutlm_data(json_path, label, output_dir="train_data"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load Textract JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    words = []
    boxes = []

    for block in data.get("Blocks", []):
        if block.get("BlockType") == "WORD":
            words.append(block["Text"])
            bbox = block["Geometry"]["BoundingBox"]
            x0 = int(bbox["Left"] * 1000)
            y0 = int(bbox["Top"] * 1000)
            x1 = int((bbox["Left"] + bbox["Width"]) * 1000)
            y1 = int((bbox["Top"] + bbox["Height"]) * 1000)
            boxes.append([x0, y0, x1, y1])

    # Final JSONL structure
    jsonl_data = {
        "words": words,
        "boxes": boxes,
        "label": label
    }

    # Always save as train.jsonl
    output_path = os.path.join(output_dir, "train.jsonl")
    with open(output_path, 'w') as f_out:
        f_out.write(json.dumps(jsonl_data) + "\n")

    print(f"✅ Generated training file at: {output_path}")
    return output_path
