import json
import os

def extract_layoutlm_data(json_path, label, output_dir="train_data"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the input JSON (assuming Textract-like structure)
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Placeholder for extracted data
    extracted_data = {
        "words": [],
        "bboxes": [],
        "label": label
    }

    # Example extraction logic
    for block in data.get("Blocks", []):
        if block.get("BlockType") == "WORD":
            extracted_data["words"].append(block["Text"])
            extracted_data["bboxes"].append(block["Geometry"]["BoundingBox"])

    # Convert bounding boxes to expected format (if needed)
    # LayoutLM expects bbox in [x0, y0, x1, y1] format scaled to 0-1000
    processed_bboxes = []
    for bbox in extracted_data["bboxes"]:
        x0 = int(bbox["Left"] * 1000)
        y0 = int(bbox["Top"] * 1000)
        x1 = int((bbox["Left"] + bbox["Width"]) * 1000)
        y1 = int((bbox["Top"] + bbox["Height"]) * 1000)
        processed_bboxes.append([x0, y0, x1, y1])

    # Prepare final JSONL line
    jsonl_data = {
        "words": extracted_data["words"],
        "bboxes": processed_bboxes,
        "label": extracted_data["label"]
    }

    # Save as .jsonl
    output_path = os.path.join(output_dir, os.path.basename(json_path).replace(".json", ".jsonl"))
    with open(output_path, 'w') as f_out:
        f_out.write(json.dumps(jsonl_data) + "\n")

    return output_path
