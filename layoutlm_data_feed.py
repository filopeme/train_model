# layoutlm_data_feed.py
import json

def textract_bbox_to_layoutlm(box):
    x0 = int(box["Left"] * 1000)
    y0 = int(box["Top"] * 1000)
    x1 = int((box["Left"] + box["Width"]) * 1000)
    y1 = int((box["Top"] + box["Height"]) * 1000)
    return [x0, y0, x1, y1]

def extract_layoutlm_data(textract_json_path, label, output_path="train.jsonl"):
    with open(textract_json_path, "r") as f:
        textract_data = json.load(f)

    words = []
    boxes = []

    for block in textract_data["Blocks"]:
        if block["BlockType"] == "WORD":
            words.append(block["Text"])
            bbox = textract_bbox_to_layoutlm(block["Geometry"]["BoundingBox"])
            boxes.append(bbox)

    result = {
        "words": words,
        "boxes": boxes,
        "label": label
    }

    with open(output_path, "w") as out:
        json.dump(result, out)
        out.write("\n")

    return output_path
