import json
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForSequenceClassification

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

def predict_document(jsonl_path):
    MODEL_VOLUME_PATH = "/train_model_dsk"
    MODEL_DIR = os.path.join(MODEL_VOLUME_PATH, "fine_tuned_layoutlmv3")

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"❌ Trained model not found at '{MODEL_DIR}'. Please train the model first.")

    print(f"🔹 Loading model from: {MODEL_DIR}")

    processor = AutoProcessor.from_pretrained(MODEL_DIR, apply_ocr=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    with open(jsonl_path) as f:
        data = json.loads(f.readline())

    dummy_image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))

    inputs = processor(
        images=dummy_image,
        text=data["words"],
        boxes=data["boxes"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_class_id = torch.argmax(probs, dim=-1).item()

    label = model.config.id2label[predicted_class_id]
    confidence = probs[0][predicted_class_id].item()

    return label, confidence

