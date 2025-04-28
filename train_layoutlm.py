# train_layoutlm.py
import os
import json
import torch
import shutil
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
dummy_image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
# ----- Labels -----
LABELS = ["Invoice", "Poliza", "Packing List", "Other"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# ----- Dataset Class -----
class DocumentDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.processor(
            images=dummy_image, 
            text=item["words"],
            boxes=item["boxes"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label2id[item["label"]])
        return encoding

# ----- Metrics -----
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

#import os
#import json
#import shutil
#from transformers import AutoProcessor, AutoModelForSequenceClassification, TrainingArguments, Trainer

def train():
    print("🔹 Loading processor and model...")

    #volume_path = "/app/model_volume"
    volume_path="/train_model_dsk"
    model_dir = os.path.join(volume_path, "fine_tuned_layoutlmv3")

    if os.path.exists(model_dir):
        print(f"🔹 Found existing model in Docker volume at '{model_dir}', loading it...")
        processor = AutoProcessor.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id
        )
    else:
        print("🔹 No existing model in volume, loading base model...")
        processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id
        )

    # Load dataset
    jsonl_dir = "train_data"
    jsonl_files = [f for f in os.listdir(jsonl_dir) if f.endswith(".jsonl")]

    if not jsonl_files:
        print("❌ No .jsonl files found in 'train_data' directory!")
        return

    jsonl_path = os.path.join(jsonl_dir, jsonl_files[0])
    print(f"🔹 Loading dataset from {jsonl_path}...")

    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]

    dataset = DocumentDataset(data, processor)

    args = TrainingArguments(
        output_dir="./model_output",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=5
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        compute_metrics=compute_metrics
    )

    print("🚀 Starting training...")
    trainer.train()

    print("💾 Saving model to Docker volume...")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)  # Clean old model

    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)

    print(f"✅ Model saved to volume at '{model_dir}'")
    print("✅ Training complete!")

if __name__ == "__main__":
    train()
