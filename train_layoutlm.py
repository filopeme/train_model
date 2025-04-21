# train_layoutlm.py
import json
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["Invoice", "Poliza", "Packing List", "Other"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

class DocumentDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["words"],
            boxes=item["boxes"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
            is_split_into_words=True,
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label2id[item["label"]])
        return encoding

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

def train():
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    model = LayoutLMForSequenceClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased",
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id
    )

    with open("train.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    dataset = DocumentDataset(data, tokenizer)

    args = TrainingArguments(
        output_dir="./model_output",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("fine_tuned_layoutlm")
    tokenizer.save_pretrained("fine_tuned_layoutlm")

if __name__ == "__main__":
    train()
