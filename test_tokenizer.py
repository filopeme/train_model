from transformers import LayoutLMv2TokenizerFast

# Use LayoutLMv2 instead of LayoutLMv1
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")

words = ["Hello", "world"]
boxes = [[100, 100, 200, 200], [150, 150, 250, 250]]

encoding = tokenizer(
    words,
    boxes=boxes,
    truncation=True,
    padding="max_length",
    max_length=512,
    return_tensors="pt"
)

print("✅ Tokenizer works! Encoding keys:", encoding.keys())
