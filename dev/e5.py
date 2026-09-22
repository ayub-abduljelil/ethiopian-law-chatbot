from transformers import AutoTokenizer, AutoModel

model_name = "intfloat/multilingual-e5-small"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer OK")

print("Loading model...")
model = AutoModel.from_pretrained(model_name)
print("Model OK")