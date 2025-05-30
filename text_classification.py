from datasets import load_dataset
import pandas as pd 
import matplotlib.pyplot as plt 


emotions = load_dataset("emotion")
train_ds = emotions['train']
emotions.set_format(type="pandas")
df = emotions["train"][:]
# df["label"].value_counts(ascending=True).plot.barh()
# plt.title("Freq classes")
# plt.savefig("plot.png")

# subword tokenization
from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
text = "Tokenizing text is a core task of NLP."
encoded_text = tokenizer(text)
print(encoded_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

# batch tokenization, ignore paddd areas with a mask

def tokenize(batch):
  return tokenizer(batch["text"], padding=True, truncation=True)

# operate on whole emotions dataset
# emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# transformers for feature extraction
from transformers import AutoModel
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device


