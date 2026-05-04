#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from torch.optim import AdamW

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_parquet("weak_labels.parquet")
print(f"Loaded {len(df)} docs")

gold_mask  = df["PIW_Label"].notna()
train_mask = ~gold_mask  # all non-gold docs for training (soft labels handle uncertainty)

df_train = df[train_mask].reset_index(drop=True)
df_test  = df[gold_mask].reset_index(drop=True)

print(f"Train: {len(df_train)} docs | Test: {len(df_test)} docs")
print(f"Test PIW: {(df_test['PIW_Label'] == 1).sum()} | NOT_PIW: {(df_test['PIW_Label'] == 0).sum()}")

# ── Dataset ────────────────────────────────────────────────────────────────────

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class SITREPDataset(Dataset):
    def __init__(self, texts, soft_labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(soft_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }

print("Tokenizing training set...")
train_texts  = df_train["text"].tolist()
train_labels = df_train["weak_prob_piw"].values.astype(float)

print("Tokenizing test set...")
test_texts  = df_test["text"].tolist()
test_labels = df_test["PIW_Label"].values.astype(float)

train_dataset = SITREPDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
test_dataset  = SITREPDataset(test_texts,  test_labels,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Model ──────────────────────────────────────────────────────────────────────

model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
loss_fn = nn.BCEWithLogitsLoss()

# ── Training ───────────────────────────────────────────────────────────────────

def evaluate_model(threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].numpy())
    probs  = np.array(all_probs)
    labels = np.array(all_labels).astype(int)
    return probs, labels

print(f"\nStarting training — {EPOCHS} epochs, {len(train_loader)} batches/epoch")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        loss   = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (i + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {total_loss/(i+1):.4f}")

    print(f"Epoch {epoch+1} complete | Avg Loss: {total_loss/len(train_loader):.4f}")

# ── Evaluation ─────────────────────────────────────────────────────────────────

print("\n── DistilBERT — soft weak labels (evaluated on gold labels) ──")
probs, y_test = evaluate_model()

print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'PIW Predicted':>14}")
for t in np.arange(0.3, 0.85, 0.05):
    y_t = (probs >= t).astype(int)
    p = precision_score(y_test, y_t, zero_division=0)
    r = recall_score(y_test, y_t, zero_division=0)
    f = f1_score(y_test, y_t, zero_division=0)
    print(f"{t:>10.2f}  {p:>10.3f}  {r:>8.3f}  {f:>6.3f}  {y_t.sum():>14}")

y_pred = (probs >= 0.5).astype(int)
print(f"\n── Classification Report @ threshold=0.50 ──")
print(classification_report(y_test, y_pred, target_names=["NOT_PIW", "PIW"]))

# Save model and tokenizer for future use
model.save_pretrained("distilbert_piw_model")
tokenizer.save_pretrained("distilbert_piw_model")
print("Model saved to distilbert_piw_model/")

# %%
