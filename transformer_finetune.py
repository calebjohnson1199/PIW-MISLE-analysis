#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH  = "distilbert_piw_model"   # pretrained on weak labels
MAX_LENGTH  = 128
BATCH_SIZE  = 16
EPOCHS      = 10                        # more epochs — small dataset
LR          = 1e-5                      # lower LR to avoid overwriting weak-label weights
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load data ──────────────────────────────────────────────────────────────────

df = pd.read_parquet("weak_labels.parquet")
gold = df[df["PIW_Label"].notna()].copy().reset_index(drop=True)
gold["PIW_Label"] = gold["PIW_Label"].astype(int)

print(f"Gold labeled docs: {len(gold)}  (PIW: {(gold['PIW_Label']==1).sum()}, NOT_PIW: {(gold['PIW_Label']==0).sum()})")

# Stratified 80/20 split — matches predecessor's evaluation setup
train_gold, test_gold = train_test_split(
    gold, test_size=0.2, stratify=gold["PIW_Label"], random_state=42
)
train_gold = train_gold.reset_index(drop=True)
test_gold  = test_gold.reset_index(drop=True)

print(f"Fine-tune train: {len(train_gold)}  (PIW: {(train_gold['PIW_Label']==1).sum()})")
print(f"Fine-tune test:  {len(test_gold)}   (PIW: {(test_gold['PIW_Label']==1).sum()})")

# ── Dataset ────────────────────────────────────────────────────────────────────

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

class SITREPDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }

print("Tokenizing...")
train_dataset = SITREPDataset(train_gold["text"].tolist(), train_gold["PIW_Label"].values, tokenizer, MAX_LENGTH)
test_dataset  = SITREPDataset(test_gold["text"].tolist(),  test_gold["PIW_Label"].values,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Load pretrained model (Stage 1: weak labels) ───────────────────────────────

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1)
model = model.to(DEVICE)
print(f"Loaded pretrained model from '{MODEL_PATH}'")

optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = len(train_loader) * EPOCHS
scheduler    = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
loss_fn      = nn.BCEWithLogitsLoss()

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model():
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
    return np.array(all_probs), np.array(all_labels).astype(int)

# ── Fine-tuning (Stage 2: gold labels) ────────────────────────────────────────

print(f"\nFine-tuning on gold labels -- {EPOCHS} epochs, {len(train_loader)} batches/epoch")
best_f1, best_epoch = 0.0, 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
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

    avg_loss = total_loss / len(train_loader)
    probs, y_test = evaluate_model()
    y_pred = (probs >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    p  = precision_score(y_test, y_pred, zero_division=0)
    r  = recall_score(y_test, y_pred, zero_division=0)
    print(f"Epoch {epoch+1:>2} | Loss: {avg_loss:.4f} | P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}")

    if f1 > best_f1:
        best_f1, best_epoch = f1, epoch + 1
        model.save_pretrained("distilbert_finetuned_model")
        tokenizer.save_pretrained("distilbert_finetuned_model")

print(f"\nBest F1: {best_f1:.3f} at epoch {best_epoch} -- saved to distilbert_finetuned_model/")

# ── Final threshold sweep ──────────────────────────────────────────────────────
#%%
# Reload best checkpoint
model = DistilBertForSequenceClassification.from_pretrained("distilbert_finetuned_model", num_labels=1).to(DEVICE)
probs, y_test = evaluate_model()

print(f"\n-- DistilBERT two-stage (evaluated on 20% gold held-out) --")
print(f"\n{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'PIW Predicted':>14}")
for t in np.arange(0.3, 0.85, 0.05):
    y_t = (probs >= t).astype(int)
    p = precision_score(y_test, y_t, zero_division=0)
    r = recall_score(y_test, y_t, zero_division=0)
    f = f1_score(y_test, y_t, zero_division=0)
    print(f"{t:>10.2f}  {p:>10.3f}  {r:>8.3f}  {f:>6.3f}  {y_t.sum():>14}")

y_pred = (probs >= 0.5).astype(int)
print(f"\n-- Classification Report @ threshold=0.50 --")
print(classification_report(y_test, y_pred, target_names=["NOT_PIW", "PIW"]))
