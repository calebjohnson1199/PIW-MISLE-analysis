#%%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ── Load model ─────────────────────────────────────────────────────────────────

MODEL_PATH = "distilbert_finetuned_model"
THRESHOLD  = 0.65   # best operating point (F1=0.769)
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=1)
model     = model.to(DEVICE)
model.eval()
print(f"Model loaded from '{MODEL_PATH}' | Device: {DEVICE}\n")

# ── Prediction function ────────────────────────────────────────────────────────

def predict(text):
    inputs = tokenizer(
        text, truncation=True, padding=True,
        max_length=128, return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logit = model(**inputs).logits.squeeze(-1)
        prob  = torch.sigmoid(logit).item()
    label = "PIW" if prob >= THRESHOLD else "NOT PIW"
    return prob, label

# ── Demo 1: Held-out gold docs (model never trained on these) ──────────────────

from sklearn.model_selection import train_test_split

df   = pd.read_parquet("weak_labels.parquet")
print(df[["PIW_Label", "text"]].head())
gold = df[df["PIW_Label"].notna()].copy().reset_index(drop=True)
gold["PIW_Label"] = gold["PIW_Label"].astype(int)

_, test_gold = train_test_split(gold, test_size=0.2, stratify=gold["PIW_Label"], random_state=42)
test_gold = test_gold.reset_index(drop=True)

print("=" * 70)
print("  DEMO 1: Random held-out gold cases (never seen during training)")
print("=" * 70)

# Show 5 PIW and 5 NOT_PIW examples
piw_examples     = test_gold[test_gold["PIW_Label"] == 1].sample(5, random_state=7)
not_piw_examples = test_gold[test_gold["PIW_Label"] == 0].sample(5, random_state=7)
examples = pd.concat([piw_examples, not_piw_examples]).sample(frac=1, random_state=42)

for _, row in examples.iterrows():
    prob, label = predict(row["text"])
    true_label  = "PIW" if row["PIW_Label"] == 1 else "NOT PIW"
    correct     = "CORRECT" if label == true_label else "WRONG"
    print(f"\nNarrative: {row['text'][:1000]}{'...' if len(row['text']) > 1000 else ''}")
    print(f"  True label:      {true_label}")
    print(f"  Predicted:       {label}  (prob={prob:.3f})  [{correct}]")

# ── Evaluation: Full test_gold set ────────────────────────────────────────────

print("\n" + "=" * 70)
print("  EVALUATION: Full held-out gold set")
print("=" * 70)

y_true, y_pred, y_prob = [], [], []
for _, row in test_gold.iterrows():
    prob, label = predict(row["text"])
    y_true.append(int(row["PIW_Label"]))
    y_pred.append(1 if label == "PIW" else 0)
    y_prob.append(prob)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Confusion Matrix (threshold={THRESHOLD}):")
print(f"  {'':20s}  Pred NOT PIW   Pred PIW")
print(f"  {'True NOT PIW':20s}  {tn:^13d}  {fp:^8d}")
print(f"  {'True PIW':20s}  {fn:^13d}  {tp:^8d}")
print(f"\n  Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"  Precision: {precision_score(y_true, y_pred):.3f}")
print(f"  Recall:    {recall_score(y_true, y_pred):.3f}")
print(f"  F1:        {f1_score(y_true, y_pred):.3f}")
print(f"\n{classification_report(y_true, y_pred, target_names=['NOT PIW', 'PIW'])}")
'''
# ── Threshold sweep plot ───────────────────────────────────────────────────────

thresholds = np.arange(0.01, 1.00, 0.01)
precisions, recalls, f1s = [], [], []

y_true_arr = np.array(y_true)
y_prob_arr = np.array(y_prob)

for t in thresholds:
    y_t = (y_prob_arr >= t).astype(int)
    precisions.append(precision_score(y_true_arr, y_t, zero_division=0))
    recalls.append(recall_score(y_true_arr, y_t, zero_division=0))
    f1s.append(f1_score(y_true_arr, y_t, zero_division=0))

best_idx = int(np.argmax(f1s))
best_t   = thresholds[best_idx]
best_f1  = f1s[best_idx]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, precisions, label="Precision", color="steelblue")
ax.plot(thresholds, recalls,    label="Recall",    color="darkorange")
ax.plot(thresholds, f1s,        label="F1",        color="seagreen", linewidth=2)

ax.axvline(THRESHOLD, color="crimson", linestyle="--", linewidth=1.5,
           label=f"Chosen threshold ({THRESHOLD}) — F1={f1_score(y_true_arr, (y_prob_arr >= THRESHOLD).astype(int), zero_division=0):.3f}")
ax.axvline(best_t, color="gray", linestyle=":", linewidth=1.5,
           label=f"Peak F1 threshold ({best_t:.2f}) — F1={best_f1:.3f}")

ax.set_xlabel("Threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Precision / Recall / F1 vs. Classification Threshold\n(DistilBERT Stage 2, gold held-out set)", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("threshold_curve.png", dpi=150)
plt.show()
print("\nPlot saved to threshold_curve.png")
'''