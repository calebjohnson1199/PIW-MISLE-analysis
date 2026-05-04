#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# ── Load data ──────────────────────────────────────────────────────────────────

embeddings = np.load("embeddings.npy", allow_pickle=True)
df = pd.read_parquet("weak_labels.parquet")

print(f"Embeddings shape: {embeddings.shape}")
print(f"DataFrame shape:  {df.shape}")

# ── Build structured features ──────────────────────────────────────────────────

subtype_dummies = pd.get_dummies(df["Incident Sub Type"], prefix="subtype", dtype=float)

weather_cols = ["Wind Speed", "Wave Ht", "Water Temp", "Air Temp", "Visibility", "Gust Speed"]

def parse_weather_col(series):
    """Extract numeric value from cells like array(['5KTS '], dtype=object) or '5KTS '."""
    import re
    def extract(val):
        if val is None:
            return 0.0
        # Unwrap single-element arrays
        if hasattr(val, '__len__') and not isinstance(val, str):
            val = val[0] if len(val) > 0 else ''
        m = re.search(r'\d+\.?\d*', str(val))
        return float(m.group()) if m else 0.0
    return series.apply(extract)

weather_feats = np.column_stack([parse_weather_col(df[c]) for c in weather_cols])

text_length = df["text"].str.len().values.reshape(-1, 1)

# Concatenate embeddings + sub type + weather + text length
X_all = np.hstack([embeddings, subtype_dummies.values, weather_feats, text_length])
print(f"Combined feature shape: {X_all.shape}  "
      f"(embeddings={embeddings.shape[1]}, subtype={subtype_dummies.shape[1]}, weather={len(weather_cols)}, length=1)")

# ── Split gold vs. weakly labeled ─────────────────────────────────────────────

gold_mask  = df["PIW_Label"].notna()
train_mask = (~gold_mask) & (df["weak_label"] != -1)

X_train = X_all[train_mask]
y_train = df.loc[train_mask, "weak_label"].values

X_test  = X_all[gold_mask]
y_test  = df.loc[gold_mask, "PIW_Label"].values.astype(int)

print(f"\nTraining set: {X_train.shape[0]} docs  "
      f"({(y_train == 1).sum()} PIW, {(y_train == 0).sum()} NOT_PIW)")
print(f"Test set:     {X_test.shape[0]} docs  "
      f"({(y_test == 1).sum()} PIW, {(y_test == 0).sum()} NOT_PIW)")

# ── Train logistic regression ──────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

def evaluate(name, y_true, y_prob_piw):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    y_pred = (y_prob_piw >= 0.5).astype(int)
    print(classification_report(y_true, y_pred, target_names=["NOT_PIW", "PIW"]))
    print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'PIW Predicted':>14}")
    for t in np.arange(0.3, 0.85, 0.05):
        y_t = (y_prob_piw >= t).astype(int)
        p = precision_score(y_true, y_t, zero_division=0)
        r = recall_score(y_true, y_t, zero_division=0)
        f = f1_score(y_true, y_t, zero_division=0)
        print(f"{t:>10.2f}  {p:>10.3f}  {r:>8.3f}  {f:>6.3f}  {y_t.sum():>14}")

# ── Model 1: Logistic Regression on weak labels ────────────────────────────────

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_scaled, y_train)
evaluate("LR — weak labels only (embeddings + subtype)", y_test, lr.predict_proba(X_test_scaled)[:, 1])

# ── Model 2: Random Forest on weak labels ─────────────────────────────────────

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # RF doesn't need scaling
evaluate("Random Forest — weak labels + weather", y_test, rf.predict_proba(X_test)[:, 1])

# ── Model 3: LightGBM on weak labels (hard labels) ────────────────────────────

lgbm = LGBMClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
lgbm.fit(X_train, y_train)
evaluate("LightGBM — hard weak labels + weather", y_test, lgbm.predict_proba(X_test)[:, 1])

# ── Model 4: LightGBM with soft weak_prob_piw sample weights ──────────────────
# Instead of hard 0/1 labels, weight each training doc by how confident the
# LabelModel was — high-probability PIW docs count more, uncertain ones less.

weak_prob = df.loc[train_mask, "weak_prob_piw"].values
# For PIW docs: weight = weak_prob_piw; for NOT_PIW docs: weight = 1 - weak_prob_piw
soft_weights = np.where(y_train == 1, weak_prob, 1.0 - weak_prob)
soft_weights = np.clip(soft_weights, 0.01, 1.0)  # avoid zero weights

lgbm_soft = LGBMClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
lgbm_soft.fit(X_train, y_train, sample_weight=soft_weights)
evaluate("LightGBM — soft weak labels + weather", y_test, lgbm_soft.predict_proba(X_test)[:, 1])

# ── Model 5: LR trained on 80% of gold labels, tested on 20% ─────────────────
# This shows the ceiling — what's achievable with real labels

X_gold = X_all[gold_mask]
y_gold = df.loc[gold_mask, "PIW_Label"].values.astype(int)
Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(X_gold, y_gold, test_size=0.2, stratify=y_gold, random_state=42)
scaler_gold = StandardScaler()
Xg_tr_s = scaler_gold.fit_transform(Xg_tr)
Xg_te_s = scaler_gold.transform(Xg_te)
lr_gold = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_gold.fit(Xg_tr_s, yg_tr)
evaluate("LR — gold labels only (upper bound reference)", yg_te, lr_gold.predict_proba(Xg_te_s)[:, 1])

# ── Model 5: LightGBM on gold labels (true ceiling) ───────────────────────────

lgbm_gold = LGBMClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
lgbm_gold.fit(Xg_tr, yg_tr)
evaluate("LightGBM — gold labels only (true ceiling)", yg_te, lgbm_gold.predict_proba(Xg_te)[:, 1])
