import pandas as pd
import numpy as np

misle_df = pd.read_json("Narratives_With_Labels_2Feb2026_fixed.json", orient="records")
labeled_df = misle_df[misle_df['PIW_Label'].notna()]

result = labeled_df.loc[labeled_df["PIW_Label"] == 1.0, "SITREP Narrative"].tolist()

with open("posPIW_SITREP.txt", "w") as f:
    for item in result:
        f.write(str(item) + "\n")

'''
This doesn't really cleanly slice the SITREP Narratives of positive PIW labeled reports.
'''

#%%

import re
from snorkel.labeling import labeling_function

ABSTAIN, NOT_PIW, PIW = -1, 0, 1

# ── PIW LFs ────────────────────────────────────────────────────────────────────

@labeling_function()
def lf_piw_keyword(x):
    """Explicit PIW acronym."""
    pattern = r'\bPIW\b'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_person_in_water(x):
    """Phrase 'person(s) in the water'."""
    pattern = r'persons?\s+in\s+the\s+water'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_fell_overboard(x):
    """Fell or fallen overboard."""
    pattern = r'fell\s+overboard|fallen\s+overboard'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_ejected(x):
    """Person ejected from vessel."""
    pattern = r'\bejected\b'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_capsized(x):
    """Capsized/overturned vessel with persons aboard — excludes unmanned/adrift vessels."""
    has_capsized  = bool(re.search(r'\bcapsized\b|\boverturned\b', x.text, re.IGNORECASE))
    no_one_aboard = bool(re.search(r'no\s+persons?\s+aboard|unmanned|unoccupied|adrift\s+unmanned', x.text, re.IGNORECASE))
    return PIW if (has_capsized and not no_one_aboard) else ABSTAIN


@labeling_function()
def lf_went_in_water(x):
    """Went in the water / went into the water."""
    pattern = r'went\s+(in(to)?\s+)?the\s+water'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_recovered_from_water(x):
    """Recovered/pulled from the water."""
    pattern = r'(pulled|recovered|removed)\s+(out\s+of|from)\s+the\s+water'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_cpr(x):
    """CPR after water rescue — requires explicit water context to avoid shore-based medical FPs."""
    has_cpr   = bool(re.search(r'\bCPR\b', x.text, re.IGNORECASE))
    has_water = bool(re.search(r'in\s+the\s+water|PIW|overboard|capsized|pulled\s+from', x.text, re.IGNORECASE))
    return PIW if (has_cpr and has_water) else ABSTAIN


@labeling_function()
def lf_abandoned_vessel(x):
    """Person abandoned the vessel."""
    pattern = r'abandoned\s+the\s+vessel'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_washed_up(x):
    """Body/person washed up — post-PIW indicator."""
    pattern = r'washed\s+up'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_persons_in_water_with_pfds(x):
    """Persons in the water with PFDs."""
    pattern = r'persons?\s+were\s+in\s+the\s+water'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_hoisted(x):
    """Hoisted in context of a person rescue — requires PIW/survivor language to avoid equipment FPs."""
    has_hoisted  = bool(re.search(r'\bhoisted\b', x.text, re.IGNORECASE))
    has_person   = bool(re.search(r'\bPIW\b|survivor|person|patient|subject|victim|overboard|in\s+the\s+water', x.text, re.IGNORECASE))
    has_equipment = bool(re.search(r'equipment|cargo|pump|gear|hoist\s+equipment', x.text, re.IGNORECASE))
    return PIW if (has_hoisted and has_person and not has_equipment) else ABSTAIN


# ── NOT_PIW LFs ────────────────────────────────────────────────────────────────

@labeling_function()
def lf_false_alert(x):
    """Explicitly closed as false alert."""
    pattern = r'false\s+alert'
    return NOT_PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_not_in_distress(x):
    """Subject confirmed not in distress."""
    pattern = r'not\s+in\s+distress'
    return NOT_PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_made_it_to_shore(x):
    """Subject made it to shore with no water entry — abstain if PIW signals present."""
    has_shore = bool(re.search(r'made\s+it\s+(safely\s+)?to\s+shore', x.text, re.IGNORECASE))
    has_piw   = bool(re.search(r'\bPIW\b|in\s+the\s+water|overboard|capsized|fell', x.text, re.IGNORECASE))
    return NOT_PIW if (has_shore and not has_piw) else ABSTAIN


@labeling_function()
def lf_no_further_assistance(x):
    """No further assistance required."""
    pattern = r'no\s+further\s+(coast\s+guard\s+)?assistance'
    return NOT_PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_refused_medical(x):
    """Subject refused medical care — self-resolved."""
    pattern = r'refused\s+medical'
    return NOT_PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_cardiac_arrest_ashore(x):
    """Cardiac arrest ashore — not a PIW event."""
    pattern = r'cardiac\s+arrest\s+ashore'
    return NOT_PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_do_not_need_assistance(x):
    """Subject stated they do not need assistance."""
    pattern = r'do\s+not\s+need\s+any\s+assistance'
    return NOT_PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


# ── Incident Sub Type LFs ─────────────────────────────────────────────────────

@labeling_function()
def lf_subtype_piw(x):
    """MISLE Sub Type explicitly 'Person in Water (PIW)'."""
    return PIW if x["Incident Sub Type"] == "Person in Water (PIW)" else ABSTAIN


@labeling_function()
def lf_subtype_capsized(x):
    """Capsized vessel sub type with persons-in-water language — sub type alone is unreliable (0.441 acc)."""
    if x["Incident Sub Type"] != "Capsized Vessel":
        return ABSTAIN
    has_piw = bool(re.search(r'\bPIW\b|in\s+the\s+water|overboard|ejected|persons?\s+aboard', x.text, re.IGNORECASE))
    return PIW if has_piw else ABSTAIN


@labeling_function()
def lf_subtype_bridge_jumper(x):
    """Bridge jumper — person entered water intentionally."""
    return PIW if x["Incident Sub Type"] == "Bridge Jumper" else ABSTAIN


@labeling_function()
def lf_subtype_not_piw(x):
    """Sub Types that are vessel/equipment issues — not a PIW event."""
    not_piw_subtypes = {
        "Disabled Vessel", "Aground", "Adrift (Unmanned)", "Beset by Weather",
        "Disoriented Vessel", "Fire", "Flooding", "Abandoned/Derelict",
        "Stranded (on island)", "Collision", "Unreported Vessel",
    }
    return NOT_PIW if x["Incident Sub Type"] in not_piw_subtypes else ABSTAIN


# ── Additional Keyword LFs ─────────────────────────────────────────────────────

@labeling_function()
def lf_man_overboard(x):
    """'Man overboard' or MOB — explicit PIW terminology."""
    pattern = r'man\s+overboard|\bMOB\b'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_drowning(x):
    """Drowning or drowned — PIW outcome indicator."""
    pattern = r'\bdrown(ing|ed|s)?\b'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_rescue_swimmer(x):
    """Rescue swimmer deployed with explicit PIW/water context — excludes pure MEDEVAC deployments."""
    has_rs    = bool(re.search(r'rescue\s+swimmer', x.text, re.IGNORECASE))
    has_water = bool(re.search(r'\bPIW\b|in\s+the\s+water|overboard|capsized|swimmer\s+in', x.text, re.IGNORECASE))
    return PIW if (has_rs and has_water) else ABSTAIN


@labeling_function()
def lf_hypothermia(x):
    """Hypothermia — strong indicator of water immersion."""
    pattern = r'\bhypothermia\b'
    return PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_towing(x):
    """Towing a vessel — vessel assist, not a PIW event."""
    pattern = r'\btow(ing|ed)?\b'
    return NOT_PIW if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_short_narrative(x):
    """Narratives under 300 chars are NOT_PIW 96.3% of the time (gold label analysis)."""
    return NOT_PIW if len(x.text) < 300 else ABSTAIN


# ── BERTopic Topic LFs ────────────────────────────────────────────────────────
# Topic assignments from bertopic_topics.npy (aligned to same filtered df).
# PIW rates from topic_piw_enrichment.csv:
#   Topic 6  → 69.8% PIW  (n=53)  -> strong PIW signal
#   Topics with 0% PIW rate and n≥5 -> strong NOT_PIW signal

@labeling_function()
def lf_topic_high_piw(x):
    """BERTopic topic 6 has 69.8% PIW rate — treat as PIW."""
    return PIW if x.topic == 6 else ABSTAIN


@labeling_function()
def lf_topic_zero_piw(x):
    """Topics with 0% PIW rate (n≥5) — treat as NOT_PIW."""
    zero_piw_topics = {3, 4, 5, 7, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23}
    return NOT_PIW if x.topic in zero_piw_topics else ABSTAIN


# ── ABSTAIN LFs (ambiguous signals) ───────────────────────────────────────────

@labeling_function()
def lf_overdue_vessel(x):
    """Overdue vessel report — PIW not yet confirmed."""
    pattern = r'overdue\s+vessel|overdue\s+\d+\s*ft'
    return ABSTAIN if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_umib_only(x):
    """UMIB issued but no other PIW indicators — ambiguous."""
    has_umib = bool(re.search(r'\bUMIB\b', x.text, re.IGNORECASE))
    has_piw  = bool(re.search(r'\bPIW\b|in\s+the\s+water|capsized|overturned|ejected|fell\s+overboard', x.text, re.IGNORECASE))
    return ABSTAIN if (has_umib and not has_piw) else ABSTAIN


@labeling_function()
def lf_negative_search(x):
    """Search yielded negative results — PIW suspected but unconfirmed."""
    pattern = r'negative\s+results|searches?\s+(were\s+)?negative|negres'
    return ABSTAIN if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_missing_person(x):
    """Missing person — could be PIW or not yet found."""
    pattern = r'\bmissing\b'
    return ABSTAIN if re.search(pattern, x.text, re.IGNORECASE) else ABSTAIN


@labeling_function()
def lf_distress_call_only(x):
    """Distress call received but no confirmation of PIW."""
    has_distress = bool(re.search(r'distress\s+call', x.text, re.IGNORECASE))
    has_piw      = bool(re.search(r'\bPIW\b|in\s+the\s+water|capsized|overturned', x.text, re.IGNORECASE))
    return ABSTAIN if (has_distress and not has_piw) else ABSTAIN


# ── Apply with Snorkel LabelModel ─────────────────────────────────────────────
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel

# Load narratives from source JSON; rename to 'text' so LFs can access x.text
df = pd.read_json(
    r"C:\Users\CSJohnson\OneDrive - Coast Guard Academy\Academic\Spring 2c year\Project AI\Narratives_With_Labels_2Feb2026_fixed.json",
    orient="records"
)
df["text"] = df["SITREP Narrative"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df["text"].str.len() > 20].copy().reset_index(drop=True)

# Attach BERTopic topic assignments (produced by BERTOPIC_misle_model.py, same filter)
df["topic"] = np.load("bertopic_topics.npy", allow_pickle=True)

lfs = [
    # PIW keyword LFs (acc >= 0.7 or very low FP count)
    lf_piw_keyword, lf_person_in_water, lf_fell_overboard,
    lf_went_in_water, lf_cpr, lf_abandoned_vessel, lf_washed_up,
    lf_persons_in_water_with_pfds, lf_rescue_swimmer, lf_hypothermia,
    # NOT_PIW keyword LFs
    lf_false_alert, lf_not_in_distress, lf_no_further_assistance, lf_towing, lf_short_narrative,
    # BERTopic topic LFs
    lf_topic_high_piw, lf_topic_zero_piw,
    # Incident Sub Type LFs
    lf_subtype_piw, lf_subtype_capsized, lf_subtype_not_piw,
]

# Build label matrix (rows = documents, cols = LFs; values in {-1, 0, 1})
applier = PandasLFApplier(lfs=lfs)
L = applier.apply(df)

# ── LF Analysis against gold labels ───────────────────────────────────────────
from snorkel.labeling import LFAnalysis

gold_mask = df["PIW_Label"].notna()
L_gold    = L[gold_mask.values]
Y_gold    = df.loc[gold_mask, "PIW_Label"].values.astype(int)

print("\n-- LF Analysis (evaluated against gold labels) --")
summary = LFAnalysis(L=L_gold, lfs=lfs).lf_summary(Y=Y_gold)
with pd.option_context("display.max_rows", None, "display.width", 120, "display.float_format", "{:.3f}".format):
    print(summary.to_string())

# Fit generative label model (cardinality=2: NOT_PIW and PIW)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L, n_epochs=500, lr=0.001, seed=42)

# Predict hard labels and soft probabilities
df["weak_label"] = label_model.predict(L, tie_break_policy="abstain")
df["weak_prob_piw"] = label_model.predict_proba(L)[:, PIW]

# Save results
df.to_parquet("weak_labels.parquet", index=False)
print(f"Saved weak_labels.parquet — {(df['weak_label'] == PIW).sum()} PIW, "
      f"{(df['weak_label'] == NOT_PIW).sum()} NOT_PIW, "
      f"{(df['weak_label'] == ABSTAIN).sum()} ABSTAIN")