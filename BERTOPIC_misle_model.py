
#%%
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from umap import UMAP
import hdbscan
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import re
import os
import spacy



#%%
misle_df = pd.read_json(
    r"C:\Users\CSJohnson\OneDrive - Coast Guard Academy\Academic\Spring 2c year\Project AI\Narratives_With_Labels_2Feb2026_fixed.json",
    orient="records"
)
misle_df['Incident Sub Type'].info()

#%%

# Clean text
misle_df["doc"] = (
    misle_df["SITREP Narrative"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Keep non-empty docs
misle_df = misle_df[misle_df["doc"].str.len() > 20].copy()

# Labeled subset for evaluation only
labeled_df = misle_df[misle_df["PIW_Label"].notna()].copy()
labeled_df["PIW_Label"] = labeled_df["PIW_Label"].astype(int)


#Checks to see if 
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CLEANED_PATH    = os.path.join(HERE, "misle_cleaned.parquet")
if os.path.exists(CLEANED_PATH):
    print("Loading cached cleaned documents...")
    misle_df["doc_clean"] = pd.read_parquet(CLEANED_PATH)["doc_clean"].values
else:
    print("Cleaning documents with spaCy")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

    def preprocess(t):
        t = t.lower()
        t = re.sub(r'\d+', ' ', t)
        t = re.sub(r'\b(sector|station|sector det|base)\b.*?\b', ' ', t)
        return t

    def tokens_from_doc(doc):
        return " ".join(
            token.text for token in doc
            if token.ent_type_ not in ["GPE", "LOC"] and not token.is_punct
        )

    preprocessed = [preprocess(t) for t in misle_df["doc"]]
    misle_df["doc_clean"] = [
        tokens_from_doc(doc)
        for doc in nlp.pipe(preprocessed, batch_size=256, n_process=1)
    ]
    misle_df[["doc_clean"]].to_parquet(CLEANED_PATH)
    print(f"Cleaned docs saved to '{CLEANED_PATH}'")

documents_all = misle_df["doc_clean"].tolist()

print("Total docs:", len(misle_df))
print("Labeled docs:", len(labeled_df))
print("PIW rate among labeled:", labeled_df["PIW_Label"].mean())



#%%

domain_stopwords = [
    "cg", "uscg", "unit", "sector", "station", "case", "sar", "umib",
    "advised", "requested", "conducted", "response", "mrcc", "opcon",
    "onscene", "assets", "crew", "pob",
    "beach", "inlet", "city", "newport", "castle", "hill",
    "virginia", "va", "ri", "nj", "sc",
    "charleston", "georgetown", "barnegat"
]

stop_words = text.ENGLISH_STOP_WORDS.union(domain_stopwords)

vectorizer_model = CountVectorizer(
    stop_words=list(stop_words),
    ngram_range=(1, 2),
    min_df=10
)

embedding_model = SentenceTransformer("all-mpnet-base-v2")

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=25,
    min_samples=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)

# Seed topics guide BERTopic toward PIW-relevant structure without full supervision
seed_topic_list = [
    ["piw", "person in water", "overboard", "capsized", "ejected", "hoisted", "fell overboard"],
    ["false alert", "not in distress", "no assistance", "refused medical"],
    ["vessel aground", "flooding", "taking on water", "sinking"],
    ["medical", "cpr", "unconscious", "unresponsive", "trauma"],
    ["missing person", "overdue vessel", "umib", "negative search"],
]

representation_model = MaximalMarginalRelevance(diversity=0.3)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    seed_topic_list=seed_topic_list,
    nr_topics="auto",
    verbose=True
)


MODEL_PATH = os.path.join(HERE, "bertopic_model")
TOPICS_PATH = os.path.join(HERE, "bertopic_topics.npy")
PROBS_PATH = os.path.join(HERE, "bertopic_probs.npy")
EMBEDDINGS_PATH = os.path.join(HERE, "embeddings.npy")

if os.path.exists(MODEL_PATH) and os.path.exists(TOPICS_PATH):
    print("Loading saved BERTopic model and embeddings...")
    topic_model = BERTopic.load(MODEL_PATH, embedding_model=embedding_model)
    topics_all  = np.load(TOPICS_PATH, allow_pickle=True).tolist()
    probs_all   = np.load(PROBS_PATH,  allow_pickle=True)
    embeddings  = np.load(EMBEDDINGS_PATH)
else:
    print("Computing embeddings...")
    embeddings = embedding_model.encode(documents_all, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)

    print("Fitting BERTopic model")
    topics_all, probs_all = topic_model.fit_transform(documents_all, embeddings=embeddings)
    topic_model.save(MODEL_PATH, serialization="pickle", save_embedding_model=False)
    np.save(TOPICS_PATH, np.array(topics_all, dtype=object))
    np.save(PROBS_PATH,  np.array(probs_all,  dtype=object))
    print(f"Model saved to '{MODEL_PATH}'")

misle_df["Topic"] = topics_all

print(topic_model.get_topic_info())



#%%

eval_df = misle_df.loc[labeled_df.index, ["doc", "Topic"]].copy()
eval_df["PIW_Label"] = labeled_df["PIW_Label"]

# Ignore outliers (topic == -1) for enrichment analysis
eval_df = eval_df[eval_df["Topic"] != -1].copy()

# PIW rate by topic + counts
topic_stats = (
    eval_df.groupby("Topic")
    .agg(
        n=("PIW_Label", "size"),
        piw_rate=("PIW_Label", "mean")
    )
    .sort_values(["piw_rate", "n"], ascending=[False, False])
)

# Save enrichment table for use in LabelingFunctions.py
topic_stats.to_csv(os.path.join(HERE, "topic_piw_enrichment.csv"))
print("Topic PIW enrichment saved to 'topic_piw_enrichment.csv'")

# Show top words for the most PIW-enriched topics
top_topics = topic_stats.head(5).index.tolist()
for t in top_topics:
    print("\nTopic", t, "piw_rate=", round(topic_stats.loc[t, "piw_rate"], 3), "n=", topic_stats.loc[t, "n"])
    print(topic_model.get_topic(t)[:15])

# Optional visuals
topic_model.visualize_topics()
topic_model.visualize_hierarchy()
