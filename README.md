# PIW-MISLE-analysis
This project takes raw Coast Guard after-action Situation Reports (SITREPS) and builds a Person In the Water (PIW) Binary classifier. 


First File: BERTOPIC_misle_model.py
This file is used to provide insight for the BERTOPIC labeling functions.
Input

•	Narratives_With_Labels_2Feb2026_fixed.json: The raw dataset with 10,000 + MISLE incident narratives
Output

•	misle_cleaned.parquet: extra spaces, numbers, and location names are stripped from the narratives – this was necessary because before this, BERTOPIC kept clustering based on geographic location, not PIW cases. This was only used for the Topic Modeling, not the supervised learning. There was no preprocessing done on the data that was put into DistilBERT.

•	bertopic_model/: this is the trained bertopic model

•	bertopic_topics.npy: contains the topic assignment for every document

•	topic_piw_enrichment.csv: using only narratives with human labels, this csv file has every topic and the percentage of that topic that is positive PIW 



Second File: LabelingFunctions.py
This file takes the BERTOPIC data and creates Learning Functions so that Snorkel can output weak_labels for every SITREP.
Input

•	Narratives_With_Labels_2Feb2026_fixed.json: The raw dataset with 10,000+ MISLE incident narratives

•	bertopic_topics.npy: Topic assignment for every document (produced by BERTOPIC_misle_model.py)

Output

•	weak_labels.parquet: The full dataset with two new columns: weak_label (Snorkel LabelModel's hard prediction: PIW, NOT_PIW, or ABSTAIN) and weak_prob_piw (soft probability of PIW). This is the training signal passed to DistilBERT since most narratives have no human label. The file also prints an LF analysis table showing coverage, accuracy, and conflict rates for each labeling function evaluated against the gold labels.

Third File: classifier.py
This file is multiple classical machine learning models
Input

•	embeddings.npy: DistilBERT embeddings for every narrative

•	weak_labels.parquet: The dataset with Snorkel weak labels, soft PIW probabilities, weather, and text

Output

•	Console output only, no files saved. Trains and evaluates six models side by side, each printing a classification report and threshold sweep table against the gold-labeled test set: (1) Logistic Regression on weak labels, (2) Random Forest on weak labels. (3) LightGBM with hard weak labels, (4) LightGBM with soft sample weights from weak_prob_piw, (5) Logistic Regression on gold labels only (upper bound reference), and (6) LightGBM on gold labels only. Features combine sentence embeddings, six weather columns, and narrative character length.



Fourth File: transformer_classifier.py
This file trains the DistilBERT model on the uncleaned SITREPS and their weak labels
Input

•	weak_labels.parquet: The uncleaned SITREPS with Snorkel-generated weak labels and PIW probabilities
Output

•	distilbert_piw_model/: A DistilBERT sequence classifier trained on the entire non-gold portion of the dataset using soft weak-label probabilities as targets (Stage 1 of two-stage training). Also prints a threshold sweep table and classification report evaluated against the gold-labeled documents.



Fifth File: transformer_finetune.py
This file finetunes the Stage 1 model using the weak labels and human gold labels.
Input

•	weak_labels.parquet: The dataset with Snorkel-generated weak labels and human gold labels

•	distilbert_piw_model/: The Stage 1 model pretrained on weak labels
Output

•	distilbert_finetuned_model/: The best-checkpoint DistilBERT model fine-tuned on the 80% gold-labeled training split (Stage 2). The best checkpoint is selected by highest F1 score across 10 epochs. Also prints a threshold sweep table and classification report evaluated against the 20% gold held-out test split.

Sixth File: predict.py 
Input

•	weak_labels.parquet: Used to reconstruct the same 20% gold held-out test split (via the same random seed as transformer_finetune.py)

•	distilbert_finetuned_model/: The final fine-tuned model

Output
•	Console output showing 5 random PIW and 5 random NOT_PIW example predictions with true vs. predicted labels, plus a full confusion matrix, accuracy, precision, recall, and F1 on the complete held-out gold set. A threshold sweep plot (threshold_curve.png) is also generated
