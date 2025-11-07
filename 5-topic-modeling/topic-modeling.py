# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="25577984"
# # TF-IDF & Sentiment Analysis & Topic Modeling — 9‑Point Homework
#   
# **Dataset:** `Amazon Musical.csv`  
# Name: Ruihuang Yang  
# NetID: rxy216  
# Date: 2025-11-07  

# %% [markdown] id="1649abe8"
# ## 0. Set up & Data import

# %% id="35ad9dfa"
# Load basic libraries
# Do NOT import these libraries again below
# Re-importing (writing inefficient code) will result in deductions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy

# %% id="de31d834"
# Load the dataset
# Read the CSV file named 'Amazon Musical.csv' into a pandas DataFrame called df
df = pd.read_csv('Amazon Musical.csv')

# %% colab={"base_uri": "https://localhost:8080/"} id="oelDbiLfGgvb" outputId="a714a823-9f6f-4171-c9be-cda09bb05f0a"
# Make sure to use the entire dataset for your analysis
# Please use the HPC for running this code
df.shape

# %% id="tYNxGKIi6tpv"
# Load the English NLP model from spaCy
# This model provides tokenization, POS tagging, and named entity recognition
nlp = spacy.load("en_core_web_sm")

# %% id="IqPxz_P56uzX"
# Define a function to process text data using spaCy with parallel processing
# It extracts token, POS, tag, and lemma information for each review
# Do NOT modify this function or its parameters, use it exactly as provided

from tqdm import tqdm

def spacy_analyze_pipe(texts):
    results = []
    for doc in tqdm(nlp.pipe(texts, batch_size=128, n_process=4), 
                    total=len(texts), 
                    desc="spaCy NLP processing"):
        tokens = [(token.text, token.pos_, token.tag_, token.lemma_) for token in doc]
        results.append(tokens)
    return results

df["spacy_tokens"] = spacy_analyze_pipe(df["review_body"].astype(str).tolist())

# %% id="e2H2kzst6xH_"
# Display the first two rows to check the original text and its spaCy token results
print(df[["review_body", "spacy_tokens"]].head(2))

# %% id="Z09c_jcf622b"
# Import TfidfVectorizer for converting text data into numerical feature vectors
from sklearn.feature_extraction.text import TfidfVectorizer

# %% id="_E2O5DrG63NH"
# Extract and clean lemma tokens from the spaCy results
# Keep only alphabetic lemmas and convert them to lowercase
df["lemmas"] = df["spacy_tokens"].apply(
    lambda rows: [
        lemma.lower().strip()
        for (_, _, _, lemma) in rows
        if lemma and lemma.strip() and lemma.isalpha()
    ]
)


# %%
# Sum TF-IDF scores across all documents
# Combine terms and their total TF-IDF scores into a DataFrame
# Sort in descending order and return the top N terms
# Do NOT modify this function — use it exactly as provided below    
def get_top_terms(X, vectorizer, top_n=10):

    # Sum TF-IDF scores across all documents
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    # Combine into DataFrame and sort descending
    df_terms = pd.DataFrame({"term": terms, "score": sums})
    df_terms = df_terms.sort_values("score", ascending=False).head(top_n)

    return df_terms


# %%
# Combine lemma lists into plain text strings
# Each document’s tokens are joined into a single string (e.g., ["great", "movie"] → "great movie")
df["lemmas_text"] = df["lemmas"].apply(" ".join)

# %% [markdown] id="47e8c2d5"
# ## Q1 (1 pt) — TF-IDF with up to 3-grams
#
# **Tasks:**
#
# Students must create separate code cells for each task.
#
# 1. Build the TF-IDF vectorizer (0.5 pt): Using the combined text column (df["lemmas_text"]), create a TF-IDF vectorizer named **vec_list_trigram** that extracts unigrams, bigrams, and trigrams.  
#
# This step must follow the specifications below exactly:
#
# - Use df["lemmas_text"] as the input, not df["lemmas"].
# - The variable name must be exactly vec_list_trigram.
# - The TfidfVectorizer parameters must be: analyzer="word", lowercase=False, ngram_range=(1, 3), sublinear_tf=True
# - The code should print progress messages.
# - Do not re-import any libraries that have already been imported above.
# - Any inefficient, renamed, or altered implementation (e.g., different parameters, variable names) will result in a point deduction.
#
#
# 2. Display top trigrams (0.5 pt): After building the TF-IDF matrix, print the top 10 keywords with the highest TF-IDF scores. Use the helper function provided (get_top_terms) exactly.
#
# This step must follow the specifications below exactly:
#
# - Use the function get_top_terms exactly as provided earlier.
# - Assign the result to a variable named top_trigrams.

# %%

# %% [markdown]
# Below is a shared NMF skeleton code that we will use throughout the assignment.
# Please treat this as the base code and, for each question, only modify the parts that are explicitly requested in the instructions.

# %%
from scipy.stats import entropy
from sklearn.decomposition import MiniBatchNMF
from sklearn.preprocessing import normalize

# %%
# Aliases
X = X_list_123g
vocab = vec_list_trigram.get_feature_names_out()

# %%
#DO NOT RUN! This is an example code
K = 10          
BATCH = 512     
RANDOM_SEED = 1

nmf = MiniBatchNMF(
    n_components=K,
    init="nndsvda",
    random_state=RANDOM_SEED,
    max_iter=300,
    batch_size=BATCH,
)

W = nmf.fit_transform(X)
H = nmf.components_

print("W shape:", W.shape)
print("H shape:", H.shape)

TOP_N = 10
for k in range(K):
    top_idx = H[k].argsort()[-TOP_N:][::-1]
    top_words = [vocab[i] for i in top_idx]
    print(f"Topic {k}: {', '.join(top_words)}")

# %% [markdown]
# ## Q2 (1 pt) — Very coarse topics (K = 2 baseline)
#
# **Tasks:**
#
# In this question, you will start with a very coarse topic model.
#
# 1. Set up a MiniBatchNMF model with:
#
# - n_components = 2 (K = 2 topics)
#
# - init = "nndsvda"
#
# - random_state = 42
#
# - max_iter = 300
#
# - batch_size = 512
#
# - Fit the model on X (X_list_123g).
#
# 2. Print the shapes of W and H.
#
# 3. For each topic, print the top 10 terms using vocab = vec_list_trigram.get_feature_names_out().
#
# 4. (Markdown cell) For each topic, look at the keywords and create your own topic name
#
# Use the code template above and only change the values needed for this question.
#

# %% [markdown]
# ## Q3 (1 pt) — Increase topic count (K = 4)
#
# **Tasks:**
#
# Now we make the topic structure more fine-grained.
#
# 1. Set up a MiniBatchNMF model with:
#
# - Starting from your Q1 code, change the number of topics to K = 4 (n_components = 4).
#
# - Keep all other parameters the same.
#
# - Fit the model on X (X_list_123g).
#
# 2. Print the shapes of W and H.
#
# 3. For each topic, print the top 10 terms using vocab = vec_list_trigram.get_feature_names_out().
#
# 4. (Markdown cell) For each topic, look at the keywords and create your own topic name
#
# Use the code template above and only change the values needed for this question.

# %% [markdown]
# ## Q4 (1 pt) — Same K=4 but different initialization
#
# **Tasks:**
#
# In this question, we keep K = 4 topics but change how the factorization is initialized.
#
# 1. Set up a MiniBatchNMF model with:
#
# - Change the initialization method from "nndsvda" to "random".
#
# - Keep all other parameters the same.
#
# - Fit the model on X (X_list_123g).
#
# 2. Print the shapes of W and H.
#
# 3. For each topic, print the top 10 terms using vocab = vec_list_trigram.get_feature_names_out().
#
# 4. (Markdown cell) For each topic, look at the keywords and create your own topic name
#
# Use the code template above and only change the values needed for this question.

# %% [markdown]
# ## Q5 (1 pt) — Same K=4, add change convergence tolerance (early stopping)
#
# **Tasks:**
#
# Now we still use K = 4 topics, but we change convergence tolerance (early stopping).
#
# 1. Set up a MiniBatchNMF model with:
#
# - K = 4, init = "random", random_state = 42, max_iter = 300, batch_size = 512
#
# - Add the following parameters to MiniBatchNMF: tol = 1e-3 (make convergence a bit looser than default).
#
# - Fit the model on X (X_list_123g).
#
# 2. Print the shapes of W and H.
#
# 3. For each topic, print the top 10 terms using vocab = vec_list_trigram.get_feature_names_out().
#
# 4. (Markdown cell) For each topic, look at the keywords and create your own topic name
#
# Use the code template above and only change the values needed for this question.

# %% [markdown]
# ## Q6 (1 pt) — Same K=4, change batch size & iterations
#
# **Tasks:**
#
# Same K = 4, different mini-batch size and max_iter
#
# 1. Set up a MiniBatchNMF model with:
#
# - K = 4, init = "nndsvda", random_state = 42
#
# - Change: batch_size 128, max_iter 350
#
# - Fit the model on X (X_list_123g).
#
# 2. Print the shapes of W and H.
#
# 3. For each topic, print the top 10 terms using vocab = vec_list_trigram.get_feature_names_out().
#
# 4. (Markdown cell) For each topic, look at the keywords and create your own topic name
#
# Use the code template above and only change the values needed for this question.

# %% [markdown]
# ## Q7 (3 pt) — Same K=4, change batch size & iterations
#
# **Tasks:**
#
# Based on the printed top terms from each question (Q1–Q5), write a short reflection (1–2 paragraphs or bullet points) addressing:
#
# 1. Effect of the number of topics  (0.5 pt):
#
# - How do the K = 2 topics  differ from the K = 4 topics ?
#
# - Do the K = 2 topics look too broad or mixed?
#
# 2. Effect of initialization  (0.5 pt):
#
# - Compare the K = 4 topics from "nndsvda"  and "random" .
#
# - Which one looks more stable and coherent?
#
# 3. Effect of early stopping (0.5 pt):
#
# - Compared to the baseline model, do the topics with the new stopping criterion (tolerance) look more or less stable and interpretable?
#
# - Do you observe any trade-off between runtime and topic quality (for example, similar topics but faster, or slightly noisier topics but shorter training time)?
#
# 4. Effect of batch size and iterations (0.5 pt):
#
# - When you changed batch_size and max_iter , did the topics change noticeably?
#
# - Do you think smaller batches + more iterations made the model more stable, less stable, or similar?
#
# 5. Summarize which configuration (Q2–Q5) you would choose as your “final” topic model for this dataset and briefly justify your choice (1 pt).

# %% [markdown]
# #your answer
