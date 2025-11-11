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
df = pd.read_csv('data/Amazon_Musical.csv')

# Sample 1% of the dataset for computational efficiency
df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
print(f"Working with {len(df)} samples (1% of the original dataset)")

# %% colab={"base_uri": "https://localhost:8080/"} id="oelDbiLfGgvb" outputId="a714a823-9f6f-4171-c9be-cda09bb05f0a"
# Make sure to use the entire dataset for your analysis
# Please use the HPC for running this code
df.shape

# %% id="tYNxGKIi6tpv"
# Load the English NLP model from spaCy
# This model provides tokenization, POS tagging, and named entity recognition
nlp = spacy.load("en_core_web_md")

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
# Use spaCy's built-in stop words
from spacy.lang.en.stop_words import STOP_WORDS

# Extract and clean lemma tokens from the spaCy results
# Keep only alphabetic lemmas, remove stop words, and convert them to lowercase
df["lemmas"] = df["spacy_tokens"].apply(
    lambda rows: [
        lemma.lower().strip()
        for (_, _, _, lemma) in rows
        if lemma and lemma.strip() and lemma.isalpha() and lemma.lower() not in STOP_WORDS
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
# Q1.1: Build the TF-IDF vectorizer (0.5 pt)
print("Building TF-IDF vectorizer with up to trigrams...")

vec_list_trigram = TfidfVectorizer(
    analyzer="word",
    lowercase=False,
    ngram_range=(1, 3),
    sublinear_tf=True
)

print("Fitting and transforming the text data...")
X_list_123g = vec_list_trigram.fit_transform(df["lemmas_text"])

print(f"TF-IDF matrix shape: {X_list_123g.shape}")
print(f"Number of documents: {X_list_123g.shape[0]}")
print(f"Number of features (terms): {X_list_123g.shape[1]}")

# %%
# Q1.2: Display top trigrams (0.5 pt)
print("\nTop 10 terms with highest TF-IDF scores:")
top_trigrams = get_top_terms(X_list_123g, vec_list_trigram, top_n=10)
print(top_trigrams)

# %%
# Save preprocessed data and TF-IDF results for quick restart
import pickle
from scipy.sparse import save_npz

print("Saving preprocessed data...")

# Save the dataframe with all processed columns
df.to_pickle('data/preprocessed_df.pkl')
print("✓ Saved preprocessed DataFrame to 'data/preprocessed_df.pkl'")

# Save the TF-IDF matrix (sparse matrix)
save_npz('data/X_list_123g.npz', X_list_123g)
print("✓ Saved TF-IDF matrix to 'data/X_list_123g.npz'")

# Save the vectorizer
with open('data/vec_list_trigram.pkl', 'wb') as f:
    pickle.dump(vec_list_trigram, f)
print("✓ Saved vectorizer to 'data/vec_list_trigram.pkl'")

print("\n✓ All preprocessing results saved successfully!")
print("  You can now start from Q2 by loading these files.")

# %%
# Load preprocessed data (use this cell to skip preprocessing and start from Q2)
# Uncomment the lines below when you want to load instead of preprocessing
import pickle
from scipy.sparse import load_npz

print("Loading preprocessed data...")

# Load the dataframe
df = pd.read_pickle('data/preprocessed_df.pkl')
print(f"✓ Loaded DataFrame with {len(df)} samples")

# Load the TF-IDF matrix
X_list_123g = load_npz('data/X_list_123g.npz')
print(f"✓ Loaded TF-IDF matrix with shape {X_list_123g.shape}")

# Load the vectorizer
with open('data/vec_list_trigram.pkl', 'rb') as f:
    vec_list_trigram = pickle.load(f)
print(f"✓ Loaded vectorizer with {len(vec_list_trigram.get_feature_names_out())} features")

print("\n✓ All data loaded successfully! Ready to continue from Q2.")

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
# K = 10          
# BATCH = 512     
# RANDOM_SEED = 1

# nmf = MiniBatchNMF(
#     n_components=K,
#     init="nndsvda",
#     random_state=RANDOM_SEED,
#     max_iter=300,
#     batch_size=BATCH,
# )

# W = nmf.fit_transform(X)
# H = nmf.components_

# print("W shape:", W.shape)
# print("H shape:", H.shape)

# TOP_N = 10
# for k in range(K):
#     top_idx = H[k].argsort()[-TOP_N:][::-1]
#     top_words = [vocab[i] for i in top_idx]
#     print(f"Topic {k}: {', '.join(top_words)}")

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

# %%
# Q2: Very coarse topics (K = 2 baseline)
print("=" * 60)
print("Q2: Training MiniBatchNMF with K=2 topics")
print("=" * 60)

# Set up aliases (as shown in the template)
X = X_list_123g
vocab = vec_list_trigram.get_feature_names_out()

# Model parameters
K = 2
BATCH = 512
RANDOM_SEED = 42

# Initialize and fit MiniBatchNMF
nmf = MiniBatchNMF(
    n_components=K,
    init="nndsvda",
    random_state=RANDOM_SEED,
    max_iter=300,
    batch_size=BATCH,
)

print(f"\nFitting MiniBatchNMF with {K} topics...")
W = nmf.fit_transform(X)
H = nmf.components_

# Print shapes
print("\nMatrix shapes:")
print(f"W shape: {W.shape}")
print(f"H shape: {H.shape}")

# Print top 10 terms for each topic
print("\nTop 10 terms for each topic:")
print("-" * 60)
TOP_N = 10
for k in range(K):
    top_idx = H[k].argsort()[-TOP_N:][::-1]
    top_words = [vocab[i] for i in top_idx]
    print(f"Topic {k}: {', '.join(top_words)}")

# %% [markdown]
# ### Q2: Topic Names
#
# Based on the top keywords, I would name the topics as follows:
#
# - **Topic 0**: The product works well.
# - **Topic 1**: The product has good price and quality.

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

# %%
# Q3: Increase topic count (K = 4)
print("=" * 60)
print("Q3: Training MiniBatchNMF with K=4 topics")
print("=" * 60)

# Set up aliases (as shown in the template)
X = X_list_123g
vocab = vec_list_trigram.get_feature_names_out()

# Model parameters - ONLY change K from 2 to 4
K = 4
BATCH = 512
RANDOM_SEED = 42

# Initialize and fit MiniBatchNMF
nmf = MiniBatchNMF(
    n_components=K,
    init="nndsvda",
    random_state=RANDOM_SEED,
    max_iter=300,
    batch_size=BATCH,
)

print(f"\nFitting MiniBatchNMF with {K} topics...")
W = nmf.fit_transform(X)
H = nmf.components_

# Print shapes
print("\nMatrix shapes:")
print(f"W shape: {W.shape}")
print(f"H shape: {H.shape}")

# Print top 10 terms for each topic
print("\nTop 10 terms for each topic:")
print("-" * 60)
TOP_N = 10
for k in range(K):
    top_idx = H[k].argsort()[-TOP_N:][::-1]
    top_words = [vocab[i] for i in top_idx]
    print(f"Topic {k}: {', '.join(top_words)}")

# %% [markdown]
# ### Q3: Topic Names
#
# Based on the top keywords, I would name the topics as follows:
#
# - **Topic 0**: Great Value Products - Products that work great and offer good price
# - **Topic 1**: Quality Strings and Accessories - Good quality products with focus on instrument strings
# - **Topic 2**: Products Working as Advertised - Items that work perfectly and meet expectations
# - **Topic 3**: Positive Musical Instrument Experiences - Love and excellent experiences with guitars and instruments

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

# %%
# Q4: Same K=4 but different initialization
print("=" * 60)
print("Q4: Training MiniBatchNMF with K=4 topics (random initialization)")
print("=" * 60)

# Set up aliases (as shown in the template)
X = X_list_123g
vocab = vec_list_trigram.get_feature_names_out()

# Model parameters - ONLY change init from "nndsvda" to "random"
K = 4
BATCH = 512
RANDOM_SEED = 42

# Initialize and fit MiniBatchNMF
nmf = MiniBatchNMF(
    n_components=K,
    init="random",  # CHANGED from "nndsvda"
    random_state=RANDOM_SEED,
    max_iter=300,
    batch_size=BATCH,
)

print(f"\nFitting MiniBatchNMF with {K} topics (random init)...")
W = nmf.fit_transform(X)
H = nmf.components_

# Print shapes
print("\nMatrix shapes:")
print(f"W shape: {W.shape}")
print(f"H shape: {H.shape}")

# Print top 10 terms for each topic
print("\nTop 10 terms for each topic:")
print("-" * 60)
TOP_N = 10
for k in range(K):
    top_idx = H[k].argsort()[-TOP_N:][::-1]
    top_words = [vocab[i] for i in top_idx]
    print(f"Topic {k}: {', '.join(top_words)}")

# %% [markdown]
# ### Q4: Topic Names
#
# Based on the top keywords, I would name the topics as follows:
#
# - **Topic 0**: Incoherent/Mixed - Random phrases (heavy string, audio enthusiast, box surprised loud, etc.)
# - **Topic 1**: Incoherent/Mixed - Scattered terms (use, depend arrange, new squier, overdrive, etc.)
# - **Topic 2**: Incoherent/Mixed - Unrelated phrases (turn volume, learn mix, laugh function, etc.)
# - **Topic 3**: Somewhat General Use - Contains "use" and "great" but mixed with odd phrases
#
# **Note**: Random initialization produced very incoherent topics with multi-word phrases stuck together, making interpretation difficult.

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

# %%
# Q5: Same K=4, add early stopping (convergence tolerance)
print("=" * 60)
print("Q5: Training MiniBatchNMF with K=4 topics (with early stopping)")
print("=" * 60)

# Set up aliases (as shown in the template)
X = X_list_123g
vocab = vec_list_trigram.get_feature_names_out()

# Model parameters - Same as Q4 but ADD tol parameter
K = 4
BATCH = 512
RANDOM_SEED = 42

# Initialize and fit MiniBatchNMF
nmf = MiniBatchNMF(
    n_components=K,
    init="random",
    random_state=RANDOM_SEED,
    max_iter=300,
    batch_size=BATCH,
    tol=1e-3,  # ADDED: early stopping with looser tolerance
)

print(f"\nFitting MiniBatchNMF with {K} topics (tol=1e-3)...")
W = nmf.fit_transform(X)
H = nmf.components_

# Print shapes
print("\nMatrix shapes:")
print(f"W shape: {W.shape}")
print(f"H shape: {H.shape}")

# Print convergence info
print(f"Number of iterations: {nmf.n_iter_}")

# Print top 10 terms for each topic
print("\nTop 10 terms for each topic:")
print("-" * 60)
TOP_N = 10
for k in range(K):
    top_idx = H[k].argsort()[-TOP_N:][::-1]
    top_words = [vocab[i] for i in top_idx]
    print(f"Topic {k}: {', '.join(top_words)}")

# %% [markdown]
# ### Q5: Topic Names
#
# Based on the top keywords, I would name the topics as follows:
#
# - **Topic 0**: Incoherent/Mixed - Random phrases (heavy string, audio enthusiast, box surprised loud, etc.)
# - **Topic 1**: Incoherent/Mixed - Scattered terms (depend arrange, new squier, overdrive, etc.)
# - **Topic 2**: Incoherent/Mixed - Unrelated phrases (turn volume, learn mix, sound tone recording, etc.)
# - **Topic 3**: Somewhat Repair/Guitar Related - Mix of phrases including "guitar grandson", "great repair"
#
# **Note**: Model converged too quickly (only 1 iteration) due to loose tolerance (tol=1e-3), resulting in poorly formed topics similar to Q4.

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

# %%
# Q6: Same K=4, change batch size & iterations
print("=" * 60)
print("Q6: Training MiniBatchNMF with K=4 topics (smaller batch, more iterations)")
print("=" * 60)

# Set up aliases (as shown in the template)
X = X_list_123g
vocab = vec_list_trigram.get_feature_names_out()

# Model parameters - Change batch_size and max_iter, init back to "nndsvda"
K = 4
BATCH = 128  # CHANGED from 512 to 128
RANDOM_SEED = 42

# Initialize and fit MiniBatchNMF
nmf = MiniBatchNMF(
    n_components=K,
    init="nndsvda",  # Back to "nndsvda"
    random_state=RANDOM_SEED,
    max_iter=350,  # CHANGED from 300 to 350
    batch_size=BATCH,
)

print(f"\nFitting MiniBatchNMF with {K} topics (batch_size={BATCH}, max_iter=350)...")
W = nmf.fit_transform(X)
H = nmf.components_

# Print shapes
print("\nMatrix shapes:")
print(f"W shape: {W.shape}")
print(f"H shape: {H.shape}")

# Print convergence info
print(f"Number of iterations: {nmf.n_iter_}")

# Print top 10 terms for each topic
print("\nTop 10 terms for each topic:")
print("-" * 60)
TOP_N = 10
for k in range(K):
    top_idx = H[k].argsort()[-TOP_N:][::-1]
    top_words = [vocab[i] for i in top_idx]
    print(f"Topic {k}: {', '.join(top_words)}")

# %% [markdown]
# ### Q6: Topic Names
#
# Based on the top keywords, I would name the topics as follows:
#
# - **Topic 0**: General Positive Reviews - Products that work great, good quality, and loved by users
# - **Topic 1**: Quality and Value Focus - Good price, quality products, especially strings and accessories
# - **Topic 2**: Functional Performance - Products working perfectly, fine, and as advertised
# - **Topic 3**: Musical Instrument Satisfaction - Love, perfect sound, excellent guitar experiences
#
# **Note**: Despite only 1 iteration, nndsvda initialization produced coherent, interpretable topics similar to Q3.

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
