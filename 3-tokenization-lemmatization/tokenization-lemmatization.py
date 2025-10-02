# %% [markdown]
# BUAI 435 Assignment 3 - Tokenization & Lemmatization  
# Name: Ruihuang Yang  
# NetID: rxy216  
# Date: 10/01/2025  

# %% [markdown]
# ### Import libraries

# %%
import pandas as pd
import numpy as np
import random
import re
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import spacy
import nltk
from nltk.corpus import stopwords

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# %% [markdown]
# ### Load and Explore Data

# %%
# Load the Amazon Musical dataset
df = pd.read_csv('data/Amazon_Musical.csv')
print(f"Dataset shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows of review_body:")
print(df['review_body'].head())

# %% [markdown]
# ## Q1 — Keras Tokenizer (1 point)
# 
# **Task:**
# - Import `text_to_word_sequence` from `tensorflow.keras.preprocessing.text`
# - Convert `review_body` column to string type using `.astype(str)`
# - Apply `text_to_word_sequence` to each row of `review_body`
# - Store result in new column `token_keras`
# - Preview first 5 rows

# %%

# Convert review_body to string type to avoid errors with non-string values
df['review_body'] = df['review_body'].astype(str)

# Apply text_to_word_sequence to each row of review_body
df['token_keras'] = df['review_body'].apply(text_to_word_sequence)

# Preview the results for the first 5 rows
print("="*80)
print("Q1 — Keras Tokenizer Results")
print("="*80)
for i in range(5):
    print(f"\n--- Row {i} ---")
    print(f"Original review_body:\n{df['review_body'].iloc[i]}")
    print(f"\nTokenized (token_keras):\n{df['token_keras'].iloc[i]}")
    print("-"*80)

# %% [markdown]
# ## Q2 — Regex Tokenizer Version 1 (1 point)
# 
# **Task:**
# - Import the built-in `re` module for regular expressions
# - Define a compiled regex pattern: `r"[A-Za-z]+"`
#   - Matches one or more English letters (A–Z or a–z)
#   - Numbers, punctuation, emojis, and symbols are excluded
#   - Hyphenated words (e.g., cost-effective) will be split (cost, effective)
#   - No lowercasing or other normalization performed here
# - Ensure `review_body` is treated as a string: `.astype(str)`
# - Apply `pattern.findall(x)` to each row of `review_body` with `.apply(...)`
# - Save the result to a new column named `token_regex_ver1`
# - Preview by printing `review_body` and `token_regex_ver1` for the first 5 rows

# %%
# Define compiled regex pattern to match one or more English letters
pattern = re.compile(r"[A-Za-z]+")

# Ensure review_body is treated as string
df['review_body'] = df['review_body'].astype(str)

# Apply pattern.findall(x) to each row of review_body
df['token_regex_ver1'] = df['review_body'].apply(lambda x: pattern.findall(x))

# Preview the results for the first 5 rows
print("="*80)
print("Q2 — Regex Tokenizer Version 1 Results")
print("="*80)
for i in range(5):
    print(f"\n--- Row {i} ---")
    print(f"Original review_body:\n{df['review_body'].iloc[i]}")
    print(f"\nTokenized (token_regex_ver1):\n{df['token_regex_ver1'].iloc[i]}")
    print("-"*80)

# %% [markdown]
# ## Q3 — Regex Tokenizer Version 2 (1 point)
# 
# **Task:**
# - Define a compiled regex pattern: `r"\w+"`, name the object `pattern2`
#   - Matches one or more "word characters" (letters A–Z/a–z, digits 0–9, underscore `_`)
#   - Punctuation, emojis, and symbols are excluded
#   - Hyphenated words will be split
#   - No lowercasing or normalization performed here
# - Ensure `review_body` is treated as a string: `.astype(str)`
# - Apply `pattern2.findall(x)` to each row of `review_body` with `.apply(...)`
# - Save the result to a new column named `token_regex_ver2`
# - Preview by printing `review_body` and `token_regex_ver2` for the first 5 rows

# %%
# Define compiled regex pattern to match one or more word characters
pattern2 = re.compile(r"\w+")

# Ensure review_body is treated as string
df['review_body'] = df['review_body'].astype(str)

# Apply pattern2.findall(x) to each row of review_body
df['token_regex_ver2'] = df['review_body'].apply(lambda x: pattern2.findall(x))

# Preview the results for the first 5 rows
print("="*80)
print("Q3 — Regex Tokenizer Version 2 Results")
print("="*80)
for i in range(5):
    print(f"\n--- Row {i} ---")
    print(f"Original review_body:\n{df['review_body'].iloc[i]}")
    print(f"\nTokenized (token_regex_ver2):\n{df['token_regex_ver2'].iloc[i]}")
    print("-"*80)

# %% [markdown]
# ## Q4 — Regex Tokenizer Version 3 & Removing Some Stop Words (2 points)
# 
# **Task:**
# - Define a stoplist as a Python set named **STOPLIST**:
#   `{"the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "br"}`
# - Define a regex pattern object named `pattern3`:
#   `pattern3 = re.compile(r"[A-Za-z_']+")`
#   - Matches sequences of letters, underscores, or apostrophes
#   - Allows simple handling of contractions like *don't*
#   - Numbers and punctuation (other than `'` and `_`) excluded
# - Ensure `review_body` is treated as string: `.astype(str)`
# - Convert `review_body` to lowercase: `.str.lower()`
# - Extract tokens with `pattern3.findall(x)`
# - Remove any tokens found in the stoplist (`if w not in STOPLIST`)
# - Save results to a new column: `token_regex_ver3`
# - Preview by printing both `review_body` and `token_regex_ver3` for the first 5 rows

# %%
# Define stoplist as a Python set
STOPLIST = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "br"}

# Define compiled regex pattern to match letters, underscores, or apostrophes
pattern3 = re.compile(r"[A-Za-z_']+")

# Ensure review_body is treated as string and convert to lowercase
df['review_body'] = df['review_body'].astype(str)

# Extract tokens and remove stopwords
df['token_regex_ver3'] = df['review_body'].str.lower().apply(
    lambda x: [w for w in pattern3.findall(x) if w not in STOPLIST]
)

# Preview the results for the first 5 rows
print("="*80)
print("Q4 — Regex Tokenizer Version 3 & Removing Stop Words Results")
print("="*80)
for i in range(5):
    print(f"\n--- Row {i} ---")
    print(f"Original review_body:\n{df['review_body'].iloc[i]}")
    print(f"\nTokenized (token_regex_ver3):\n{df['token_regex_ver3'].iloc[i]}")
    print("-"*80)

# %% [markdown]
# ## Q5 — Lemmatizer for Regex Version 2 (2 points)
# 
# **Task:**
# - Import required libraries: `spaCy` and `nltk`
# - Download the NLTK stopwords list: `nltk.download('stopwords')`
# - Load English stopwords from NLTK into a Python set `stop_words`
# - Load the spaCy English model: `en_core_web_sm` → variable `nlp`
# - Define function **`lemmatize_tokens`** that:
#   1. Takes a list of tokens as input
#   2. Joins them into a string for spaCy to process
#   3. Creates a spaCy `doc` object
#   4. Extracts each token's lemma (base form)
#   5. Converts lemma to lowercase
#   6. Keeps only alphabetic tokens (`token.is_alpha`)
#   7. Removes tokens found in `stop_words`
# - Apply `lemmatize_tokens` to the column `token_regex_ver2` to produce a new column `lemmas`
# - Preview by printing `review_body` and `lemmas` for the first 5 rows

# %%
# Download NLTK stopwords
nltk.download('stopwords')

# Load English stopwords from NLTK into a Python set
stop_words = set(stopwords.words('english'))

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Define lemmatize_tokens function
def lemmatize_tokens(tokens):
    """
    Takes a list of tokens as input and returns lemmatized tokens
    with stopwords removed and only alphabetic tokens kept.
    """
    # Join tokens into a string for spaCy to process
    text = ' '.join(tokens)
    
    # Create a spaCy doc object
    doc = nlp(text)
    
    # Extract lemmas, convert to lowercase, keep only alphabetic tokens, remove stopwords
    lemmas = [
        token.lemma_.lower() 
        for token in doc 
        if token.is_alpha and token.lemma_.lower() not in stop_words
    ]
    
    return lemmas

# Apply lemmatize_tokens to token_regex_ver2 to produce lemmas column
df['lemmas'] = df['token_regex_ver2'].apply(lemmatize_tokens)

# Preview the results for the first 5 rows
print("="*80)
print("Q5 — Lemmatizer for Regex Version 2 Results")
print("="*80)
for i in range(5):
    print(f"\n--- Row {i} ---")
    print(f"Original review_body:\n{df['review_body'].iloc[i]}")
    print(f"\nLemmatized (lemmas):\n{df['lemmas'].iloc[i]}")
    print("-"*80)
