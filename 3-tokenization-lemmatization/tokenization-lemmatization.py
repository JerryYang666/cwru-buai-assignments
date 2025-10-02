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
