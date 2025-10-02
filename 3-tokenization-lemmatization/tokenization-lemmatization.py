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
print(f"Original dataset shape: {df.shape}")

# Sample 1% of the dataset for faster processing (especially for lemmatization in Q5)
df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)
print(f"Sampled dataset shape (1%): {df.shape}")

print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows of review_body:")
print(df['review_body'].head())

# %% [markdown]
# ## Q1 — Keras Tokenizer
# 

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
# ## Q2 — Regex Tokenizer Version 1
# 

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
# ## Q3 — Regex Tokenizer Version 2
# 

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
# ## Q4 — Regex Tokenizer Version 3 & Removing Some Stop Words
# 

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
# ## Q5 — Lemmatizer for Regex Version 2
# 

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

# %% [markdown]
# ## Q6 — Suggestion of Your Own Tokenizer
# 
# **My Approach: Sentiment-Aware Lemmatization Pipeline for Amazon Reviews**
# 
# ### Explanation:
# 
# For Amazon review analysis, I propose a comprehensive tokenization pipeline that combines:
# 1. **Regex tokenization with contraction handling** (`[A-Za-z']+` pattern)
# 2. **Lowercase normalization** for consistency
# 3. **Lemmatization** to reduce words to base forms
# 4. **Sentiment-aware stopword removal** - preserves critical sentiment modifiers
# 5. **Minimum token length filtering** (≥2 characters) to remove noise
# 6. **Alphabetic-only filtering** to remove remaining noise
# 
# ### Why This Approach is Appropriate for Amazon Reviews:
# 
# 1. **Sentiment Analysis Ready**: Lemmatization normalizes variations like "loved/loving/loves" → "love",
#    making it easier to identify sentiment patterns across reviews.
# 
# 2. **Handles Contractions**: The regex pattern `[A-Za-z']+` preserves contractions like "don't", "can't",
#    which are common in informal review text and carry important sentiment information.
# 
# 3. **Preserves Sentiment Modifiers**: Unlike standard stopword removal, this keeps important words
#    like "not", "never", "very", "really", "too" that are crucial for sentiment analysis.
#    E.g., "not good" vs "good" have opposite meanings!
# 
# 4. **Reduces Vocabulary Size**: Lemmatization significantly reduces vocabulary while preserving meaning,
#    which is crucial for machine learning models and topic analysis.
# 
# 5. **Removes Ultra-Short Tokens**: Filtering tokens with length < 2 removes artifacts like standalone
#    apostrophes or single characters that survived regex, improving data quality.
# 
# 6. **Focuses on Content Words**: By removing most stopwords (while keeping sentiment-critical ones),
#    we emphasize product-specific terms and quality descriptors most relevant for review analysis.
# 
# 7. **Noise Reduction**: Filtering out numbers and special characters removes rating artifacts
#    (e.g., "5/5", "10/10") that don't add semantic value when already captured in structured fields.
# 
# ### Differences from Previous Methods:
# - Unlike Q3 (regex ver2), this excludes numbers which are noise in sentiment analysis
# - Unlike Q4 (regex ver3), this applies lemmatization for better normalization
# - Unlike Q5, this uses a contraction-friendly regex AND preserves sentiment-critical stopwords
# - **NEW**: Adds sentiment-aware stopword filtering and minimum length requirement
# - **NEW**: Single integrated pipeline optimized specifically for sentiment analysis tasks

# %%
# Define sentiment-critical words to preserve (important for review sentiment analysis)
SENTIMENT_WORDS = {
    'not', 'no', 'never', 'neither', 'nor', 'nobody', 'nothing', 'nowhere',
    'very', 'really', 'extremely', 'absolutely', 'totally', 'completely',
    'too', 'quite', 'rather', 'highly', 'barely', 'hardly', 'scarcely'
}

# Define custom tokenization function for Amazon reviews
def tokenize_amazon_review(text):
    """
    Custom tokenization pipeline optimized for Amazon review sentiment analysis.
    
    Steps:
    1. Convert to lowercase
    2. Extract tokens using regex pattern that preserves contractions
    3. Apply lemmatization via spaCy
    4. Filter: keep only alphabetic tokens
    5. Remove stopwords BUT preserve sentiment-critical words
    6. Remove tokens with length < 2 characters
    
    Returns: List of cleaned, lemmatized tokens
    """
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Extract tokens using regex (letters and apostrophes)
    pattern_custom = re.compile(r"[a-z']+")
    tokens = pattern_custom.findall(text)
    
    # Join tokens for spaCy processing
    text_for_spacy = ' '.join(tokens)
    
    # Apply lemmatization
    doc = nlp(text_for_spacy)
    
    # Extract lemmas with sentiment-aware filtering
    cleaned_tokens = [
        token.lemma_.lower() 
        for token in doc 
        if token.is_alpha  # Keep only alphabetic tokens
        and len(token.lemma_) >= 2  # Minimum length requirement
        and (token.lemma_.lower() not in stop_words or token.lemma_.lower() in SENTIMENT_WORDS)  # Remove stopwords EXCEPT sentiment-critical ones
    ]
    
    return cleaned_tokens

# Apply custom tokenizer to review_body
df['token_custom'] = df['review_body'].apply(tokenize_amazon_review)

# Preview the results for the first 5 rows
print("="*80)
print("Q6 — Custom Tokenizer for Amazon Reviews Results")
print("="*80)
for i in range(5):
    print(f"\n--- Row {i} ---")
    print(f"Original review_body:\n{df['review_body'].iloc[i]}")
    print(f"\nCustom Tokenized (token_custom):\n{df['token_custom'].iloc[i]}")
    print("-"*80)
