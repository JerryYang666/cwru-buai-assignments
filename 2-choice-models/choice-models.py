# %% [markdown]
# BUAI 435 Assignment 2 - Choice Models  
# Name: Ruihuang Yang  
# NetID: rxy216  
# Date: 09/24/2025  

# %% [markdown]
# ### Import libraries

# %%
import pandas as pd
import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# %% [markdown]
# ## Q1 - Preprocessing & Variable Setup
# 
# According to the assignment requirements:
# - Load dataset into long-format DataFrame (one row per individual × mode)
# - Create exact columns: `ids` (from individual), `alts` (from mode), `choice` (1/0)
# - Manually create dummies & interactions (>8 interaction terms)
# - Show df.info() and first 5 rows

# %%
# Load the TravelMode dataset
df = pd.read_csv('data/TravelMode.csv')

# Create the required column structure
df_clean = df.copy()

# Rename columns to match requirements
df_clean = df_clean.rename(columns={
    'individual': 'ids',
    'mode': 'alts'
})

# Convert choice from 'yes'/'no' to 1/0
df_clean['choice'] = (df_clean['choice'] == 'yes').astype(int)

# Verify data structure: should have 840 rows (210 individuals × 4 modes)
print(f"Dataset shape: {df_clean.shape}")
print(f"Number of unique individuals: {df_clean['ids'].nunique()}")
print(f"Number of unique alternatives: {df_clean['alts'].nunique()}")
print(f"Alternatives: {sorted(df_clean['alts'].unique())}")

# %% [markdown]
# ### Create Dummy Variables and Interactions

# %%
# Create mode dummy variables (car will be the reference category)
df_clean['air'] = (df_clean['alts'] == 'air').astype(int)
df_clean['train'] = (df_clean['alts'] == 'train').astype(int) 
df_clean['bus'] = (df_clean['alts'] == 'bus').astype(int)
# car is the reference category (all dummies = 0)

# Create individual-specific interaction terms (>8 required)
# Income interactions with modes (3 terms)
df_clean['income_air'] = df_clean['income'] * df_clean['air']
df_clean['income_train'] = df_clean['income'] * df_clean['train'] 
df_clean['income_bus'] = df_clean['income'] * df_clean['bus']

# Size interactions with modes (3 terms)
df_clean['size_air'] = df_clean['size'] * df_clean['air']
df_clean['size_train'] = df_clean['size'] * df_clean['train']
df_clean['size_bus'] = df_clean['size'] * df_clean['bus']

# Cost and time interactions with individual characteristics (4 terms)
df_clean['wait_income'] = df_clean['wait'] * df_clean['income'] / 100  # scaled for numerical stability
df_clean['travel_income'] = df_clean['travel'] * df_clean['income'] / 1000  # scaled for numerical stability
df_clean['vcost_size'] = df_clean['vcost'] * df_clean['size'] / 10  # scaled for numerical stability  
df_clean['gcost_size'] = df_clean['gcost'] * df_clean['size'] / 10  # scaled for numerical stability

print("Created interaction terms:")
interaction_vars = ['income_air', 'income_train', 'income_bus', 'size_air', 'size_train', 'size_bus', 
                   'wait_income', 'travel_income', 'vcost_size', 'gcost_size']
print(f"Total interaction terms: {len(interaction_vars)} (requirement: >8 ✓)")
for var in interaction_vars:
    print(f"  - {var}")

# %% [markdown] 
# ### Data Structure Verification

# %%
# Verify that each individual has exactly one choice=1
choice_per_individual = df_clean.groupby('ids')['choice'].sum()
print(f"Choices per individual - Min: {choice_per_individual.min()}, Max: {choice_per_individual.max()}")
print(f"All individuals have exactly 1 choice: {(choice_per_individual == 1).all()}")

# Verify 4 rows per individual (one for each mode)
rows_per_individual = df_clean.groupby('ids').size()
print(f"Rows per individual - Min: {rows_per_individual.min()}, Max: {rows_per_individual.max()}")
print(f"All individuals have exactly 4 alternatives: {(rows_per_individual == 4).all()}")

# %% [markdown]
# ### Required Output: df.info() and First 5 Rows

# %%
print("=== DATASET INFO ===")
df_clean.info()

print("\n=== FIRST 5 ROWS ===")
print(df_clean.head())
