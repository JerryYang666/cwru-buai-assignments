# %% [markdown]
# BUAI 446 Assignment 1 - Akron Zoo  
# Name: Ruihuang Yang  
# NetID: rxy216  
# Date: 09/07/2025  

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
# ### Load and Explore Data

# %%
# Load training and test datasets
train_data = pd.read_csv('data/ZOOLOG1-TRAIN-2025.csv')
test_data = pd.read_csv('data/ZOOLOG1-TEST-2025.csv')

print("Training Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)

# %% [markdown]
# ### Basic Data Exploration

# %%
# Display first few rows of training data
print("Training Data - First 5 rows:")
print(train_data.head())

# %%
# Display column information
print("\nTraining Data - Column Info:")
print(train_data.info())

# %%
# Display basic statistics
print("\nTraining Data - Descriptive Statistics:")
print(train_data.describe())

# %%
# Check for missing values
print("\nMissing values in training data:")
print(train_data.isnull().sum())

print("\nMissing values in test data:")
print(test_data.isnull().sum())

# %%
# Check target variable distribution
print("\nTarget variable (UPD) distribution in training data:")
print(train_data['UPD'].value_counts())
print(f"\nUpgrade rate: {train_data['UPD'].mean():.3f}")

