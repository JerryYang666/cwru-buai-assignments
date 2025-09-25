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
# ### Load and Explore Data

# %%
def f(x):
  return 3*x+1
