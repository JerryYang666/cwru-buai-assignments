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
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style for beautiful visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create graphs folder if it doesn't exist
if not os.path.exists('graphs'):
    os.makedirs('graphs')
    print("Created 'graphs' folder for saving plots")

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# %% [markdown]
# ## Q1 - Preprocessing & Variable Setup
# 

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

# %% [markdown]
# ## Q2 - Exploratory Data Analysis (EDA)
# 

# %% [markdown]
# ### Descriptive Statistics

# %%
# Descriptive statistics for key variables
print("=== DESCRIPTIVE STATISTICS ===")
key_vars = ['wait', 'vcost', 'travel', 'gcost', 'income', 'size']
desc_stats = df_clean[key_vars].describe()
print(desc_stats.round(2))

# Summary statistics by mode choice
print("\n=== STATISTICS BY CHOSEN MODE ===")
chosen_data = df_clean[df_clean['choice'] == 1]
stats_by_mode = chosen_data.groupby('alts')[key_vars].mean().round(2)
print(stats_by_mode)

# %% [markdown]
# ### Distribution Plots

# %%
# Create distribution plots for key variables (requirement: at least 1 distribution plot)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Variables', fontsize=16, fontweight='bold', y=1.02)

# Income distribution
sns.histplot(data=df_clean, x='income', bins=20, ax=axes[0,0], color='skyblue', alpha=0.7)
axes[0,0].set_title('Income Distribution', fontweight='bold')
axes[0,0].set_xlabel('Household Income')
axes[0,0].set_ylabel('Frequency')

# Party size distribution
sns.countplot(data=df_clean, x='size', ax=axes[0,1], hue='size', legend=False, palette='viridis')
axes[0,1].set_title('Party Size Distribution', fontweight='bold')
axes[0,1].set_xlabel('Party Size')
axes[0,1].set_ylabel('Count')

# Travel time distribution
sns.histplot(data=df_clean, x='travel', bins=30, ax=axes[0,2], color='coral', alpha=0.7)
axes[0,2].set_title('Travel Time Distribution', fontweight='bold')
axes[0,2].set_xlabel('Travel Time (minutes)')
axes[0,2].set_ylabel('Frequency')

# Waiting time distribution
sns.histplot(data=df_clean, x='wait', bins=20, ax=axes[1,0], color='lightgreen', alpha=0.7)
axes[1,0].set_title('Terminal Waiting Time Distribution', fontweight='bold')
axes[1,0].set_xlabel('Waiting Time (minutes)')
axes[1,0].set_ylabel('Frequency')

# Vehicle cost distribution
sns.histplot(data=df_clean, x='vcost', bins=25, ax=axes[1,1], color='gold', alpha=0.7)
axes[1,1].set_title('Vehicle Cost Distribution', fontweight='bold')
axes[1,1].set_xlabel('Vehicle Cost')
axes[1,1].set_ylabel('Frequency')

# Generalized cost distribution
sns.histplot(data=df_clean, x='gcost', bins=25, ax=axes[1,2], color='plum', alpha=0.7)
axes[1,2].set_title('Generalized Cost Distribution', fontweight='bold')
axes[1,2].set_xlabel('Generalized Cost')
axes[1,2].set_ylabel('Frequency')

plt.tight_layout()
# Save the figure
plt.savefig('graphs/01_distribution_plots.png', dpi=300, bbox_inches='tight')
print("Saved distribution plots to graphs/01_distribution_plots.png")
plt.show()

# %% [markdown]
# ### Mode Share Analysis

# %%
# Overall mode shares (chosen alternatives only)
print("=== OVERALL MODE SHARES ===")
chosen_modes = df_clean[df_clean['choice'] == 1].copy()  # Use .copy() to avoid SettingWithCopyWarning
mode_shares = chosen_modes['alts'].value_counts(normalize=True).sort_index()
mode_counts = chosen_modes['alts'].value_counts().sort_index()

# Create mode shares table
mode_share_table = pd.DataFrame({
    'Mode': mode_shares.index,
    'Count': mode_counts.values,
    'Share (%)': (mode_shares.values * 100).round(1)
})
print(mode_share_table)

# Mode shares by income groups
print("\n=== MODE SHARES BY INCOME GROUPS ===")
chosen_modes.loc[:, 'income_group'] = pd.cut(chosen_modes['income'], 
                                             bins=[0, 20, 40, 100], 
                                             labels=['Low (≤20)', 'Medium (21-40)', 'High (>40)'])

mode_by_income = chosen_modes.groupby(['income_group', 'alts'], observed=True).size().unstack(fill_value=0)
mode_by_income_pct = mode_by_income.div(mode_by_income.sum(axis=1), axis=0) * 100
print("Mode shares by income group (%):")
print(mode_by_income_pct.round(1))

# Mode shares by party size
print("\n=== MODE SHARES BY PARTY SIZE ===")
mode_by_size = chosen_modes.groupby(['size', 'alts'], observed=True).size().unstack(fill_value=0)
mode_by_size_pct = mode_by_size.div(mode_by_size.sum(axis=1), axis=0) * 100
print("Mode shares by party size (%):")
print(mode_by_size_pct.round(1))

# %% [markdown]
# ### Mode Share Visualizations

# %%
# Create labeled bar charts for mode shares (requirement: labeled bar chart)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Travel Mode Share Analysis', fontsize=16, fontweight='bold', y=0.98)

# Overall mode shares
ax1 = axes[0,0]
bars1 = ax1.bar(mode_share_table['Mode'], mode_share_table['Share (%)'], 
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
ax1.set_title('Overall Mode Shares', fontweight='bold', fontsize=14)
ax1.set_xlabel('Travel Mode', fontweight='bold')
ax1.set_ylabel('Share (%)', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

# Mode shares by income group
ax2 = axes[0,1]
mode_by_income_pct.T.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax2.set_title('Mode Shares by Income Group', fontweight='bold', fontsize=14)
ax2.set_xlabel('Travel Mode', fontweight='bold')
ax2.set_ylabel('Share (%)', fontweight='bold')
ax2.legend(title='Income Group', title_fontsize=10, fontsize=10)
ax2.grid(axis='y', alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# Mode shares by party size - stacked bar chart
ax3 = axes[1,0]
mode_by_size_pct.plot(kind='bar', stacked=True, ax=ax3, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax3.set_title('Mode Shares by Party Size (Stacked)', fontweight='bold', fontsize=14)
ax3.set_xlabel('Party Size', fontweight='bold')
ax3.set_ylabel('Share (%)', fontweight='bold')
ax3.legend(title='Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(axis='y', alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

# Cost comparison by chosen mode
ax4 = axes[1,1]
cost_by_mode = chosen_modes.groupby('alts')[['vcost', 'gcost']].mean()
cost_by_mode.plot(kind='bar', ax=ax4, color=['#FFA07A', '#20B2AA'])
ax4.set_title('Average Costs by Chosen Mode', fontweight='bold', fontsize=14)
ax4.set_xlabel('Travel Mode', fontweight='bold')
ax4.set_ylabel('Average Cost', fontweight='bold')
ax4.legend(['Vehicle Cost', 'Generalized Cost'])
ax4.grid(axis='y', alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
# Save the figure
plt.savefig('graphs/02_mode_share_analysis.png', dpi=300, bbox_inches='tight')
print("Saved mode share analysis to graphs/02_mode_share_analysis.png")
plt.show()

# %% [markdown]
# ### Key Observations from EDA

# %%
print("=== KEY OBSERVATIONS ===")
print("• **Income strongly influences mode choice**: High-income travelers (>$40k) prefer air travel (40.3%)")
print("  and car (37.5%), while low-income travelers (≤$20k) overwhelmingly choose train (57.1%),")
print("  likely due to cost considerations, with car being least preferred (12.7%) for this group.")
print()
print("• **Party size drives car preference for larger groups**: Single travelers have balanced preferences")
print("  between train (30.7%) and air (29.8%), but larger parties strongly favor car travel, with 4+ person")
print("  parties choosing car 53-100% of the time, reflecting its capacity and cost-sharing advantages.")
print()
print("• **Clear cost-convenience trade-offs across modes**: Air offers the fastest travel (avg 125 min) but")
print("  highest cost ($98), while train provides good value with moderate costs ($37) and no waiting time")
print("  penalty. Car emerges as the most chosen mode overall (28.1%) with zero waiting and moderate costs.")
