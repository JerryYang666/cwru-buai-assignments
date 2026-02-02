# %% [markdown]
# BUAI 445 Micro Assignment 2  
# Name: Ruihuang Yang  
# NetID: rxy216  
# Date: 02/02/2026  

# %% [markdown]
# ### Import libraries

# %%
import pandas as pd
import pyreadstat

# %% [markdown]
# ### Load the CustomerLimited SPSS Dataset

# %%
# Read the SPSS file
df, meta = pyreadstat.read_sav('data/customerLimited-1.sav')

# Display basic info about the dataset
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

# %% [markdown]
# ## Problem A: Summary Descriptive Statistics for Personal Tendency Variables
# 
# The four Personal Tendency (scale) variables are:
# - PT1Planning: Personal tendency for Planning
# - PT2Spend: Personal tendency for Spending Control Trouble
# - PT3Local: Personal tendency to Prefer Locally-made Products
# - PT4Health: Personal tendency for Caring about Health Benefits

# %%
# Select the four Personal Tendency variables
pt_variables = ['PT1Planning', 'PT2Spend', 'PT3Local', 'PT4Health']

# Get descriptive statistics using describe()
pt_stats = df[pt_variables].describe()

print("Descriptive Statistics for Personal Tendency Variables:")
print("=" * 60)
print(pt_stats)

# %% [markdown]
# ### Interpretation of Personal Tendency Variables
# 
# The descriptive statistics reveal the following insights about the customer base:

# %%
# Additional detailed interpretation
print("\n" + "=" * 60)
print("DETAILED INTERPRETATION")
print("=" * 60)

for var in pt_variables:
    print(f"\n{var} ({meta.column_names_to_labels.get(var, var)}):")
    print(f"  - Mean: {df[var].mean():.2f}")
    print(f"  - Std Dev: {df[var].std():.2f}")
    print(f"  - Min: {df[var].min():.0f}, Max: {df[var].max():.0f}")
    print(f"  - Median (50%): {df[var].median():.2f}")

# %% [markdown]
# ### Problem A Interpretation Summary
# 
# Based on the descriptive statistics for the four Personal Tendency variables (all measured on a 1-5 scale):
# 
# **1. PT1Planning (Personal Tendency for Planning):**
# - Mean = 1.83, Median = 2.0, Std Dev = 0.78
# - The customer base tends to be planners. The low mean and median indicate that most customers 
#   agree they plan ahead (assuming 1 = strongly agree). The relatively low standard deviation 
#   shows consistency in this trait across the sample.
# 
# **2. PT2Spend (Spending Control Trouble):**
# - Mean = 2.94, Median = 3.0, Std Dev = 0.95
# - This variable shows the most variability (highest std dev) and a neutral mean around 3.
#   Customers are mixed on whether they have trouble controlling spending - some do, some don't.
#   This is the most heterogeneous personal tendency in the customer base.
# 
# **3. PT3Local (Preference for Locally-made Products):**
# - Mean = 1.82, Median = 2.0, Std Dev = 0.70
# - Customers strongly prefer locally-made products. The low mean and tight standard deviation
#   indicate this is a consistent characteristic of the customer base - they value local sourcing.
# 
# **4. PT4Health (Caring about Health Benefits):**
# - Mean = 1.69, Median = 2.0, Std Dev = 0.69
# - This has the lowest mean, indicating customers strongly care about health benefits.
#   The low standard deviation shows this is a consistent trait - health consciousness is
#   a defining characteristic of this customer base.
# 
# **Overall Customer Profile:**
# The customer base consists of health-conscious planners who prefer locally-made products.
# They show mixed tendencies regarding spending control. This suggests marketing should 
# emphasize health benefits, local sourcing, and value for planned purchases.

# %% [markdown]
# ## Problem B: Frequencies, Mode, and Median for Demographic Variables
# 
# The three demographic variables are:
# - Gender: female (1) or male (2)
# - Age: under 25 (1), 26-40 (2), 41-65 (3), 66 or older (4)
# - Income: $100,000 and up (1), $50,000 to $100,000 (2), under $50,000 (3)

# %%
# Define demographic variables
demo_variables = ['Gender', 'Age', 'Income']

print("=" * 60)
print("FREQUENCIES, MODE, AND MEDIAN FOR DEMOGRAPHIC VARIABLES")
print("=" * 60)

for var in demo_variables:
    print(f"\n{'=' * 60}")
    print(f"{var.upper()}")
    print("=" * 60)
    
    # Get value labels for interpretation
    value_labels = meta.variable_value_labels.get(var, {})
    
    # Frequency table
    freq = df[var].value_counts().sort_index()
    freq_pct = df[var].value_counts(normalize=True).sort_index() * 100
    
    print("\nFrequency Distribution:")
    print("-" * 40)
    for val in freq.index:
        label = value_labels.get(val, f"Value {val}")
        print(f"  {val:.0f} ({label}): {freq[val]} ({freq_pct[val]:.1f}%)")
    
    # Mode
    mode_val = df[var].mode()[0]
    mode_label = value_labels.get(mode_val, f"Value {mode_val}")
    print(f"\nMode: {mode_val:.0f} ({mode_label})")
    
    # Median
    median_val = df[var].median()
    median_label = value_labels.get(median_val, f"Value {median_val}")
    print(f"Median: {median_val:.0f} ({median_label})")

# %% [markdown]
# ### Problem B Interpretation Summary
# 
# Based on the frequencies, mode, and median for the three demographic variables:
# 
# **1. Gender:**
# - Mode = 1 (female), Median = 1 (female)
# - The customer base is predominantly female (approximately 79% female vs 21% male).
#   This suggests the restaurant/business appeals more strongly to women, or women
#   are more likely to respond to surveys.
# 
# **2. Age:**
# - Mode = 2 (26-40), Median = 2 (26-40)
# - The customer base is primarily working-age adults. The 26-40 age group is the 
#   largest segment (~45%), followed closely by 41-65 (~40%). Very few customers
#   are under 25 or over 65. This is a middle-aged customer base.
# 
# **3. Income:**
# - Mode = 2 ($50,000 to $100,000), Median = 2 ($50,000 to $100,000)
# - The customer base has moderate to high income. The middle income bracket is
#   most common (~42%), with roughly equal proportions in high income (~28%) and
#   lower income (~30%) brackets. This is a middle-class customer base.
# 
# **Overall Demographic Profile:**
# The typical customer is a middle-aged (26-65) female with moderate household income
# ($50,000-$100,000). This demographic profile suggests marketing should target
# working women with families who have disposable income for dining out.

# %% [markdown]
# ## Problem C: Bar Chart and Scatter Plot
# 
# Creating visualizations to better understand the customer base:
# - Bar Chart: Age distribution (categorical variable)
# - Scatter Plot: Relationship between two scale variables

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ### Bar Chart: Age Distribution by Gender

# %%
# Create a bar chart showing Age distribution
fig, ax = plt.subplots(figsize=(10, 6))

# Get age value labels
age_labels = {1: 'Under 25', 2: '26-40', 3: '41-65', 4: '66+'}
gender_labels = {1: 'Female', 2: 'Male'}

# Create labeled columns for plotting
df['Age_Label'] = df['Age'].map(age_labels)
df['Gender_Label'] = df['Gender'].map(gender_labels)

# Count by age and gender
age_gender_counts = df.groupby(['Age_Label', 'Gender_Label']).size().unstack(fill_value=0)

# Reorder age categories
age_order = ['Under 25', '26-40', '41-65', '66+']
age_gender_counts = age_gender_counts.reindex(age_order)

# Create grouped bar chart
age_gender_counts.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'], edgecolor='black')

ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
ax.set_title('Customer Age Distribution by Gender', fontsize=14, fontweight='bold')
ax.legend(title='Gender')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fontsize=9)

plt.tight_layout()
plt.savefig('bar_chart_age_gender.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nBar chart saved as 'bar_chart_age_gender.png'")

# %% [markdown]
# ### Bar Chart Interpretation
# 
# The bar chart shows the age distribution of customers segmented by gender:
# 
# - **Females dominate all age groups**, consistent with the overall 79% female customer base
# - **The 26-40 age group** has the highest number of customers (especially females)
# - **The 41-65 age group** is the second largest segment
# - **Under 25 and 66+** age groups have relatively few customers
# - The gender ratio appears relatively consistent across age groups, suggesting age 
#   does not significantly affect the gender composition of the customer base

# %% [markdown]
# ### Scatter Plot: Health Consciousness vs. Local Product Preference

# %%
# Create a scatter plot showing relationship between two Personal Tendency variables
fig, ax = plt.subplots(figsize=(10, 6))

# Add jitter to see overlapping points better
import numpy as np
np.random.seed(42)
jitter_strength = 0.15

x = df['PT4Health'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
y = df['PT3Local'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))

# Color by income level
income_labels = {1: '$100K+', 2: '$50K-$100K', 3: 'Under $50K'}
df['Income_Label'] = df['Income'].map(income_labels)

colors = {'$100K+': '#2ECC71', '$50K-$100K': '#3498DB', 'Under $50K': '#E74C3C'}

for income_cat in ['$100K+', '$50K-$100K', 'Under $50K']:
    mask = df['Income_Label'] == income_cat
    ax.scatter(x[mask], y[mask], c=colors[income_cat], label=income_cat, 
               alpha=0.6, s=60, edgecolors='white', linewidth=0.5)

ax.set_xlabel('PT4Health (Health Consciousness)\n1=Strongly Agree, 5=Don\'t Know', fontsize=11)
ax.set_ylabel('PT3Local (Local Product Preference)\n1=Strongly Agree, 5=Don\'t Know', fontsize=11)
ax.set_title('Health Consciousness vs. Local Product Preference\n(colored by Income Level)', 
             fontsize=14, fontweight='bold')
ax.legend(title='Income Level', loc='upper right')

# Set axis limits
ax.set_xlim(0.5, 5.5)
ax.set_ylim(0.5, 5.5)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_yticks([1, 2, 3, 4, 5])

plt.tight_layout()
plt.savefig('scatter_plot_health_local.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nScatter plot saved as 'scatter_plot_health_local.png'")

# %% [markdown]
# ### Scatter Plot Interpretation
# 
# The scatter plot shows the relationship between health consciousness (PT4Health) 
# and local product preference (PT3Local), colored by income level:
# 
# - **Strong clustering in the lower-left quadrant** (values 1-2 on both axes) indicates
#   most customers both care about health AND prefer local products
# - **Positive correlation**: Customers who are health-conscious also tend to prefer 
#   locally-made products - these traits go together
# - **Income distribution**: All income levels show similar patterns, suggesting these
#   personal tendencies are consistent across income brackets
# - **Few outliers**: Very few customers scored high (3-5) on either variable, confirming
#   that health consciousness and local preference are defining characteristics of this
#   customer base regardless of income level
# - This suggests marketing messages combining health benefits with local sourcing would
#   resonate across all income segments

# %% [markdown]
# ## Problem D: Crosstab Analysis
# 
# Creating a crosstab to understand the relationship between demographic and behavioral variables.
# We'll examine Age vs. Income to understand the socioeconomic profile of the customer base.

# %%
# Crosstab: Age vs Income
print("=" * 70)
print("CROSSTAB: AGE GROUP BY INCOME LEVEL")
print("=" * 70)

# Create crosstab with counts
crosstab_counts = pd.crosstab(df['Age_Label'], df['Income_Label'], 
                               margins=True, margins_name='Total')

# Reorder rows and columns for logical presentation
age_order = ['Under 25', '26-40', '41-65', '66+', 'Total']
income_order = ['Under $50K', '$50K-$100K', '$100K+', 'Total']

crosstab_counts = crosstab_counts.reindex(index=age_order, columns=income_order)

print("\nFrequency Counts:")
print("-" * 70)
print(crosstab_counts)

# %%
# Crosstab with row percentages (% within each age group)
print("\n" + "=" * 70)
print("ROW PERCENTAGES (% within each Age Group)")
print("=" * 70)

crosstab_row_pct = pd.crosstab(df['Age_Label'], df['Income_Label'], 
                                normalize='index') * 100

crosstab_row_pct = crosstab_row_pct.reindex(index=age_order[:-1], columns=income_order[:-1])

print(crosstab_row_pct.round(1))

# %%
# Crosstab with column percentages (% within each income group)
print("\n" + "=" * 70)
print("COLUMN PERCENTAGES (% within each Income Level)")
print("=" * 70)

crosstab_col_pct = pd.crosstab(df['Age_Label'], df['Income_Label'], 
                                normalize='columns') * 100

crosstab_col_pct = crosstab_col_pct.reindex(index=age_order[:-1], columns=income_order[:-1])

print(crosstab_col_pct.round(1))

# %%
# Visualize the crosstab as a heatmap
fig, ax = plt.subplots(figsize=(10, 6))

# Use the row percentage data for the heatmap
heatmap_data = crosstab_row_pct.reindex(index=['Under 25', '26-40', '41-65', '66+'],
                                         columns=['Under $50K', '$50K-$100K', '$100K+'])

sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlGnBu', 
            cbar_kws={'label': 'Percentage (%)'}, ax=ax,
            linewidths=0.5, linecolor='white')

ax.set_xlabel('Income Level', fontsize=12)
ax.set_ylabel('Age Group', fontsize=12)
ax.set_title('Income Distribution within Each Age Group (%)\n(Row Percentages)', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('crosstab_age_income_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nCrosstab heatmap saved as 'crosstab_age_income_heatmap.png'")

# %% [markdown]
# ### Problem D Crosstab Interpretation
# 
# The crosstab of Age Group by Income Level reveals important insights about the customer base:
# 
# **Key Findings:**
# 
# 1. **Under 25 age group**: 
#    - Predominantly lower income (50% under $50K)
#    - Only 16.7% earn $100K+
#    - This is expected as younger customers are early in their careers
# 
# 2. **26-40 age group (largest segment)**:
#    - More evenly distributed across income levels
#    - 34.6% earn $50K-$100K, 32.1% earn $100K+
#    - This prime working-age group has higher earning potential
# 
# 3. **41-65 age group**:
#    - Highest proportion in middle income (50.7% earn $50K-$100K)
#    - 26.0% earn $100K+
#    - Established careers with stable middle-class income
# 
# 4. **66+ age group**:
#    - Highest proportion of lower income (50% under $50K)
#    - Likely reflects fixed retirement income
#    - Small sample size (n=10) limits conclusions
# 
# **Business Implications:**
# - The 26-40 age group represents the most affluent segment - target premium offerings here
# - The 41-65 group is solidly middle-class - emphasize value propositions
# - Younger and older customers may be more price-sensitive
# - Marketing strategies should be tailored to the income profile of each age segment
