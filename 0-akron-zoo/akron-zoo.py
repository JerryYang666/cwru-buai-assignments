# %% [markdown]
# BUAI 446 Assignment 1 - Akron Zoo  
# Name: Ruihuang Yang  
# NetID: rxy216  
# Date: 09/07/2025  
# Disclaimer: Some code in this document was generated with assistance from Claude 4.0 Sonnet.

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
# ### EDA

# %%
# Load training and test datasets
train_data = pd.read_csv('data/ZOOLOG1-TRAIN-2025.csv')
test_data = pd.read_csv('data/ZOOLOG1-TEST-2025.csv')

print("Training Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)

# %% [markdown]
# #### Basic Data Exploration

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

# %% [markdown]
# ### EDA Visualization
#

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import chi2_contingency
warnings.filterwarnings('ignore')

# Set up beautiful styling for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Define a professional color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
upgrade_colors = ['#E74C3C', '#2ECC71']  # Red for No Upgrade, Green for Upgrade


# %% [markdown]
# #### Dataset Balance Analysis
#

# %%
# Create a comprehensive view of dataset balance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart showing upgrade distribution
upgrade_counts = train_data['UPD'].value_counts().sort_index()
bars = ax1.bar(['No Upgrade', 'Upgrade'], upgrade_counts.values, 
               color=upgrade_colors, alpha=0.8, edgecolor='white', linewidth=2)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, upgrade_counts.values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{count}\n({count/len(train_data)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=14)

ax1.set_title('Customer Upgrade Distribution\n(Training Dataset)', fontweight='bold', pad=20)
ax1.set_ylabel('Number of Customers', fontweight='bold')
ax1.set_ylim(0, max(upgrade_counts.values) * 1.15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add sample size annotation
ax1.text(0.5, -0.15, f'Total Sample Size: {len(train_data)} customers', 
         transform=ax1.transAxes, ha='center', fontsize=12, style='italic')

# Pie chart for visual balance
wedges, texts, autotexts = ax2.pie(upgrade_counts.values, 
                                  labels=['No Upgrade\n(50.0%)', 'Upgrade\n(50.0%)'],
                                  colors=upgrade_colors, autopct='',
                                  startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'},
                                  wedgeprops={'edgecolor': 'white', 'linewidth': 3})

ax2.set_title('Dataset Balance Visualization\n(Perfectly Balanced)', fontweight='bold', pad=20)

# Add center circle for donut effect
centre_circle = plt.Circle((0,0), 0.50, fc='white', linewidth=2, edgecolor='gray')
ax2.add_artist(centre_circle)
ax2.text(0, 0, 'Balanced\nDataset', ha='center', va='center', 
         fontsize=14, fontweight='bold', color='gray')

plt.tight_layout()
plt.show()

print("Dataset Balance Summary:")
print(f"• No Upgrade: {upgrade_counts[0]} customers ({upgrade_counts[0]/len(train_data)*100:.1f}%)")
print(f"• Upgrade: {upgrade_counts[1]} customers ({upgrade_counts[1]/len(train_data)*100:.1f}%)")
print("• Perfect balance ratio: 1:1")
print(f"• Total sample size: {len(train_data)} customers")


# %% [markdown]
# #### Distance vs Upgrade Behavior Analysis
#

# %%
# Create distance labels for better visualization
distance_labels = {1: '< 10 min', 2: '10-20 min', 3: '21-30 min', 4: '> 30 min'}
train_data['dist_label'] = train_data['dist'].map(distance_labels)

# Calculate upgrade rates by distance
dist_analysis = train_data.groupby('dist_label').agg({
    'UPD': ['count', 'sum', 'mean']
}).round(3)
dist_analysis.columns = ['Total_Customers', 'Upgrades', 'Upgrade_Rate']
dist_analysis = dist_analysis.reindex(['< 10 min', '10-20 min', '21-30 min', '> 30 min'])

print("Distance vs Upgrade Analysis:")
print("=" * 50)
for dist, row in dist_analysis.iterrows():
    print(f"{dist:12s}: {row['Total_Customers']:3.0f} customers, "
          f"{row['Upgrades']:3.0f} upgrades ({row['Upgrade_Rate']*100:5.1f}%)")
print("=" * 50)


# %%
# Create comprehensive distance vs upgrade visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

# 1. Stacked Bar Chart showing absolute numbers
dist_crosstab = pd.crosstab(train_data['dist_label'], train_data['UPD'])
dist_crosstab = dist_crosstab.reindex(['< 10 min', '10-20 min', '21-30 min', '> 30 min'])

bars1 = ax1.bar(dist_crosstab.index, dist_crosstab[0], 
                color=upgrade_colors[0], alpha=0.8, label='No Upgrade', edgecolor='white', linewidth=1)
bars2 = ax1.bar(dist_crosstab.index, dist_crosstab[1], 
                bottom=dist_crosstab[0], color=upgrade_colors[1], alpha=0.8, 
                label='Upgrade', edgecolor='white', linewidth=1)

# Add value labels
for i, (idx, row) in enumerate(dist_crosstab.iterrows()):
    total = row.sum()
    ax1.text(i, row[0]/2, f'{row[0]}', ha='center', va='center', fontweight='bold', color='white')
    ax1.text(i, row[0] + row[1]/2, f'{row[1]}', ha='center', va='center', fontweight='bold', color='white')
    ax1.text(i, total + 5, f'Total: {total}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_title('Customer Distribution by Distance\n(Absolute Numbers)', fontweight='bold', pad=20)
ax1.set_ylabel('Number of Customers', fontweight='bold')
ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 2. Upgrade Rate by Distance (Percentage)
upgrade_rates = dist_analysis['Upgrade_Rate'] * 100
colors_gradient = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = ax2.bar(upgrade_rates.index, upgrade_rates.values, 
               color=colors_gradient, alpha=0.8, edgecolor='white', linewidth=2)

# Add percentage labels on bars
for bar, rate in zip(bars, upgrade_rates.values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax2.set_title('Upgrade Rate by Distance from Zoo\n(Percentage)', fontweight='bold', pad=20)
ax2.set_ylabel('Upgrade Rate (%)', fontweight='bold')
ax2.set_ylim(0, max(upgrade_rates.values) * 1.2)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add average line
avg_rate = train_data['UPD'].mean() * 100
ax2.axhline(y=avg_rate, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax2.text(0.02, avg_rate + 1, f'Overall Average: {avg_rate:.1f}%', 
         transform=ax2.get_yaxis_transform(), fontsize=10, color='red', fontweight='bold')

# 3. Grouped Bar Chart for detailed comparison
x = np.arange(len(dist_analysis.index))
width = 0.35

bars1 = ax3.bar(x - width/2, dist_analysis['Total_Customers'], width, 
                label='Total Customers', color='#3498DB', alpha=0.8, edgecolor='white')
bars2 = ax3.bar(x + width/2, dist_analysis['Upgrades'], width,
                label='Upgrades', color='#2ECC71', alpha=0.8, edgecolor='white')

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax3.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 2,
             f'{int(bar1.get_height())}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax3.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 2,
             f'{int(bar2.get_height())}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax3.set_title('Customer Count vs Upgrades by Distance\n(Side-by-side Comparison)', fontweight='bold', pad=20)
ax3.set_ylabel('Number of Customers', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(dist_analysis.index, rotation=45)
ax3.legend(framealpha=0.9)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# 4. Heatmap showing upgrade patterns
pivot_data = train_data.groupby(['dist_label', 'UPD']).size().unstack(fill_value=0)
pivot_data = pivot_data.reindex(['< 10 min', '10-20 min', '21-30 min', '> 30 min'])
pivot_pct = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

sns.heatmap(pivot_pct, annot=True, fmt='.1f', cmap='RdYlGn', 
            center=50, ax=ax4, cbar_kws={'label': 'Percentage'})
ax4.set_title('Upgrade Behavior Heatmap\n(Percentage Distribution)', fontweight='bold', pad=20)
ax4.set_xlabel('Upgrade Decision', fontweight='bold')
ax4.set_ylabel('Distance from Zoo', fontweight='bold')
ax4.set_xticklabels(['No Upgrade', 'Upgrade'], rotation=0)
ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()


# %%
# Generate insights summary for presentation
print("\n" + "="*70)
print("KEY INSIGHTS FROM DISTANCE ANALYSIS")
print("="*70)

# Find the distance category with highest and lowest upgrade rates
max_upgrade_dist = dist_analysis['Upgrade_Rate'].idxmax()
min_upgrade_dist = dist_analysis['Upgrade_Rate'].idxmin()
max_rate = dist_analysis.loc[max_upgrade_dist, 'Upgrade_Rate'] * 100
min_rate = dist_analysis.loc[min_upgrade_dist, 'Upgrade_Rate'] * 100

print("\n📍 DISTANCE IMPACT ON UPGRADES:")
print(f"   • Highest upgrade rate: {max_upgrade_dist} ({max_rate:.1f}%)")
print(f"   • Lowest upgrade rate:  {min_upgrade_dist} ({min_rate:.1f}%)")
print(f"   • Difference: {max_rate - min_rate:.1f} percentage points")

# Calculate statistical significance (Chi-square test)
chi2, p_value, dof, expected = chi2_contingency(dist_crosstab)

print("\n📊 STATISTICAL SIGNIFICANCE:")
print(f"   • Chi-square statistic: {chi2:.3f}")
print(f"   • P-value: {p_value:.4f}")
print(f"   • Significance: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")

# Business implications
print("\n💡 BUSINESS IMPLICATIONS:")
if max_rate > 50:
    print(f"   • Customers living {max_upgrade_dist} show higher upgrade propensity")
    print("   • Consider targeted marketing for this distance segment")
else:
    print("   • Distance appears to negatively impact upgrade likelihood")
    print("   • Focus on improving value proposition for distant customers")

print(f"   • Overall upgrade rate: {train_data['UPD'].mean()*100:.1f}%")
print("   • Distance-based segmentation may be valuable for strategy")
print("="*70)


# %% [markdown]
# #### Visit Frequency vs Upgrade Behavior
#

# %%
# Create visit frequency labels for better visualization
visit_labels = {1: '≤ 2 visits', 2: '3-4 visits', 3: '5-6 visits', 4: '7-8 visits', 5: '> 8 visits'}
train_data['tvis_label'] = train_data['tvis'].map(visit_labels)

# Calculate upgrade rates by visit frequency
visit_analysis = train_data.groupby('tvis_label').agg({
    'UPD': ['count', 'sum', 'mean']
}).round(3)
visit_analysis.columns = ['Total_Customers', 'Upgrades', 'Upgrade_Rate']
visit_analysis = visit_analysis.reindex(['≤ 2 visits', '3-4 visits', '5-6 visits', '7-8 visits', '> 8 visits'])

# Create a beautiful visualization for visit frequency vs upgrades
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# 1. Upgrade rate by visit frequency
upgrade_rates_visits = visit_analysis['Upgrade_Rate'] * 100
colors_visits = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']

bars = ax1.bar(range(len(upgrade_rates_visits)), upgrade_rates_visits.values, 
               color=colors_visits, alpha=0.8, edgecolor='white', linewidth=2)

# Add percentage labels on bars
for i, (bar, rate) in enumerate(zip(bars, upgrade_rates_visits.values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax1.set_title('Upgrade Rate by Visit Frequency\n(Higher engagement = Higher upgrades?)', fontweight='bold', pad=20)
ax1.set_ylabel('Upgrade Rate (%)', fontweight='bold')
ax1.set_xticks(range(len(upgrade_rates_visits)))
ax1.set_xticklabels(upgrade_rates_visits.index, rotation=45, ha='right')
ax1.set_ylim(0, max(upgrade_rates_visits.values) * 1.2)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add average line
avg_rate = train_data['UPD'].mean() * 100
ax1.axhline(y=avg_rate, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax1.text(0.02, avg_rate + 2, f'Overall Average: {avg_rate:.1f}%', 
         transform=ax1.get_yaxis_transform(), fontsize=10, color='red', fontweight='bold')

# 2. Customer distribution by visit frequency
visit_counts = train_data['tvis_label'].value_counts()
visit_counts = visit_counts.reindex(['≤ 2 visits', '3-4 visits', '5-6 visits', '7-8 visits', '> 8 visits'])

bars = ax2.bar(range(len(visit_counts)), visit_counts.values, 
               color=colors_visits, alpha=0.8, edgecolor='white', linewidth=2)

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, visit_counts.values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 3,
             f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    # Add percentage of total
    pct = count / len(train_data) * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height/2,
             f'{pct:.1f}%', ha='center', va='center', fontweight='bold', color='white', fontsize=10)

ax2.set_title('Customer Distribution by Visit Frequency\n(Engagement Patterns)', fontweight='bold', pad=20)
ax2.set_ylabel('Number of Customers', fontweight='bold')
ax2.set_xticks(range(len(visit_counts)))
ax2.set_xticklabels(visit_counts.index, rotation=45, ha='right')
ax2.set_ylim(0, max(visit_counts.values) * 1.15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nVisit Frequency vs Upgrade Analysis:")
print("=" * 55)
for visit, row in visit_analysis.iterrows():
    print(f"{visit:12s}: {row['Total_Customers']:3.0f} customers, "
          f"{row['Upgrades']:3.0f} upgrades ({row['Upgrade_Rate']*100:5.1f}%)")
print("=" * 55)


# %% [markdown]
# ### Data Preprocessing & Feature Engineering
#
# #### Objective: Prepare data for machine learning models with appropriate encoding and scaling
#

# %%
# Import preprocessing libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Check current data shape and missing values
print("Data Preprocessing Setup")
print("=" * 50)
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Missing values in training: {train_data.isnull().sum().sum()}")
print(f"Missing values in test: {test_data.isnull().sum().sum()}")
print("=" * 50)


# %%
# Feature categorization based on business logic and statistical properties
print("\nFeature Categorization:")
print("=" * 50)

# Features to exclude (ID variables)
id_features = ['NID']

# Target variable
target = 'UPD'

# Categorical features (non-linear relationship) - need dummy encoding
categorical_nonlinear = [
    'gender',        # 1=Male, 2=Female (nominal)
    'mstat',         # 1=Married, 2=Single, 3=Divorced, 4=Widow (nominal)  
    'educ',          # 1=Some college, 2=College, 3=Graduate school (treated as nominal per user)
    'educnew',       # Recoded education variable (nominal)
    'age_rec',       # 1=18-34, 2=35-44, 3=45-54, 4=54+ (treated as nominal per user)
    'tvis'           # 1=≤2, 2=3-4, 3=5-6, 4=7-8, 5=>8 visits (treated as nominal per user)
]

# Ordinal features with linear relationship - use numeric + normalize
ordinal_linear = [
    'dist'           # 1=<10min, 2=10-20min, 3=21-30min, 4=>30min (clear distance progression)
]

# Already standardized perception features (mean≈0, std≈1)
standardized_features = [
    'benefits', 'costs', 'value', 'identity', 'know', 'sat', 'fle', 'trustfor'
]

# Other numeric features that need scaling
numeric_features = [
    'age',           # Age in categories (1-5 scale)
    'size',          # Household size (1-6)
    'child1'         # Number of children/grandchildren (0-6)
]

print(f"Categorical (dummy encode): {categorical_nonlinear}")
print(f"Ordinal linear (normalize): {ordinal_linear}")
print(f"Already standardized: {standardized_features}")
print(f"Numeric (need scaling): {numeric_features}")
print(f"ID features (exclude): {id_features}")
print(f"Target variable: {target}")
print("=" * 50)


# %%
# Create comprehensive preprocessing pipeline
print("\nBuilding Preprocessing Pipeline:")
print("=" * 50)

# Create preprocessing steps for different feature types
preprocessor = ColumnTransformer(
    transformers=[
        # One-hot encode categorical features (drop='first' to avoid multicollinearity)
        ('categorical', OneHotEncoder(drop='first', sparse_output=False), categorical_nonlinear),
        
        # Standardize ordinal features with linear relationship
        ('ordinal', StandardScaler(), ordinal_linear),
        
        # Keep already standardized features as-is
        ('standardized', 'passthrough', standardized_features),
        
        # Standardize other numeric features
        ('numeric', StandardScaler(), numeric_features)
    ],
    remainder='drop'  # Drop any remaining features (like NID)
)

print("Pipeline Components:")
print("• OneHotEncoder: Categorical features → Binary dummy variables")
print("• StandardScaler: Ordinal/Numeric features → Mean=0, Std=1") 
print("• PassThrough: Pre-standardized features → Unchanged")
print("• Remainder: ID features → Dropped")
print("=" * 50)


# %%
# Prepare data for preprocessing
print("\nPreparing Data for Preprocessing:")
print("=" * 50)

# Separate features and target for training data
X_train_raw = train_data.drop(columns=[target])
y_train = train_data[target]

# Separate features for test data (test data has target too, but we'll use it for final evaluation)
X_test_raw = test_data.drop(columns=[target])
y_test = test_data[target]

print(f"Training features shape: {X_train_raw.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test features shape: {X_test_raw.shape}")
print(f"Test target shape: {y_test.shape}")

# Show feature names before preprocessing
print(f"\nOriginal features ({len(X_train_raw.columns)}):")
print(list(X_train_raw.columns))
print("=" * 50)


# %%
# Apply preprocessing pipeline
print("\nApplying Preprocessing Pipeline:")
print("=" * 50)

# Fit preprocessor on training data only (prevents data leakage)
print("Fitting preprocessor on training data...")
X_train_processed = preprocessor.fit_transform(X_train_raw)

# Transform test data using fitted preprocessor
print("Transforming test data...")
X_test_processed = preprocessor.transform(X_test_raw)

print(f"Processed training shape: {X_train_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")
print("=" * 50)


# %%
# Get feature names after preprocessing
print("\nFeature Names After Preprocessing:")
print("=" * 50)

# Get feature names from the fitted preprocessor
feature_names = []

# Get names from each transformer
categorical_names = list(preprocessor.named_transformers_['categorical'].get_feature_names_out(categorical_nonlinear))
ordinal_names = [f"ordinal__{col}" for col in ordinal_linear]
standardized_names = [f"standardized__{col}" for col in standardized_features] 
numeric_names = [f"numeric__{col}" for col in numeric_features]

# Combine all feature names
feature_names = categorical_names + ordinal_names + standardized_names + numeric_names

print(f"Total features after preprocessing: {len(feature_names)}")
print(f"Feature expansion: {X_train_raw.shape[1]} → {len(feature_names)} features")
print("\nFeature breakdown:")
print(f"• Categorical (dummy): {len(categorical_names)} features")
print(f"• Ordinal (scaled): {len(ordinal_names)} features") 
print(f"• Standardized (passthrough): {len(standardized_names)} features")
print(f"• Numeric (scaled): {len(numeric_names)} features")

# Show first few categorical feature names (dummy variables)
print(f"\nSample categorical features (first 10):")
print(categorical_names[:10])
print("=" * 50)


# %%
# Create validation split from training data
print("\nCreating Validation Split:")
print("=" * 50)

# Split training data into train/validation (80/20 split, stratified)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train
)

print(f"Training split: {X_train_split.shape}")
print(f"Validation split: {X_val_split.shape}")
print(f"Training target distribution: {np.bincount(y_train_split)}")
print(f"Validation target distribution: {np.bincount(y_val_split)}")
print(f"Training upgrade rate: {y_train_split.mean():.3f}")
print(f"Validation upgrade rate: {y_val_split.mean():.3f}")
print("=" * 50)


# %% [markdown]
# ### Feature Engineering & Analysis

# %%
# Create DataFrame for easier manipulation and feature engineering
print("\nFeature Engineering Setup:")
print("=" * 50)

# Convert to DataFrame for easier feature engineering
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_val_df = pd.DataFrame(X_val_split, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

print(f"Training DataFrame shape: {X_train_df.shape}")
print(f"Validation DataFrame shape: {X_val_df.shape}")
print(f"Test DataFrame shape: {X_test_df.shape}")
print("=" * 50)


# %%
# Correlation analysis and feature relationships
print("\nCorrelation Analysis:")
print("=" * 50)

# Calculate correlation matrix
correlation_matrix = X_train_df.corr()

# Find highly correlated features (threshold = 0.8)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((
                correlation_matrix.columns[i], 
                correlation_matrix.columns[j], 
                correlation_matrix.iloc[i, j]
            ))

print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
    print(f"  {feat1} ↔ {feat2}: {corr:.3f}")

if len(high_corr_pairs) > 10:
    print(f"  ... and {len(high_corr_pairs) - 10} more pairs")

print("=" * 50)


# %%
# Feature correlation with target variable
print("\nFeature-Target Correlations:")
print("=" * 50)

# Calculate correlations with target
target_correlations = []
for feature in feature_names:
    corr = np.corrcoef(X_train_df[feature], y_train_split)[0, 1]
    target_correlations.append((feature, corr))

# Sort by absolute correlation
target_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("Top 15 features most correlated with upgrade decision:")
for i, (feature, corr) in enumerate(target_correlations[:15], 1):
    print(f"{i:2d}. {feature:<40}: {corr:6.3f}")

print("\nBottom 5 features least correlated with upgrade decision:")
for i, (feature, corr) in enumerate(target_correlations[-5:], len(target_correlations)-4):
    print(f"{i:2d}. {feature:<40}: {corr:6.3f}")

print("=" * 50)


# %% [markdown]
# ### Machine Learning Model Development
# 
# #### Implementing 5 models: Logistic Regression, SVM, Naive Bayes, Random Forest, GBM

# %%
# Import machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import time

print("Machine Learning Model Setup")
print("=" * 50)


# %%
# Define models with initial parameters
print("\nInitializing Models:")
print("=" * 50)

models = {
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000,
        penalty='l2'
    ),
    
    'SVM': SVC(
        random_state=42,
        probability=True,  # Enable probability estimates for ROC-AUC
        kernel='rbf'
    ),
    
    'Naive Bayes': GaussianNB(),
    
    'Random Forest': RandomForestClassifier(
        random_state=42,
        n_estimators=100
    ),
    
    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42,
        n_estimators=100
    )
}

for name, model in models.items():
    print(f"✓ {name}: {type(model).__name__}")

print("=" * 50)


# %%
# Cross-validation evaluation
print("\nCross-Validation Evaluation:")
print("=" * 50)

# Setup stratified k-fold cross-validation
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
cv_results = {}
training_times = {}

print("Performing 5-fold cross-validation for each model...")
print("\nModel Performance (CV Mean ± Std):")
print("-" * 70)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Time the training
    start_time = time.time()
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X_train_split, y_train_split, 
        cv=cv_folds, scoring='roc_auc', n_jobs=-1
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Store results
    cv_results[name] = cv_scores
    training_times[name] = training_time
    
    # Print results
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    print(f"{name:<20}: {mean_score:.4f} ± {std_score:.4f} (Time: {training_time:.2f}s)")

print("-" * 70)
print("Metric: ROC-AUC (Higher is better)")
print("=" * 50)


# %%
# Train models on full training split and evaluate on validation
print("\nValidation Set Evaluation:")
print("=" * 50)

# Store validation results
val_results = {}
trained_models = {}

print("Training models on full training split and evaluating on validation set...")
print("\nValidation Performance:")
print("-" * 80)
print(f"{'Model':<20} {'ROC-AUC':<8} {'Accuracy':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}")
print("-" * 80)

for name, model in models.items():
    # Train model on training split
    model.fit(X_train_split, y_train_split)
    trained_models[name] = model
    
    # Predict on validation set
    val_pred = model.predict(X_val_split)
    val_pred_proba = model.predict_proba(X_val_split)[:, 1]
    
    # Calculate metrics
    val_auc = roc_auc_score(y_val_split, val_pred_proba)
    val_accuracy = (val_pred == y_val_split).mean()
    
    # Get precision, recall, f1 from classification report
    report = classification_report(y_val_split, val_pred, output_dict=True)
    val_precision = report['1']['precision']
    val_recall = report['1']['recall']
    val_f1 = report['1']['f1-score']
    
    # Store results
    val_results[name] = {
        'auc': val_auc,
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'predictions': val_pred,
        'probabilities': val_pred_proba
    }
    
    # Print results
    print(f"{name:<20} {val_auc:<8.4f} {val_accuracy:<8.4f} {val_precision:<10.4f} {val_recall:<8.4f} {val_f1:<8.4f}")

print("-" * 80)
print("=" * 50)


# %%
# Model ranking and selection
print("\nModel Ranking and Selection:")
print("=" * 50)

# Rank models by ROC-AUC on validation set
model_ranking = sorted(val_results.items(), key=lambda x: x[1]['auc'], reverse=True)

print("Model Performance Ranking (by ROC-AUC):")
print("-" * 60)
print(f"{'Rank':<5} {'Model':<20} {'ROC-AUC':<10} {'CV Score':<12} {'Robustness'}")
print("-" * 60)

for i, (name, results) in enumerate(model_ranking, 1):
    cv_mean = cv_results[name].mean()
    cv_std = cv_results[name].std()
    robustness = "High" if cv_std < 0.02 else "Medium" if cv_std < 0.05 else "Low"
    
    print(f"{i:<5} {name:<20} {results['auc']:<10.4f} {cv_mean:.4f}±{cv_std:.3f} {robustness}")

print("-" * 60)

# Select best model
best_model_name = model_ranking[0][0]
best_model = trained_models[best_model_name]
best_results = val_results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   • ROC-AUC: {best_results['auc']:.4f}")
print(f"   • Accuracy: {best_results['accuracy']:.4f}")
print(f"   • F1-Score: {best_results['f1']:.4f}")
print(f"   • Cross-validation: {cv_results[best_model_name].mean():.4f} ± {cv_results[best_model_name].std():.3f}")
print("=" * 50)


# %%
# Feature importance analysis for interpretable models
print("\nFeature Importance Analysis:")
print("=" * 50)

# Extract feature importance from different models
importance_results = {}

# 1. Logistic Regression - Coefficients
if 'Logistic Regression' in trained_models:
    lr_model = trained_models['Logistic Regression']
    lr_coef = lr_model.coef_[0]
    importance_results['Logistic Regression'] = list(zip(feature_names, lr_coef))

# 2. Random Forest - Feature Importance
if 'Random Forest' in trained_models:
    rf_model = trained_models['Random Forest']
    rf_importance = rf_model.feature_importances_
    importance_results['Random Forest'] = list(zip(feature_names, rf_importance))

# 3. Gradient Boosting - Feature Importance
if 'Gradient Boosting' in trained_models:
    gb_model = trained_models['Gradient Boosting']
    gb_importance = gb_model.feature_importances_
    importance_results['Gradient Boosting'] = list(zip(feature_names, gb_importance))

# Display feature importance for best model
if best_model_name in importance_results:
    print(f"\nTop 15 Most Important Features ({best_model_name}):")
    print("-" * 70)
    
    best_importance = importance_results[best_model_name]
    if best_model_name == 'Logistic Regression':
        # Sort by absolute coefficient value
        best_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"{'Feature':<40} {'Coefficient':<15} {'Abs Value':<10}")
        print("-" * 70)
        for i, (feature, coef) in enumerate(best_importance[:15], 1):
            print(f"{i:2d}. {feature:<35} {coef:8.4f} {abs(coef):8.4f}")
    else:
        # Sort by importance value
        best_importance.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Feature':<40} {'Importance':<15}")
        print("-" * 70)
        for i, (feature, importance) in enumerate(best_importance[:15], 1):
            print(f"{i:2d}. {feature:<35} {importance:8.4f}")

print("=" * 50)


# %%
# Final test set evaluation
print("\nFinal Test Set Evaluation:")
print("=" * 50)

# Retrain best model on full training data (training + validation)
print(f"Retraining {best_model_name} on full training data...")
final_model = models[best_model_name]
final_model.fit(X_train_processed, y_train)

# Predict on test set
test_pred = final_model.predict(X_test_processed)
test_pred_proba = final_model.predict_proba(X_test_processed)[:, 1]

# Calculate test metrics
test_auc = roc_auc_score(y_test, test_pred_proba)
test_accuracy = (test_pred == y_test).mean()

# Get detailed metrics
test_report = classification_report(y_test, test_pred, output_dict=True)
test_precision = test_report['1']['precision']
test_recall = test_report['1']['recall']
test_f1 = test_report['1']['f1-score']

print("\nFinal Model Performance on Test Set:")
print("-" * 50)
print(f"Model: {best_model_name}")
print(f"ROC-AUC:   {test_auc:.4f}")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

# Confusion Matrix
test_cm = confusion_matrix(y_test, test_pred)
print("\nConfusion Matrix:")
print("                 Predicted")
print("Actual    No Upgrade  Upgrade")
print(f"No Upgrade    {test_cm[0,0]:4d}     {test_cm[0,1]:4d}")
print(f"Upgrade       {test_cm[1,0]:4d}     {test_cm[1,1]:4d}")

print("-" * 50)
print("=" * 50)


# %%
# Business insights and recommendations
print("\nBusiness Insights & Recommendations:")
print("=" * 60)

print("\n📊 MODEL PERFORMANCE SUMMARY:")
print(f"   • Best performing model: {best_model_name}")
print(f"   • Test set accuracy: {test_accuracy:.1%}")
print(f"   • Test set ROC-AUC: {test_auc:.3f}")
print(f"   • Model can identify {test_recall:.1%} of customers who will upgrade")
print(f"   • {test_precision:.1%} of predicted upgrades are correct")

# Calculate business impact
total_customers = len(test_data)
actual_upgrades = y_test.sum()
predicted_upgrades = test_pred.sum()
correctly_identified = (test_pred & y_test).sum()

print("\n💼 BUSINESS IMPACT:")
print(f"   • Total test customers: {total_customers}")
print(f"   • Actual upgrades: {actual_upgrades} ({actual_upgrades/total_customers:.1%})")
print(f"   • Predicted upgrades: {predicted_upgrades}")
print(f"   • Correctly identified upgrades: {correctly_identified}")
print("   • Potential revenue impact: High (targeted marketing efficiency)")

print("\n🎯 KEY RECOMMENDATIONS:")
if best_model_name in importance_results:
    top_features = importance_results[best_model_name]
    if best_model_name == 'Logistic Regression':
        top_features.sort(key=lambda x: abs(x[1]), reverse=True)
    else:
        top_features.sort(key=lambda x: x[1], reverse=True)
    
    print("   • Focus on top predictive features:")
    for i, (feature, _) in enumerate(top_features[:3], 1):
        clean_feature = feature.replace('standardized__', '').replace('categorical__', '').replace('numeric__', '').replace('ordinal__', '')
        print(f"     {i}. {clean_feature}")

print("   • Implement targeted retention strategies")
print("   • Use model for customer segmentation")
print("   • Monitor model performance quarterly")

print("=" * 60)

