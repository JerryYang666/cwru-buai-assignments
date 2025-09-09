# %% [markdown]
# BUAI 446 Assignment 1 - Akron Zoo
# Name: Ruihuang Yang
# NetID: rxy216
# Date: 09/07/2025
# Disclaimer: Some code in this document was generated with assistance from Claude 4.0 Sonnet.
# Prompts used:
# - "Please clean up the code, make it more well formatted and more readable. For all comments, please optimize the grammar and sentence structure."
# - "Please generate beautiful plots for this ML pipeline at important points along the way so that I can use the visualizations to report the results to executives of the company."
# - "Please add comprehensive print statements in the code so that I can get lots of useful information for the report to executives of the company."

# %% [markdown]
# ### Import libraries

# %%
import pandas as pd
import numpy as np
import random

SEED = 42

# Set seeds for reproducibility
np.random.seed(SEED)
random.seed(SEED)

# %% [markdown]
# ### EDA

# %%
# Load training and test datasets
train_data = pd.read_csv("data/ZOOLOG1-TRAIN-2025.csv")
test_data = pd.read_csv("data/ZOOLOG1-TEST-2025.csv")

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
print(train_data["UPD"].value_counts())
print(f"\nUpgrade rate: {train_data['UPD'].mean():.3f}")

# %% [markdown]
# ### EDA Visualization
#

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import chi2_contingency
import os

warnings.filterwarnings("ignore")

# Create graphs directory if it doesn't exist
os.makedirs("graphs", exist_ok=True)

# Set up beautiful styling for plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

# Define a professional color palette
colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E83", "#27AE60"]
upgrade_colors = ["#E74C3C", "#2ECC71"]  # Red for No Upgrade, Green for Upgrade


# %% [markdown]
# #### Dataset Balance Analysis
#

# %%
# Create a comprehensive view of dataset balance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart showing upgrade distribution
upgrade_counts = train_data["UPD"].value_counts().sort_index()
bars = ax1.bar(
    ["No Upgrade", "Upgrade"],
    upgrade_counts.values,
    color=upgrade_colors,
    alpha=0.8,
    edgecolor="white",
    linewidth=2,
)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, upgrade_counts.values)):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 5,
        f"{count}\n({count / len(train_data) * 100:.1f}%)",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=14,
    )

ax1.set_title(
    "Customer Upgrade Distribution\n(Training Dataset)", fontweight="bold", pad=20
)
ax1.set_ylabel("Number of Customers", fontweight="bold")
ax1.set_ylim(0, max(upgrade_counts.values) * 1.15)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# Add sample size annotation
ax1.text(
    0.5,
    -0.15,
    f"Total Sample Size: {len(train_data)} customers",
    transform=ax1.transAxes,
    ha="center",
    fontsize=12,
    style="italic",
)

# Pie chart for visual balance
wedges, texts, autotexts = ax2.pie(
    upgrade_counts.values,
    labels=["No Upgrade\n(50.0%)", "Upgrade\n(50.0%)"],
    colors=upgrade_colors,
    autopct="",
    startangle=90,
    textprops={"fontsize": 14, "fontweight": "bold"},
    wedgeprops={"edgecolor": "white", "linewidth": 3},
)

ax2.set_title(
    "Dataset Balance Visualization\n(Perfectly Balanced)", fontweight="bold", pad=20
)

# Add center circle for donut effect
centre_circle = plt.Circle((0, 0), 0.50, fc="white", linewidth=2, edgecolor="gray")
ax2.add_artist(centre_circle)
ax2.text(
    0,
    0,
    "Balanced\nDataset",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="gray",
)

plt.tight_layout()
plt.savefig("graphs/01_dataset_balance.png", dpi=300, bbox_inches="tight")
plt.close()

print("Dataset Balance Summary:")
print(
    f"• No Upgrade: {upgrade_counts[0]} customers ({upgrade_counts[0] / len(train_data) * 100:.1f}%)"
)
print(
    f"• Upgrade: {upgrade_counts[1]} customers ({upgrade_counts[1] / len(train_data) * 100:.1f}%)"
)
print("• Perfect balance ratio: 1:1")
print(f"• Total sample size: {len(train_data)} customers")


# %% [markdown]
# #### Distance vs Upgrade Behavior Analysis
#

# %%
# Create distance labels for better visualization
distance_labels = {1: "< 10 min", 2: "10-20 min", 3: "21-30 min", 4: "> 30 min"}
train_data["dist_label"] = train_data["dist"].map(distance_labels)

# Calculate upgrade rates by distance
dist_analysis = (
    train_data.groupby("dist_label").agg({"UPD": ["count", "sum", "mean"]}).round(3)
)
dist_analysis.columns = ["Total_Customers", "Upgrades", "Upgrade_Rate"]
dist_analysis = dist_analysis.reindex(
    ["< 10 min", "10-20 min", "21-30 min", "> 30 min"]
)

print("Distance vs Upgrade Analysis:")
print("=" * 50)
for dist, row in dist_analysis.iterrows():
    print(
        f"{dist:12s}: {row['Total_Customers']:3.0f} customers, "
        f"{row['Upgrades']:3.0f} upgrades ({row['Upgrade_Rate'] * 100:5.1f}%)"
    )
print("=" * 50)


# %%
# Create comprehensive distance vs upgrade visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

# 1. Stacked Bar Chart showing absolute numbers
dist_crosstab = pd.crosstab(train_data["dist_label"], train_data["UPD"])
dist_crosstab = dist_crosstab.reindex(
    ["< 10 min", "10-20 min", "21-30 min", "> 30 min"]
)

bars1 = ax1.bar(
    dist_crosstab.index,
    dist_crosstab[0],
    color=upgrade_colors[0],
    alpha=0.8,
    label="No Upgrade",
    edgecolor="white",
    linewidth=1,
)
bars2 = ax1.bar(
    dist_crosstab.index,
    dist_crosstab[1],
    bottom=dist_crosstab[0],
    color=upgrade_colors[1],
    alpha=0.8,
    label="Upgrade",
    edgecolor="white",
    linewidth=1,
)

# Add value labels
for i, (idx, row) in enumerate(dist_crosstab.iterrows()):
    total = row.sum()
    ax1.text(
        i,
        row[0] / 2,
        f"{row[0]}",
        ha="center",
        va="center",
        fontweight="bold",
        color="white",
    )
    ax1.text(
        i,
        row[0] + row[1] / 2,
        f"{row[1]}",
        ha="center",
        va="center",
        fontweight="bold",
        color="white",
    )
    ax1.text(
        i,
        total + 5,
        f"Total: {total}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

ax1.set_title(
    "Customer Distribution by Distance\n(Absolute Numbers)", fontweight="bold", pad=20
)
ax1.set_ylabel("Number of Customers", fontweight="bold")
ax1.legend(loc="upper right", framealpha=0.9)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# 2. Upgrade Rate by Distance (Percentage)
upgrade_rates = dist_analysis["Upgrade_Rate"] * 100
colors_gradient = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

bars = ax2.bar(
    upgrade_rates.index,
    upgrade_rates.values,
    color=colors_gradient,
    alpha=0.8,
    edgecolor="white",
    linewidth=2,
)

# Add percentage labels on bars
for bar, rate in zip(bars, upgrade_rates.values):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{rate:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )

ax2.set_title(
    "Upgrade Rate by Distance from Zoo\n(Percentage)", fontweight="bold", pad=20
)
ax2.set_ylabel("Upgrade Rate (%)", fontweight="bold")
ax2.set_ylim(0, max(upgrade_rates.values) * 1.2)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

# Add average line
avg_rate = train_data["UPD"].mean() * 100
ax2.axhline(y=avg_rate, color="red", linestyle="--", alpha=0.7, linewidth=2)
ax2.text(
    0.02,
    avg_rate + 1,
    f"Overall Average: {avg_rate:.1f}%",
    transform=ax2.get_yaxis_transform(),
    fontsize=10,
    color="red",
    fontweight="bold",
)

# 3. Grouped Bar Chart for detailed comparison
x = np.arange(len(dist_analysis.index))
width = 0.35

bars1 = ax3.bar(
    x - width / 2,
    dist_analysis["Total_Customers"],
    width,
    label="Total Customers",
    color="#3498DB",
    alpha=0.8,
    edgecolor="white",
)
bars2 = ax3.bar(
    x + width / 2,
    dist_analysis["Upgrades"],
    width,
    label="Upgrades",
    color="#2ECC71",
    alpha=0.8,
    edgecolor="white",
)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax3.text(
        bar1.get_x() + bar1.get_width() / 2.0,
        bar1.get_height() + 2,
        f"{int(bar1.get_height())}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )
    ax3.text(
        bar2.get_x() + bar2.get_width() / 2.0,
        bar2.get_height() + 2,
        f"{int(bar2.get_height())}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

ax3.set_title(
    "Customer Count vs Upgrades by Distance\n(Side-by-side Comparison)",
    fontweight="bold",
    pad=20,
)
ax3.set_ylabel("Number of Customers", fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(dist_analysis.index, rotation=45)
ax3.legend(framealpha=0.9)
ax3.grid(axis="y", alpha=0.3, linestyle="--")

# 4. Heatmap showing upgrade patterns
pivot_data = train_data.groupby(["dist_label", "UPD"]).size().unstack(fill_value=0)
pivot_data = pivot_data.reindex(["< 10 min", "10-20 min", "21-30 min", "> 30 min"])
pivot_pct = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

sns.heatmap(
    pivot_pct,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    center=50,
    ax=ax4,
    cbar_kws={"label": "Percentage"},
)
ax4.set_title(
    "Upgrade Behavior Heatmap\n(Percentage Distribution)", fontweight="bold", pad=20
)
ax4.set_xlabel("Upgrade Decision", fontweight="bold")
ax4.set_ylabel("Distance from Zoo", fontweight="bold")
ax4.set_xticklabels(["No Upgrade", "Upgrade"], rotation=0)
ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig("graphs/02_distance_analysis.png", dpi=300, bbox_inches="tight")
plt.close()


# %%
# Generate insights summary for presentation
print("\n" + "=" * 70)
print("KEY INSIGHTS FROM DISTANCE ANALYSIS")
print("=" * 70)

# Find the distance category with highest and lowest upgrade rates
max_upgrade_dist = dist_analysis["Upgrade_Rate"].idxmax()
min_upgrade_dist = dist_analysis["Upgrade_Rate"].idxmin()
max_rate = dist_analysis.loc[max_upgrade_dist, "Upgrade_Rate"] * 100
min_rate = dist_analysis.loc[min_upgrade_dist, "Upgrade_Rate"] * 100

print("\n📍 DISTANCE IMPACT ON UPGRADES:")
print(f"   • Highest upgrade rate: {max_upgrade_dist} ({max_rate:.1f}%)")
print(f"   • Lowest upgrade rate:  {min_upgrade_dist} ({min_rate:.1f}%)")
print(f"   • Difference: {max_rate - min_rate:.1f} percentage points")

# Calculate statistical significance (Chi-square test)
chi2, p_value, dof, expected = chi2_contingency(dist_crosstab)

print("\n📊 STATISTICAL SIGNIFICANCE:")
print(f"   • Chi-square statistic: {chi2:.3f}")
print(f"   • P-value: {p_value:.4f}")
print(
    f"   • Significance: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05"
)

# Business implications
print("\n💡 BUSINESS IMPLICATIONS:")
if max_rate > 50:
    print(f"   • Customers living {max_upgrade_dist} show higher upgrade propensity")
    print("   • Consider targeted marketing for this distance segment")
else:
    print("   • Distance appears to negatively impact upgrade likelihood")
    print("   • Focus on improving value proposition for distant customers")

print(f"   • Overall upgrade rate: {train_data['UPD'].mean() * 100:.1f}%")
print("   • Distance-based segmentation may be valuable for strategy")
print("=" * 70)


# %% [markdown]
# #### Visit Frequency vs Upgrade Behavior
#

# %%
# Create visit frequency labels for better visualization
visit_labels = {
    1: "≤ 2 visits",
    2: "3-4 visits",
    3: "5-6 visits",
    4: "7-8 visits",
    5: "> 8 visits",
}
train_data["tvis_label"] = train_data["tvis"].map(visit_labels)

# Calculate upgrade rates by visit frequency
visit_analysis = (
    train_data.groupby("tvis_label").agg({"UPD": ["count", "sum", "mean"]}).round(3)
)
visit_analysis.columns = ["Total_Customers", "Upgrades", "Upgrade_Rate"]
visit_analysis = visit_analysis.reindex(
    ["≤ 2 visits", "3-4 visits", "5-6 visits", "7-8 visits", "> 8 visits"]
)

# Create a beautiful visualization for visit frequency vs upgrades
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# 1. Upgrade rate by visit frequency
upgrade_rates_visits = visit_analysis["Upgrade_Rate"] * 100
colors_visits = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"]

bars = ax1.bar(
    range(len(upgrade_rates_visits)),
    upgrade_rates_visits.values,
    color=colors_visits,
    alpha=0.8,
    edgecolor="white",
    linewidth=2,
)

# Add percentage labels on bars
for i, (bar, rate) in enumerate(zip(bars, upgrade_rates_visits.values)):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{rate:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

ax1.set_title(
    "Upgrade Rate by Visit Frequency\n(Higher engagement = Higher upgrades?)",
    fontweight="bold",
    pad=20,
)
ax1.set_ylabel("Upgrade Rate (%)", fontweight="bold")
ax1.set_xticks(range(len(upgrade_rates_visits)))
ax1.set_xticklabels(upgrade_rates_visits.index, rotation=45, ha="right")
ax1.set_ylim(0, max(upgrade_rates_visits.values) * 1.2)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# Add average line
avg_rate = train_data["UPD"].mean() * 100
ax1.axhline(y=avg_rate, color="red", linestyle="--", alpha=0.7, linewidth=2)
ax1.text(
    0.02,
    avg_rate + 2,
    f"Overall Average: {avg_rate:.1f}%",
    transform=ax1.get_yaxis_transform(),
    fontsize=10,
    color="red",
    fontweight="bold",
)

# 2. Customer distribution by visit frequency
visit_counts = train_data["tvis_label"].value_counts()
visit_counts = visit_counts.reindex(
    ["≤ 2 visits", "3-4 visits", "5-6 visits", "7-8 visits", "> 8 visits"]
)

bars = ax2.bar(
    range(len(visit_counts)),
    visit_counts.values,
    color=colors_visits,
    alpha=0.8,
    edgecolor="white",
    linewidth=2,
)

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, visit_counts.values)):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 3,
        f"{count}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )
    # Add percentage of total
    pct = count / len(train_data) * 100
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height / 2,
        f"{pct:.1f}%",
        ha="center",
        va="center",
        fontweight="bold",
        color="white",
        fontsize=10,
    )

ax2.set_title(
    "Customer Distribution by Visit Frequency\n(Engagement Patterns)",
    fontweight="bold",
    pad=20,
)
ax2.set_ylabel("Number of Customers", fontweight="bold")
ax2.set_xticks(range(len(visit_counts)))
ax2.set_xticklabels(visit_counts.index, rotation=45, ha="right")
ax2.set_ylim(0, max(visit_counts.values) * 1.15)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("graphs/03_visit_frequency_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# Print summary statistics
print("\nVisit Frequency vs Upgrade Analysis:")
print("=" * 55)
for visit, row in visit_analysis.iterrows():
    print(
        f"{visit:12s}: {row['Total_Customers']:3.0f} customers, "
        f"{row['Upgrades']:3.0f} upgrades ({row['Upgrade_Rate'] * 100:5.1f}%)"
    )
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
id_features = ["NID"]

# Target variable
target = "UPD"

# Categorical features (non-linear relationship) - need dummy encoding
categorical_nonlinear = [
    "gender",  # 1=Male, 2=Female (nominal)
    "mstat",  # 1=Married, 2=Single, 3=Divorced, 4=Widow (nominal)
    "educ",  # 1=Some college, 2=College, 3=Graduate school (treated as nominal per user)
    "educnew",  # Recoded education variable (nominal)
    "age_rec",  # 1=18-34, 2=35-44, 3=45-54, 4=54+ (treated as nominal per user)
    "tvis",  # 1=≤2, 2=3-4, 3=5-6, 4=7-8, 5=>8 visits (treated as nominal per user)
]

# Ordinal features with linear relationship - use numeric + normalize
ordinal_linear = [
    "dist",  # 1=<10min, 2=10-20min, 3=21-30min, 4=>30min (clear distance progression)
]

# Already standardized perception features (mean≈0, std≈1)
standardized_features = [
    "benefits",
    "costs",
    "value",
    "identity",
    "know",
    "sat",
    "fle",
    "trustfor",
]

# Other numeric features that need scaling
numeric_features = [
    "age",  # Age in categories (1-5 scale)
    "size",  # Household size (1-6)
    "child1",  # Number of children/grandchildren (0-6)
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
        (
            "categorical",
            OneHotEncoder(drop="first", sparse_output=False),
            categorical_nonlinear,
        ),
        # Standardize ordinal features with linear relationship
        ("ordinal", StandardScaler(), ordinal_linear),
        # Keep already standardized features as-is
        ("standardized", "passthrough", standardized_features),
        # Standardize other numeric features
        ("numeric", StandardScaler(), numeric_features),
    ],
    remainder="drop",  # Drop any remaining features (like NID)
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
categorical_names = list(
    preprocessor.named_transformers_["categorical"].get_feature_names_out(
        categorical_nonlinear
    )
)
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
print("\nSample categorical features (first 10):")
print(categorical_names[:10])
print("=" * 50)


# %%
# Create validation split from training data
print("\nCreating Validation Split:")
print("=" * 50)

# Split training data into train/validation (80/20 split, stratified)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=SEED, stratify=y_train
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
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append(
                (
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j],
                )
            )

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
    corr = np.corrcoef(X_train_df[feature], y_train)[0, 1]
    target_correlations.append((feature, corr))

# Sort by absolute correlation
target_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("Top 15 features most correlated with upgrade decision:")
for i, (feature, corr) in enumerate(target_correlations[:15], 1):
    print(f"{i:2d}. {feature:<40}: {corr:6.3f}")

print("\nBottom 5 features least correlated with upgrade decision:")
for i, (feature, corr) in enumerate(
    target_correlations[-5:], len(target_correlations) - 4
):
    print(f"{i:2d}. {feature:<40}: {corr:6.3f}")

print("=" * 50)


# %% [markdown]
# ### Machine Learning Model Development
#
# #### Implementing 6 models: Logistic Regression, SVM, Naive Bayes, Random Forest, GBM, KNN

# %%
# Import machine learning libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import time

print("Machine Learning Model Setup")
print("=" * 50)


# %%
# Define models with initial parameters
print("\nInitializing Models:")
print("=" * 50)

models = {
    "Logistic Regression": LogisticRegression(
        random_state=SEED, max_iter=1000, penalty="l2"
    ),
    "SVM": SVC(
        random_state=SEED,
        probability=True,  # Enable probability estimates for ROC-AUC
        kernel="rbf",
    ),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=SEED, n_estimators=500),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=SEED, n_estimators=500
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights="distance"),
}

for name, model in models.items():
    print(f"✓ {name}: {type(model).__name__}")

print("=" * 50)


# %%
# Cross-validation evaluation
print("\nCross-Validation Evaluation:")
print("=" * 50)

# Setup stratified k-fold cross-validation
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

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
        model, X_train_split, y_train_split, cv=cv_folds, scoring="roc_auc", n_jobs=-1
    )

    end_time = time.time()
    training_time = end_time - start_time

    # Store results
    cv_results[name] = cv_scores
    training_times[name] = training_time

    # Print results
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    print(
        f"{name:<20}: {mean_score:.4f} ± {std_score:.4f} (Time: {training_time:.2f}s)"
    )

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
print(
    f"{'Model':<20} {'ROC-AUC':<8} {'Accuracy':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<8}"
)
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
    val_precision = report["1"]["precision"]
    val_recall = report["1"]["recall"]
    val_f1 = report["1"]["f1-score"]

    # Store results
    val_results[name] = {
        "auc": val_auc,
        "accuracy": val_accuracy,
        "precision": val_precision,
        "recall": val_recall,
        "f1": val_f1,
        "predictions": val_pred,
        "probabilities": val_pred_proba,
    }

    # Print results
    print(
        f"{name:<20} {val_auc:<8.4f} {val_accuracy:<8.4f} {val_precision:<10.4f} {val_recall:<8.4f} {val_f1:<8.4f}"
    )

print("-" * 80)
print("=" * 50)


# %%
# Model ranking and selection
print("\nModel Ranking and Selection:")
print("=" * 50)

# Rank models by ROC-AUC on validation set
model_ranking = sorted(val_results.items(), key=lambda x: x[1]["auc"], reverse=True)

print("Model Performance Ranking (by ROC-AUC):")
print("-" * 60)
print(f"{'Rank':<5} {'Model':<20} {'ROC-AUC':<10} {'CV Score':<12} {'Robustness'}")
print("-" * 60)

for i, (name, results) in enumerate(model_ranking, 1):
    cv_mean = cv_results[name].mean()
    cv_std = cv_results[name].std()
    robustness = "High" if cv_std < 0.02 else "Medium" if cv_std < 0.05 else "Low"

    print(
        f"{i:<5} {name:<20} {results['auc']:<10.4f} {cv_mean:.4f}±{cv_std:.3f} {robustness}"
    )

print("-" * 60)

# Select best model
best_model_name = model_ranking[0][0]
best_model = trained_models[best_model_name]
best_results = val_results[best_model_name]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   • ROC-AUC: {best_results['auc']:.4f}")
print(f"   • Accuracy: {best_results['accuracy']:.4f}")
print(f"   • F1-Score: {best_results['f1']:.4f}")
print(
    f"   • Cross-validation: {cv_results[best_model_name].mean():.4f} ± {cv_results[best_model_name].std():.3f}"
)
print("=" * 50)


# %%
# Feature importance analysis for interpretable models
print("\nFeature Importance Analysis:")
print("=" * 50)

# Extract feature importance from different models
importance_results = {}

# 1. Logistic Regression - Coefficients
if "Logistic Regression" in trained_models:
    lr_model = trained_models["Logistic Regression"]
    lr_coef = lr_model.coef_[0]
    importance_results["Logistic Regression"] = list(zip(feature_names, lr_coef))

# 2. Random Forest - Feature Importance
if "Random Forest" in trained_models:
    rf_model = trained_models["Random Forest"]
    rf_importance = rf_model.feature_importances_
    importance_results["Random Forest"] = list(zip(feature_names, rf_importance))

# 3. Gradient Boosting - Feature Importance
if "Gradient Boosting" in trained_models:
    gb_model = trained_models["Gradient Boosting"]
    gb_importance = gb_model.feature_importances_
    importance_results["Gradient Boosting"] = list(zip(feature_names, gb_importance))

# 4. For models without inherent feature importance (KNN, SVM, Naive Bayes)
# Use permutation importance as a model-agnostic approach
from sklearn.inspection import permutation_importance

non_interpretable_models = ["K-Nearest Neighbors", "SVM", "Naive Bayes"]
for model_name in non_interpretable_models:
    if model_name in trained_models:
        model = trained_models[model_name]
        # Calculate permutation importance on validation set (faster than full training set)
        perm_importance = permutation_importance(
            model, X_val_split, y_val_split, n_repeats=5, random_state=SEED, n_jobs=-1
        )
        importance_results[model_name] = list(
            zip(feature_names, perm_importance.importances_mean)
        )

# Display feature importance for best model
if best_model_name in importance_results:
    print(f"\nTop 15 Most Important Features ({best_model_name}):")
    print("-" * 70)

    best_importance = importance_results[best_model_name]
    if best_model_name == "Logistic Regression":
        # Sort by absolute coefficient value
        best_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"{'Feature':<40} {'Coefficient':<15} {'Abs Value':<10}")
        print("-" * 70)
        for i, (feature, coef) in enumerate(best_importance[:15], 1):
            print(f"{i:2d}. {feature:<35} {coef:8.4f} {abs(coef):8.4f}")
    elif best_model_name in non_interpretable_models:
        # Sort by permutation importance value
        best_importance.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Feature':<40} {'Perm Importance':<15}")
        print("-" * 70)
        for i, (feature, importance) in enumerate(best_importance[:15], 1):
            print(f"{i:2d}. {feature:<35} {importance:8.4f}")
    else:
        # Sort by feature importance value (tree-based models)
        best_importance.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Feature':<40} {'Importance':<15}")
        print("-" * 70)
        for i, (feature, importance) in enumerate(best_importance[:15], 1):
            print(f"{i:2d}. {feature:<35} {importance:8.4f}")

print("=" * 50)


# %% [markdown]
# #### Model Comparison Visualizations
#

# %%
# Create comprehensive model comparison and prediction visualizations
print("\nCreating Model Comparison Visualizations:")
print("=" * 50)

# Create a large figure with multiple subplots for model comparison
fig = plt.figure(figsize=(20, 16))

# 1. Model Performance Comparison (ROC-AUC with CV confidence)
ax1 = plt.subplot(2, 3, 1)
model_names = list(cv_results.keys())
cv_means = [cv_results[name].mean() for name in model_names]
cv_stds = [cv_results[name].std() for name in model_names]
val_aucs = [val_results[name]["auc"] for name in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

# CV scores with error bars
bars1 = ax1.bar(
    x_pos - width / 2,
    cv_means,
    width,
    yerr=cv_stds,
    color=colors[: len(model_names)],
    alpha=0.8,
    label="CV Mean ± Std",
    capsize=5,
    edgecolor="white",
    linewidth=1,
)

# Validation scores
bars2 = ax1.bar(
    x_pos + width / 2,
    val_aucs,
    width,
    color=colors[: len(model_names)],
    alpha=0.6,
    label="Validation AUC",
    edgecolor="white",
    linewidth=1,
)

# Add value labels on bars
for i, (bar1, bar2, mean_val, val_auc) in enumerate(
    zip(bars1, bars2, cv_means, val_aucs)
):
    ax1.text(
        bar1.get_x() + bar1.get_width() / 2,
        bar1.get_height() + cv_stds[i] + 0.01,
        f"{mean_val:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )
    ax1.text(
        bar2.get_x() + bar2.get_width() / 2,
        bar2.get_height() + 0.01,
        f"{val_auc:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )

ax1.set_title(
    "Model Performance Comparison\n(ROC-AUC Scores)", fontweight="bold", pad=15
)
ax1.set_ylabel("ROC-AUC Score", fontweight="bold")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([name.replace(" ", "\n") for name in model_names], fontsize=10)
ax1.legend(loc="lower right", framealpha=0.9)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.set_ylim(0, 1.1)

# 2. Model Robustness (CV Standard Deviation)
ax2 = plt.subplot(2, 3, 2)
robustness_colors = [
    "#2ECC71" if std < 0.02 else "#F39C12" if std < 0.05 else "#E74C3C"
    for std in cv_stds
]

bars = ax2.bar(
    x_pos, cv_stds, color=robustness_colors, alpha=0.8, edgecolor="white", linewidth=2
)

# Add value labels
for bar, std in zip(bars, cv_stds):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{std:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

ax2.set_title("Model Robustness\n(Lower = More Robust)", fontweight="bold", pad=15)
ax2.set_ylabel("Cross-Validation Std Dev", fontweight="bold")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([name.replace(" ", "\n") for name in model_names], fontsize=10)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

# Add robustness legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#2ECC71", label="High (< 0.02)"),
    Patch(facecolor="#F39C12", label="Medium (0.02-0.05)"),
    Patch(facecolor="#E74C3C", label="Low (> 0.05)"),
]
ax2.legend(
    handles=legend_elements, title="Robustness", loc="upper right", framealpha=0.9
)

# 3. Training Time Comparison
ax3 = plt.subplot(2, 3, 3)
training_time_values = [training_times[name] for name in model_names]
time_colors = [
    "#3498DB" if t < 1 else "#F39C12" if t < 10 else "#E74C3C"
    for t in training_time_values
]

bars = ax3.bar(
    x_pos,
    training_time_values,
    color=time_colors,
    alpha=0.8,
    edgecolor="white",
    linewidth=2,
)

# Add value labels
for bar, time_val in zip(bars, training_time_values):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(training_time_values) * 0.02,
        f"{time_val:.1f}s",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=10,
    )

ax3.set_title(
    "Training Time Comparison\n(5-Fold Cross-Validation)", fontweight="bold", pad=15
)
ax3.set_ylabel("Training Time (seconds)", fontweight="bold")
ax3.set_xticks(x_pos)
ax3.set_xticklabels([name.replace(" ", "\n") for name in model_names], fontsize=10)
ax3.grid(axis="y", alpha=0.3, linestyle="--")

# 4. ROC Curves for All Models
ax4 = plt.subplot(2, 3, 4)
for i, (name, results) in enumerate(val_results.items()):
    y_pred_proba = results["probabilities"]
    fpr, tpr, _ = roc_curve(y_val_split, y_pred_proba)
    auc = results["auc"]

    ax4.plot(
        fpr,
        tpr,
        color=colors[i],
        linewidth=2.5,
        alpha=0.8,
        label=f"{name} (AUC = {auc:.3f})",
    )

# Add diagonal line (random classifier)
ax4.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.5, label="Random Classifier")

ax4.set_title("ROC Curves Comparison\n(Validation Set)", fontweight="bold", pad=15)
ax4.set_xlabel("False Positive Rate", fontweight="bold")
ax4.set_ylabel("True Positive Rate", fontweight="bold")
ax4.legend(loc="lower right", framealpha=0.9, fontsize=9)
ax4.grid(alpha=0.3)
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])

# 5. Model Scores Heatmap
ax5 = plt.subplot(2, 3, 5)
metrics = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
scores_matrix = []
for name in model_names:
    results = val_results[name]
    scores = [
        results["auc"],
        results["accuracy"],
        results["precision"],
        results["recall"],
        results["f1"],
    ]
    scores_matrix.append(scores)

scores_df = pd.DataFrame(
    scores_matrix,
    index=[name.replace(" ", "\n") for name in model_names],
    columns=metrics,
)
sns.heatmap(
    scores_df,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    center=0.5,
    ax=ax5,
    cbar_kws={"label": "Score"},
    vmin=0,
    vmax=1,
)
ax5.set_title(
    "Model Performance Heatmap\n(Validation Metrics)", fontweight="bold", pad=15
)
ax5.set_xlabel("Metrics", fontweight="bold")
ax5.set_ylabel("Models", fontweight="bold")

# 6. Best Model Confusion Matrix
ax6 = plt.subplot(2, 3, 6)
best_cm = confusion_matrix(y_val_split, val_results[best_model_name]["predictions"])
best_cm_normalized = best_cm.astype("float") / best_cm.sum(axis=1)[:, np.newaxis]

sns.heatmap(
    best_cm_normalized,
    annot=True,
    fmt=".2%",
    cmap="Blues",
    ax=ax6,
    cbar_kws={"label": "Percentage"},
)
ax6.set_title(
    f"Best Model Confusion Matrix\n({best_model_name})", fontweight="bold", pad=15
)
ax6.set_xlabel("Predicted", fontweight="bold")
ax6.set_ylabel("Actual", fontweight="bold")
ax6.set_xticklabels(["No Upgrade", "Upgrade"])
ax6.set_yticklabels(["No Upgrade", "Upgrade"])

# Add count annotations
for i in range(best_cm.shape[0]):
    for j in range(best_cm.shape[1]):
        ax6.text(
            j + 0.5,
            i + 0.7,
            f"n={best_cm[i, j]}",
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig("graphs/04_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Model comparison plots saved to: graphs/04_model_comparison.png")


# %%
# Best Model Detailed Analysis
print("\nCreating Best Model Analysis Visualizations:")
print("=" * 50)

# Create detailed visualizations for the best model
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

# 1. Feature Importance for Best Model
if best_model_name in importance_results:
    best_importance = importance_results[best_model_name].copy()

    if best_model_name == "Logistic Regression":
        # Sort by absolute coefficient value
        best_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = best_importance[:15]
        feature_names_clean = [
            feat.replace("standardized__", "")
            .replace("categorical__", "")
            .replace("numeric__", "")
            .replace("ordinal__", "")
            for feat, _ in top_features
        ]
        importance_values = [abs(val) for _, val in top_features]
        title_suffix = "(Absolute Coefficients)"
    elif best_model_name in non_interpretable_models:
        # Sort by permutation importance value
        best_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = best_importance[:15]
        feature_names_clean = [
            feat.replace("standardized__", "")
            .replace("categorical__", "")
            .replace("numeric__", "")
            .replace("ordinal__", "")
            for feat, _ in top_features
        ]
        importance_values = [val for _, val in top_features]
        title_suffix = "(Permutation Importance)"
    else:
        # Sort by importance value (tree-based models)
        best_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = best_importance[:15]
        feature_names_clean = [
            feat.replace("standardized__", "")
            .replace("categorical__", "")
            .replace("numeric__", "")
            .replace("ordinal__", "")
            for feat, _ in top_features
        ]
        importance_values = [val for _, val in top_features]
        title_suffix = "(Feature Importance)"

    # Create horizontal bar plot
    colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(importance_values)))
    bars = ax1.barh(
        range(len(importance_values)),
        importance_values,
        color=colors_gradient,
        alpha=0.8,
        edgecolor="white",
        linewidth=1,
    )

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance_values)):
        ax1.text(
            val + max(importance_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontweight="bold",
            fontsize=9,
        )

    ax1.set_title(
        f"Top 15 Most Important Features\n{best_model_name} {title_suffix}",
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlabel("Importance Score", fontweight="bold")
    ax1.set_yticks(range(len(feature_names_clean)))
    ax1.set_yticklabels(feature_names_clean, fontsize=10)
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    ax1.invert_yaxis()

# 2. Prediction Probability Distribution
best_proba = val_results[best_model_name]["probabilities"]
upgrade_proba = best_proba[y_val_split == 1]
no_upgrade_proba = best_proba[y_val_split == 0]

ax2.hist(
    no_upgrade_proba,
    bins=30,
    alpha=0.7,
    color=upgrade_colors[0],
    label=f"No Upgrade (n={len(no_upgrade_proba)})",
    density=True,
    edgecolor="white",
)
ax2.hist(
    upgrade_proba,
    bins=30,
    alpha=0.7,
    color=upgrade_colors[1],
    label=f"Upgrade (n={len(upgrade_proba)})",
    density=True,
    edgecolor="white",
)

ax2.set_title(
    f"Prediction Probability Distribution\n({best_model_name})",
    fontweight="bold",
    pad=15,
)
ax2.set_xlabel("Predicted Probability of Upgrade", fontweight="bold")
ax2.set_ylabel("Density", fontweight="bold")
ax2.legend(framealpha=0.9)
ax2.grid(alpha=0.3, linestyle="--")

# Add vertical line at decision threshold (0.5)
ax2.axvline(x=0.5, color="red", linestyle="--", linewidth=2, alpha=0.8)
ax2.text(
    0.52, ax2.get_ylim()[1] * 0.9, "Decision\nThreshold", color="red", fontweight="bold"
)

# 3. Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_val_split, best_proba)
avg_precision = average_precision_score(y_val_split, best_proba)

ax3.plot(
    recall,
    precision,
    color=colors[0],
    linewidth=3,
    alpha=0.8,
    label=f"{best_model_name}\n(AP = {avg_precision:.3f})",
)
ax3.fill_between(recall, precision, alpha=0.2, color=colors[0])

# Add baseline (random classifier)
baseline = y_val_split.mean()
ax3.axhline(
    y=baseline,
    color="gray",
    linestyle="--",
    alpha=0.8,
    linewidth=2,
    label=f"Random Classifier\n(AP = {baseline:.3f})",
)

ax3.set_title("Precision-Recall Curve\n(Validation Set)", fontweight="bold", pad=15)
ax3.set_xlabel("Recall (True Positive Rate)", fontweight="bold")
ax3.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
ax3.legend(loc="lower left", framealpha=0.9)
ax3.grid(alpha=0.3)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])

# 4. Model Calibration (Reliability Diagram)
from sklearn.calibration import calibration_curve

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_val_split, best_proba, n_bins=10
)

ax4.plot(
    mean_predicted_value,
    fraction_of_positives,
    "s-",
    color=colors[0],
    linewidth=2,
    markersize=8,
    alpha=0.8,
    label=f"{best_model_name}",
)
ax4.plot([0, 1], [0, 1], "k--", alpha=0.8, linewidth=2, label="Perfectly Calibrated")

# Fill area between perfect calibration and actual
ax4.fill_between(
    mean_predicted_value,
    fraction_of_positives,
    mean_predicted_value,
    alpha=0.3,
    color=colors[0],
)

ax4.set_title("Model Calibration\n(Reliability Diagram)", fontweight="bold", pad=15)
ax4.set_xlabel("Mean Predicted Probability", fontweight="bold")
ax4.set_ylabel("Fraction of Positives", fontweight="bold")
ax4.legend(framealpha=0.9)
ax4.grid(alpha=0.3)
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])

# Add calibration score annotation
from sklearn.metrics import brier_score_loss

brier_score = brier_score_loss(y_val_split, best_proba)
ax4.text(
    0.02,
    0.98,
    f"Brier Score: {brier_score:.4f}\n(Lower is Better)",
    transform=ax4.transAxes,
    va="top",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

plt.tight_layout()
plt.savefig("graphs/05_best_model_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Best model analysis plots saved to: graphs/05_best_model_analysis.png")

# Generate insights summary
print("\n" + "=" * 70)
print("VISUALIZATION INSIGHTS SUMMARY")
print("=" * 70)

print(f"\n📈 MODEL COMPARISON INSIGHTS:")
best_cv = max(cv_results.items(), key=lambda x: x[1].mean())
most_robust = min(cv_results.items(), key=lambda x: x[1].std())
fastest = min(training_times.items(), key=lambda x: x[1])

print(f"   • Best CV performance: {best_cv[0]} ({best_cv[1].mean():.4f})")
print(f"   • Most robust model: {most_robust[0]} (std: {most_robust[1].std():.4f})")
print(f"   • Fastest training: {fastest[0]} ({fastest[1]:.2f}s)")

print(f"\n🎯 BEST MODEL ({best_model_name}) INSIGHTS:")
print(f"   • Validation AUC: {val_results[best_model_name]['auc']:.4f}")
print(f"   • Average Precision: {avg_precision:.4f}")
print(
    f"   • Model calibration: {'Well calibrated' if brier_score < 0.25 else 'Needs calibration'}"
)
print(
    f"   • Probability separation: {'Good' if abs(upgrade_proba.mean() - no_upgrade_proba.mean()) > 0.2 else 'Fair'}"
)

if best_model_name in importance_results:
    top_3_features = importance_results[best_model_name][:3]
    if best_model_name == "Logistic Regression":
        top_3_features.sort(key=lambda x: abs(x[1]), reverse=True)
    else:
        # For both permutation importance and tree-based importance, sort by value
        top_3_features.sort(key=lambda x: x[1], reverse=True)

    print(f"   • Top 3 predictive features:")
    for i, (feature, _) in enumerate(top_3_features[:3], 1):
        clean_feature = (
            feature.replace("standardized__", "")
            .replace("categorical__", "")
            .replace("numeric__", "")
            .replace("ordinal__", "")
        )
        print(f"     {i}. {clean_feature}")

print("=" * 70)


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
test_precision = test_report["1"]["precision"]
test_recall = test_report["1"]["recall"]
test_f1 = test_report["1"]["f1-score"]

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
print(f"No Upgrade    {test_cm[0, 0]:4d}     {test_cm[0, 1]:4d}")
print(f"Upgrade       {test_cm[1, 0]:4d}     {test_cm[1, 1]:4d}")

print("-" * 50)
print("=" * 50)


# %%
# Final Test Results Visualization
print("\nCreating Final Test Results Visualization:")
print("=" * 50)

# Create final test results visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Performance Metrics Comparison (Validation vs Test)
metrics = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
val_scores = [
    val_results[best_model_name]["auc"],
    val_results[best_model_name]["accuracy"],
    val_results[best_model_name]["precision"],
    val_results[best_model_name]["recall"],
    val_results[best_model_name]["f1"],
]
test_scores = [test_auc, test_accuracy, test_precision, test_recall, test_f1]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(
    x_pos - width / 2,
    val_scores,
    width,
    label="Validation",
    color=colors[0],
    alpha=0.8,
    edgecolor="white",
    linewidth=1,
)
bars2 = ax1.bar(
    x_pos + width / 2,
    test_scores,
    width,
    label="Test",
    color=colors[1],
    alpha=0.8,
    edgecolor="white",
    linewidth=1,
)

# Add value labels
for i, (bar1, bar2, val_score, test_score) in enumerate(
    zip(bars1, bars2, val_scores, test_scores)
):
    ax1.text(
        bar1.get_x() + bar1.get_width() / 2,
        bar1.get_height() + 0.01,
        f"{val_score:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )
    ax1.text(
        bar2.get_x() + bar2.get_width() / 2,
        bar2.get_height() + 0.01,
        f"{test_score:.3f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=9,
    )

ax1.set_title(
    f"Final Model Performance\n{best_model_name} (Validation vs Test)",
    fontweight="bold",
    pad=15,
)
ax1.set_ylabel("Score", fontweight="bold")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(metrics, rotation=45, ha="right")
ax1.legend(framealpha=0.9)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.set_ylim(0, 1.1)

# 2. Test Set Confusion Matrix (Enhanced)
test_cm = confusion_matrix(y_test, test_pred)
test_cm_normalized = test_cm.astype("float") / test_cm.sum(axis=1)[:, np.newaxis]

# Create custom colormap
from matplotlib.colors import LinearSegmentedColormap

colors_cm = ["white", colors[0]]
cm_custom = LinearSegmentedColormap.from_list("custom", colors_cm)

sns.heatmap(
    test_cm_normalized,
    annot=True,
    fmt=".2%",
    cmap=cm_custom,
    ax=ax2,
    cbar_kws={"label": "Percentage"},
    vmin=0,
    vmax=1,
)

# Add count annotations
for i in range(test_cm.shape[0]):
    for j in range(test_cm.shape[1]):
        ax2.text(
            j + 0.5,
            i + 0.3,
            f"n={test_cm[i, j]}",
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            fontweight="bold",
        )

ax2.set_title(
    f"Final Test Confusion Matrix\n{best_model_name}", fontweight="bold", pad=15
)
ax2.set_xlabel("Predicted", fontweight="bold")
ax2.set_ylabel("Actual", fontweight="bold")
ax2.set_xticklabels(["No Upgrade", "Upgrade"])
ax2.set_yticklabels(["No Upgrade", "Upgrade"])

# 3. Test Set ROC Curve
test_fpr, test_tpr, _ = roc_curve(y_test, test_pred_proba)
ax3.plot(
    test_fpr,
    test_tpr,
    color=colors[0],
    linewidth=3,
    alpha=0.8,
    label=f"{best_model_name}\n(AUC = {test_auc:.3f})",
)
ax3.fill_between(test_fpr, test_tpr, alpha=0.2, color=colors[0])

# Add diagonal line (random classifier)
ax3.plot([0, 1], [0, 1], "k--", alpha=0.8, linewidth=2, label="Random Classifier")

ax3.set_title("Final Test ROC Curve", fontweight="bold", pad=15)
ax3.set_xlabel("False Positive Rate", fontweight="bold")
ax3.set_ylabel("True Positive Rate", fontweight="bold")
ax3.legend(loc="lower right", framealpha=0.9)
ax3.grid(alpha=0.3)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])

# 4. Prediction Probability Distribution (Test Set)
test_upgrade_proba = test_pred_proba[y_test == 1]
test_no_upgrade_proba = test_pred_proba[y_test == 0]

ax4.hist(
    test_no_upgrade_proba,
    bins=25,
    alpha=0.7,
    color=upgrade_colors[0],
    label=f"No Upgrade (n={len(test_no_upgrade_proba)})",
    density=True,
    edgecolor="white",
)
ax4.hist(
    test_upgrade_proba,
    bins=25,
    alpha=0.7,
    color=upgrade_colors[1],
    label=f"Upgrade (n={len(test_upgrade_proba)})",
    density=True,
    edgecolor="white",
)

ax4.set_title(
    f"Test Set Probability Distribution\n{best_model_name}", fontweight="bold", pad=15
)
ax4.set_xlabel("Predicted Probability of Upgrade", fontweight="bold")
ax4.set_ylabel("Density", fontweight="bold")
ax4.legend(framealpha=0.9)
ax4.grid(alpha=0.3, linestyle="--")

# Add vertical line at decision threshold (0.5)
ax4.axvline(x=0.5, color="red", linestyle="--", linewidth=2, alpha=0.8)
ax4.text(
    0.52, ax4.get_ylim()[1] * 0.9, "Decision\nThreshold", color="red", fontweight="bold"
)

# Add mean probabilities
mean_upgrade = test_upgrade_proba.mean()
mean_no_upgrade = test_no_upgrade_proba.mean()
ax4.axvline(
    x=mean_upgrade, color=upgrade_colors[1], linestyle=":", alpha=0.8, linewidth=2
)
ax4.axvline(
    x=mean_no_upgrade, color=upgrade_colors[0], linestyle=":", alpha=0.8, linewidth=2
)

plt.tight_layout()
plt.savefig("graphs/06_final_test_results.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Final test results plots saved to: graphs/06_final_test_results.png")

# Final test summary
print("\n" + "=" * 70)
print("FINAL TEST RESULTS SUMMARY")
print("=" * 70)

print(f"\n🏆 FINAL MODEL: {best_model_name}")
print(f"   • Test ROC-AUC: {test_auc:.4f}")
print(f"   • Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.1%})")
print(f"   • Test Precision: {test_precision:.4f}")
print(f"   • Test Recall: {test_recall:.4f}")
print(f"   • Test F1-Score: {test_f1:.4f}")

print(f"\n📊 PROBABILITY ANALYSIS:")
print(f"   • Mean probability for upgrades: {test_upgrade_proba.mean():.3f}")
print(f"   • Mean probability for no upgrades: {test_no_upgrade_proba.mean():.3f}")
print(
    f"   • Separation: {abs(test_upgrade_proba.mean() - test_no_upgrade_proba.mean()):.3f}"
)

print(f"\n🎯 CONFUSION MATRIX BREAKDOWN:")
tn, fp, fn, tp = test_cm.ravel()
print(f"   • True Negatives (Correct No-Upgrade): {tn}")
print(f"   • False Positives (Incorrect Upgrade): {fp}")
print(f"   • False Negatives (Missed Upgrade): {fn}")
print(f"   • True Positives (Correct Upgrade): {tp}")

print(f"\n💼 BUSINESS METRICS:")
print(f"   • Total test customers: {len(y_test)}")
print(f"   • Actual upgrades: {y_test.sum()} ({y_test.mean():.1%})")
print(f"   • Predicted upgrades: {test_pred.sum()}")
print(f"   • Correctly identified upgrades: {tp}")
print(
    f"   • Marketing efficiency: {tp / max(test_pred.sum(), 1) * 100:.1f}% (precision)"
)

print("=" * 70)


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
print(
    f"   • Actual upgrades: {actual_upgrades} ({actual_upgrades / total_customers:.1%})"
)
print(f"   • Predicted upgrades: {predicted_upgrades}")
print(f"   • Correctly identified upgrades: {correctly_identified}")
print("   • Potential revenue impact: High (targeted marketing efficiency)")

print("\n🎯 KEY RECOMMENDATIONS:")
if best_model_name in importance_results:
    top_features = importance_results[best_model_name]
    if best_model_name == "Logistic Regression":
        top_features.sort(key=lambda x: abs(x[1]), reverse=True)
    else:
        # For both permutation importance and tree-based importance, sort by value
        top_features.sort(key=lambda x: x[1], reverse=True)

    print("   • Focus on top predictive features:")
    for i, (feature, _) in enumerate(top_features[:3], 1):
        clean_feature = (
            feature.replace("standardized__", "")
            .replace("categorical__", "")
            .replace("numeric__", "")
            .replace("ordinal__", "")
        )
        print(f"     {i}. {clean_feature}")

print("   • Implement targeted retention strategies")
print("   • Use model for customer segmentation")
print("   • Monitor model performance quarterly")

print("=" * 60)
