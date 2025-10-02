import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/Amazon_Musical.csv')

# Count unique customer_ids
unique_customers = df['customer_id'].nunique()
print(f"Number of unique customer_ids: {unique_customers:,}")

# Get the distribution of customer_id occurrences
customer_counts = df['customer_id'].value_counts()

print(f"\nTotal number of reviews: {len(df):,}")

# Count customers with more than 1 review
customers_with_multiple_reviews = (customer_counts > 1).sum()
percentage_multiple = (customers_with_multiple_reviews / unique_customers) * 100

# Count customers with more than 10 reviews
customers_with_10plus_reviews = (customer_counts > 10).sum()
percentage_10plus = (customers_with_10plus_reviews / unique_customers) * 100

print(f"\nCustomers with more than 1 review: {customers_with_multiple_reviews:,} ({percentage_multiple:.2f}%)")
print(f"Customers with exactly 1 review: {(customer_counts == 1).sum():,} ({100-percentage_multiple:.2f}%)")
print(f"Customers with more than 10 reviews: {customers_with_10plus_reviews:,} ({percentage_10plus:.2f}%)")

print("\nDistribution statistics:")
print(f"  Mean reviews per customer: {customer_counts.mean():.2f}")
print(f"  Median reviews per customer: {customer_counts.median():.0f}")
print(f"  Max reviews by a single customer: {customer_counts.max():,}")
print(f"  Min reviews by a single customer: {customer_counts.min():,}")

# Show top customers
print("\nTop 10 customers by number of reviews:")
print(customer_counts.head(10))

# Distribution of occurrence frequencies
occurrence_dist = customer_counts.value_counts().sort_index()
print("\n\nDistribution of number of occurrences:")
print(f"{'# of Reviews':<15} {'# of Customers':<15}")
print("-" * 30)
for num_reviews, num_customers in occurrence_dist.head(20).items():
    print(f"{num_reviews:<15} {num_customers:<15,}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer ID Distribution Analysis', fontsize=16, fontweight='bold')

# Plot 1: Distribution of reviews per customer (log scale)
axes[0, 0].hist(customer_counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].set_xlabel('Number of Reviews per Customer', fontsize=11)
axes[0, 0].set_ylabel('Number of Customers', fontsize=11)
axes[0, 0].set_title('Distribution of Reviews per Customer', fontweight='bold')
axes[0, 0].set_yscale('log')
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Top 20 customers
top_20 = customer_counts.head(20)
axes[0, 1].barh(range(len(top_20)), top_20.values, color='coral', edgecolor='black')
axes[0, 1].set_yticks(range(len(top_20)))
axes[0, 1].set_yticklabels([f'Customer {i+1}' for i in range(len(top_20))], fontsize=9)
axes[0, 1].set_xlabel('Number of Reviews', fontsize=11)
axes[0, 1].set_title('Top 20 Most Active Customers', fontweight='bold')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(axis='x', alpha=0.3)

# Plot 3: Cumulative distribution
sorted_counts = customer_counts.sort_values(ascending=False)
cumulative_pct = (sorted_counts.cumsum() / sorted_counts.sum() * 100)
axes[1, 0].plot(range(len(cumulative_pct)), cumulative_pct.values, color='green', linewidth=2)
axes[1, 0].set_xlabel('Number of Customers (sorted by activity)', fontsize=11)
axes[1, 0].set_ylabel('Cumulative % of Reviews', fontsize=11)
axes[1, 0].set_title('Cumulative Distribution of Reviews', fontweight='bold')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
axes[1, 0].legend()

# Plot 4: Occurrence frequency distribution (limited to first 20)
occ_dist_plot = occurrence_dist.head(20)
axes[1, 1].bar(occ_dist_plot.index.astype(str), occ_dist_plot.values, 
               color='mediumpurple', edgecolor='black', alpha=0.8)
axes[1, 1].set_xlabel('Number of Reviews', fontsize=11)
axes[1, 1].set_ylabel('Number of Customers', fontsize=11)
axes[1, 1].set_title('Frequency of Review Counts (1-20 reviews)', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('customer_id_distribution.png', dpi=300, bbox_inches='tight')
print("\n\nVisualization saved as 'customer_id_distribution.png'")
plt.show()

