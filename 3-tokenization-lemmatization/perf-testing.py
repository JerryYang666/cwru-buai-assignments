import time
from tqdm import tqdm
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords

df = pd.read_csv('data/Amazon_Musical.csv')
print(f"Dataset shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst few rows of review_body:")
print(df['review_body'].head())

# Download NLTK stopwords
nltk.download('stopwords')

# Load English stopwords from NLTK into a Python set
stop_words = set(stopwords.words('english'))

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Sample 5% of data
df_sample = df.sample(frac=0.05, random_state=42).copy()
print(f"Testing on {len(df_sample)} documents (5% sample)")

# Define configurations to test
configs = [
    {'n_process': 1, 'batch_size': 1},
    {'n_process': 1, 'batch_size': 50},
    {'n_process': 2, 'batch_size': 30},
    {'n_process': 2, 'batch_size': 64},
    {'n_process': 2, 'batch_size': 128},
    {'n_process': 4, 'batch_size': 50},
    {'n_process': 4, 'batch_size': 64},
    {'n_process': 4, 'batch_size': 128},
    {'n_process': 4, 'batch_size': 256},
    {'n_process': 8, 'batch_size': 128},
    {'n_process': 8, 'batch_size': 256},
    {'n_process': 48, 'batch_size': 100},
    {'n_process': 48, 'batch_size': 200},
    {'n_process': 48, 'batch_size': 256},
    {'n_process': 48, 'batch_size': 512},
    {'n_process': 72, 'batch_size': 256},
    {'n_process': 72, 'batch_size': 512},
    {'n_process': 72, 'batch_size': 800},
]

# Store results
results = []

# Prepare texts once (same for all tests)
print("\n" + "="*80)
print("PREPARING TEXTS (one-time)")
print("="*80)
texts = df_sample['token_regex_ver2'].apply(lambda x: ' '.join(x)).tolist()
print(f"Prepared {len(texts)} texts\n")

# Test each configuration
for i, config in enumerate(configs, 1):
    n_proc = config['n_process']
    batch = config['batch_size']
    
    print("="*80)
    print(f"TEST {i}/{len(configs)}: n_process={n_proc}, batch_size={batch}")
    print("="*80)
    
    try:
        # Timing for spaCy processing
        processing_start = time.time()
        
        docs = list(tqdm(
            nlp.pipe(texts, batch_size=batch, n_process=n_proc),
            total=len(texts),
            desc=f"Processing (n_proc={n_proc}, batch={batch})"
        ))
        
        processing_time = time.time() - processing_start
        
        # Timing for lemma extraction
        extract_start = time.time()
        
        lemmas = [
            [token.lemma_.lower() for token in doc 
             if token.is_alpha and token.lemma_.lower() not in stop_words]
            for doc in docs
        ]
        
        extract_time = time.time() - extract_start
        total_time = processing_time + extract_time
        
        # Store results
        results.append({
            'n_process': n_proc,
            'batch_size': batch,
            'processing_time': processing_time,
            'extract_time': extract_time,
            'total_time': total_time,
            'docs_per_sec': len(texts) / total_time,
            'avg_time_per_doc': total_time / len(texts),
            'status': 'SUCCESS'
        })
        
        print(f"✓ Processing: {processing_time:.2f}s")
        print(f"✓ Extraction: {extract_time:.2f}s")
        print(f"✓ Total: {total_time:.2f}s ({len(texts)/total_time:.2f} docs/sec)")
        print()
        
    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        results.append({
            'n_process': n_proc,
            'batch_size': batch,
            'processing_time': None,
            'extract_time': None,
            'total_time': None,
            'docs_per_sec': None,
            'avg_time_per_doc': None,
            'status': f'FAILED: {str(e)}'
        })
        print()

# Generate report
print("\n" + "="*80)
print("PERFORMANCE REPORT")
print("="*80)

results_df = pd.DataFrame(results)

# Filter successful runs
successful = results_df[results_df['status'] == 'SUCCESS'].copy()

if len(successful) > 0:
    # Sort by total time (fastest first)
    successful_sorted = successful.sort_values('total_time')
    
    print("\n--- TOP 5 FASTEST CONFIGURATIONS ---")
    print(successful_sorted[['n_process', 'batch_size', 'total_time', 'docs_per_sec']].head().to_string(index=False))
    
    print("\n--- DETAILED RESULTS (sorted by speed) ---")
    for idx, row in successful_sorted.iterrows():
        print(f"\nn_process={row['n_process']:2d}, batch_size={row['batch_size']:3d} | "
              f"Total: {row['total_time']:6.2f}s | "
              f"Processing: {row['processing_time']:6.2f}s | "
              f"Extract: {row['extract_time']:5.2f}s | "
              f"Speed: {row['docs_per_sec']:6.2f} docs/sec")
    
    # Find best configuration
    best = successful_sorted.iloc[0]
    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATION")
    print("="*80)
    print(f"n_process: {best['n_process']}")
    print(f"batch_size: {best['batch_size']}")
    print(f"Total time: {best['total_time']:.2f} seconds")
    print(f"Processing speed: {best['docs_per_sec']:.2f} docs/sec")
    print(f"Estimated time for full dataset: {best['total_time'] * 100 / 60:.2f} minutes")
    
    # Speedup comparison
    slowest = successful_sorted.iloc[-1]
    speedup = slowest['total_time'] / best['total_time']
    print(f"Speedup vs slowest config: {speedup:.2f}x faster")
    print("="*80)
else:
    print("\n⚠ No successful runs to report")

# Show any failures
failed = results_df[results_df['status'] != 'SUCCESS']
if len(failed) > 0:
    print("\n--- FAILED CONFIGURATIONS ---")
    print(failed[['n_process', 'batch_size', 'status']].to_string(index=False))

# Save results to CSV
results_df.to_csv('spacy_performance_test.csv', index=False)
print(f"\n📊 Full results saved to: spacy_performance_test.csv")
