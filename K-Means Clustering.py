# Banknote K-Means Clustering (Robust Version)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
data = pd.read_csv("banknote_data.csv")

# Step 2: Clean column names (remove spaces)
data.columns = data.columns.str.strip()

print("Detected Columns:", data.columns)

# Step 3: If dataset has no proper headers (numbers appear as column names)
# This handles CSV files that don't include headers
if data.columns[0].replace('.', '', 1).isdigit():
    columns = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']
    data = pd.read_csv("banknote_data.csv", names=columns, header=None)
    print("Header was missing. Assigned correct column names.")

# Step 4: Fix possible spelling issue (curtosis vs kurtosis)
if 'curtosis' in data.columns:
    data.rename(columns={'curtosis': 'kurtosis'}, inplace=True)

# Step 5: Final column cleanup
data.columns = data.columns.str.strip()

print("Final Columns Used:", data.columns)

# Step 6: Select features
X = data[['variance', 'skewness', 'kurtosis', 'entropy']]

# Convert to numeric (in case first row had text)
X = X.apply(pd.to_numeric, errors='coerce')

# Drop rows with any invalid values
X = X.dropna()

# Step 7: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Run K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_scaled)

data = data.loc[X.index]  # keep aligned rows
data['cluster'] = kmeans.labels_

# Step 9: Single Run Visualization
plt.figure(figsize=(8,6))
plt.scatter(data['kurtosis'], data['entropy'],
            c=data['cluster'], cmap='viridis', alpha=0.6)

plt.xlabel('Kurtosis')
plt.ylabel('Entropy')
plt.title('K-Means Clustering of Banknotes')
plt.show()

# Step 10: Multiple Runs Comparison
fig, axes = plt.subplots(1, 3, figsize=(18,5))

for i in range(3):
    kmeans = KMeans(n_clusters=2, random_state=i*10, n_init=10)
    kmeans.fit(X_scaled)
    data['cluster'] = kmeans.labels_

    axes[i].scatter(data['kurtosis'], data['entropy'],
                    c=data['cluster'], cmap='viridis', alpha=0.6)

    axes[i].set_xlabel('Kurtosis')
    axes[i].set_ylabel('Entropy')
    axes[i].set_title(f'Run {i+1}')

plt.tight_layout()
plt.show()

fig.savefig("kmeans_banknotes_comparison.png")

print("\nClustering completed successfully.")