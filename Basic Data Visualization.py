import pandas as pd
import matplotlib.pyplot as plt

# Define column names
columns = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']

# Load CSV with column names
data = pd.read_csv("banknote_data.csv", names=columns)

# Example scatter plot
plt.scatter(data['kurtosis'], data['entropy'], c='green', alpha=0.5)
plt.xlabel('Kurtosis')
plt.ylabel('Entropy')
plt.title('Kurtosis vs Entropy')
plt.show()