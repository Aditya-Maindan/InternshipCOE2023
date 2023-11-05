
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a DataFrame with your data
data = pd.read_csv('/Users/adityamaindan/Desktop/internship proj/instagram.csv')  # Replace 'your_data.csv' with your file path

# Plot histograms for each variable
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed



for col in data.columns:
    plt.subplot(3, 4, data.columns.get_loc(col) + 1)  # Adjust the subplot layout as needed
    sns.histplot(data[col], kde=True, bins=30)  # You can change the number of bins
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()


