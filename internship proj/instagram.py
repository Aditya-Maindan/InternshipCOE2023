import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

df = pd.read_csv('/Users/adityamaindan/Desktop/internship proj/instagram.csv')



# First few rows of the DataFrame
print("First Few Rows of the DataFrame:")
print(df.head(10)) 
print("\n\nLast 10 rows of DataFrame:") 
print(df.tail(10))

print("\n\n Number of rows and columns:")
print(df.shape)

print("\n\n Data type of each column:")
print(df.dtypes)

print("\n\n Checking the presence of missing values:")
print(df.isnull().sum())

print("\n\n Presence of Outliers:")
df.boxplot(column=['nums/length username'])
plt.show()

df.boxplot(column=['fullname words'])
plt.show()

df.boxplot(column=['description length'])
plt.show()

df.boxplot(column=['#posts'])
plt.show()

df.boxplot(column=['#followers'])
plt.show()

df.boxplot(column=['#follows'])
plt.show()

# median = np.median('#follows')
# print(f"Median of follows: {median}")

# # Calculate the IQR (robust measure of dispersion)
# q1 = np.percentile(df, 25)
# q3 = np.percentile(df, 75)
# iqr = q3 - q1
# print(f"IQR: {iqr}")

# # Identify potential outliers using the IQR method
# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr

# column_name = '#follows'
# column = df[column_name]

# outliers = column[(column < lower_bound) | (column > upper_bound)]
# print("Outliers:")
# print(outliers)

# You can also use robust statistical functions from the scipy library
# For example, to calculate the median and IQR using scipy
# median_scipy = stats.median(df)
# iqr_scipy = stats.iqr(df)

# print(f"Median (Scipy): {median_scipy}")
# print(f"IQR (Scipy): {iqr_scipy}")



# Create a figure with subplots for each variable
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
fig.suptitle("Histograms for Variables")

# Plot histograms for each variable
df['profile pic'].plot(kind='hist', ax=axes[0, 0], bins=2)
df['nums/length username'].plot(kind='hist', ax=axes[0, 1], bins=20)
df['fullname words'].plot(kind='hist', ax=axes[0, 2], bins=20)
df['nums/length fullname'].plot(kind='hist', ax=axes[0, 3], bins=20)
df['name==username'].plot(kind='hist', ax=axes[1, 0], bins=2)
df['description length'].plot(kind='hist', ax=axes[1, 1], bins=20)
df['external URL'].plot(kind='hist', ax=axes[1, 2], bins=2)
df['private'].plot(kind='hist', ax=axes[1, 3], bins=2)
df['#posts'].plot(kind='hist', ax=axes[2, 0], bins=20)
df['#followers'].plot(kind='hist', ax=axes[2, 1], bins=20)
df['#follows'].plot(kind='hist', ax=axes[2, 2], bins=20)
df['fake'].plot(kind='hist', ax=axes[2, 3], bins=2)

# Set titles for each subplot
axes[0, 0].set_title('Profile Pic')
axes[0, 1].set_title('Num/Length Username')
axes[0, 2].set_title('Full Name Words')
axes[0, 3].set_title('Num/Length Full Name')
axes[1, 0].set_title('Name == Username')
axes[1, 1].set_title('Description Length')
axes[1, 2].set_title('External URL')
axes[1, 3].set_title('Private')
axes[2, 0].set_title('# Posts')
axes[2, 1].set_title('# Followers')
axes[2, 2].set_title('# Follows')
axes[2, 3].set_title('Fake')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Show the plot
plt.show()

print("\n\nDescibe the dataframe:")
print(df.describe())

profile_pic_counts = df['profile pic'].value_counts()
username_counts = df['nums/length username'].value_counts()
full_name_words_counts = df['fullname words'].value_counts()
full_name_length_counts = df['nums/length fullname'].value_counts()
name_equals_username_counts = df['name==username'].value_counts()
description_length_counts = df['description length'].value_counts()
external_url_counts = df['external URL'].value_counts()
private_counts = df['private'].value_counts()
posts_counts = df['#posts'].value_counts()
followers_counts = df['#followers'].value_counts()
follows_counts = df['#follows'].value_counts()
fake_counts = df['fake'].value_counts()

# print("Profile Pic counts:")
# print(profile_pic_counts)

# print("Num/Length Username counts:")
# print(username_counts)

print("Full Name Words counts:")
print(full_name_words_counts)

# print("Num/Length Full Name counts:")
# print(full_name_length_counts)

# print("Name == Username counts:")
# print(name_equals_username_counts)

# print("Description Length counts:")
# print(description_length_counts)

# print("External URL counts:")
# print(external_url_counts)

# print("Private counts:")
# print(private_counts)

# print("# Posts counts:")
# print(posts_counts)

# print("# Followers counts:")
# print(followers_counts)

# print("# Follows counts:")
# print(follows_counts)

# print("Fake counts:")
# print(fake_counts)

numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()