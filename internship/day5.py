import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/Users/adityamaindan/Desktop/internship/onlinefraud.csv')

condition = df['amount'] > 100000
filtered_data = df[condition]
print("Rows where amount > 100000:")
print(filtered_data)

print("Renaming the column 'oldbalanceOrg' to 'oldbalanceOriginal':")
df.rename(columns={'oldbalanceOrg': 'oldbalanceOriginal'}, inplace=True)

# Print the DataFrame to verify the column name change
print(df)

numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# x_column = 'step'
# y_column = 'amount'
# plt.scatter(df[x_column], df[y_column])
# plt.title(f'Scatter Plot for {x_column} vs. {y_column}')
# plt.show()

# x_column = 'step'
# y_column = 'oldbalanceOriginal'
# plt.scatter(df[x_column], df[y_column])
# plt.title(f'Scatter Plot for {x_column} vs. {y_column}')
# plt.show()

# x_column = 'step'
# y_column = 'newbalanceOrig'
# plt.scatter(df[x_column], df[y_column])
# plt.title(f'Scatter Plot for {x_column} vs. {y_column}')
# plt.show()

# x_column = 'step'
# y_column = 'oldbalanceDest'
# plt.scatter(df[x_column], df[y_column])
# plt.title(f'Scatter Plot for {x_column} vs. {y_column}')
# plt.show()

# x_column = 'step'
# y_column = 'newbalanceDest'
# plt.scatter(df[x_column], df[y_column])
# plt.title(f'Scatter Plot for {x_column} vs. {y_column}')
# plt.show()

df.boxplot(column=['amount'])
plt.show()

df.boxplot(column=['oldbalanceOriginal'])
plt.show()

df.boxplot(column=['newbalanceOrig'])
plt.show()

df.boxplot(column=['oldbalanceDest'])
plt.show()

df.boxplot(column=['newbalanceDest'])
plt.show()


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Fraud or not'] = label_encoder.fit_transform(df['isFraud'])

print(df)