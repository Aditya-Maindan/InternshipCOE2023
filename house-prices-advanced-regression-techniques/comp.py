import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
df = pd.read_csv('/Users/adityamaindan/Desktop/house-prices-advanced-regression-techniques/train.csv')
db = pd.read_csv('/Users/adityamaindan/Desktop/house-prices-advanced-regression-techniques/test.csv')
print("Train dataset shape is: ")
print(df.shape)

print("\nFew rows of train dataset are: ")
print(df.head())

print('\nInformation about train dataset is: ')
print(df.info())

print('\nStatistical information about train dataset is: ')
print(df['SalePrice'].describe())

print("\nNumber of Missing Values in Each Column:")
print(df.isnull().sum())

print("\n Filling missing values in Lotfrontage:")
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
print(df)

# column_name = 'MSSubClass'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'LotFrontage'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'LotArea'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'OverallQual'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'OverallCond'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()tr

# column_name = 'YearBuilt'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'YearRemodAdd'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'MasVnrArea'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'BsmtFinSF1'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'BsmtFinSF2'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'BsmtUnfSF'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'TotalBsmtSF'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()

# column_name = 'BsmtFinSF2'
# plt.boxplot(df[column_name], vert=False)
# plt.title(f'Box Plot for {column_name}')
# plt.show()
df.rename(columns={'1stFlrSF': 'onestFlrSF'}, inplace=True)
db.rename(columns={'1stFlrSF': 'onestFlrSF'}, inplace=True)
numeric_cols = df.select_dtypes(include = ['number'])
corr = numeric_cols.corr()
print(corr['SalePrice'].sort_values(ascending=True))

lm = smf.ols(formula='SalePrice ~ GarageCars+GrLivArea+OverallQual+YearBuilt+TotalBsmtSF+onestFlrSF',data=df).fit()
print("\n\n")
print(lm.params)

print("\n\n Summary")
print(lm.summary())

saleprice=lm.predict(db)
print(saleprice)
db['SalePrice'] = saleprice
print(db)

db.to_csv('/Users/adityamaindan/Desktop/house-prices-advanced-regression-techniques/submission.csv', index=False)





