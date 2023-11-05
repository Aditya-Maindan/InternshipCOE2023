# A DataFrame is a data structure that organizes data into a 2-dimensional table of rows and columns, much like a spreadsheet.
# DataFrames are widely used in data science, machine learning, scientific computing, and many other data-intensive fields. 

import pandas as pd

# Create a dictionary of data
data = {
    'Name': ['Neha', 'Sanjay', 'Bhoomika', 'Murshid','Aditya'],
    'Age': [20, 35, 19, 25 ,22],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston','Mangalore']
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)


print(df)
print ("\n\n")

print("Adding rows and columns to a DataFrame")

new_row = {'Name': 'Rohan', 'Age': 24, 'City': 'Bangalore'}


df = df._append(new_row, ignore_index=True)

df['Gender'] = ['Female', 'Male', 'Female', 'Male', 'Male','Male']
print(df)

print("\n \n Removing rows and columns from a DataFrame")
del df['Gender']
df = df.drop(5)
print(df)

print("\n\n Iterating over rows and columns in a DataFrame")
# Iterate over rows
for index, row in df.iterrows():
    print(f"Row {index}: {row['Name']}, {row['Age']}, {row['City']}")

# Iterate over columns
for column_name in df:
    column_data = df[column_name]
    print(f"Column {column_name}:")
    for index, value in column_data.items():
        print(f"    Row {index}: {value}")

print("\n\n Indexting the dataframe")
df = df.set_index('Name')

print(df)

data = {
    'Name': ['Neha', 'Sanjay', 'Bhoomika', 'Murshid', 'Aditya'],
    'Age': [20, 35, 19, 25, 22],
    'Marks': [90, 80, 70, None, None],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Mangalore']
}

# Create a DataFrame
df = pd.DataFrame(data)

print("\n\n handling missing data in a DataFrame")

#df.dropna(subset=['Marks'], inplace=True)


#df['Marks'].fillna(0, inplace=True)


df['Marks'].fillna(df['Marks'].mean(), inplace=True)


print(df)

# print("\n\n Selecting data with male genderfrom a DataFrame")

# # Select a single row by label
# selected_row = df.loc[:,]

# # Select rows that meet a condition (e.g., Age > 30)
# selected_rows = df[df['Gender'] =="Female"]

# Univariate analysis is a statistical and data analysis technique that focuses on examining and summarizing the characteristics and patterns within a single variable or attribute in a dataset

