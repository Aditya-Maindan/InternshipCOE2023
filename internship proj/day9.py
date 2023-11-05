import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import statsmodels.api as sm

df = pd.read_csv('/Users/adityamaindan/Desktop/internship proj/instagram.csv')

dependent_variable = df['fake']
independent_variables = df[['profile pic', 'nums/length username','#posts', '#followers', '#follows']]




X_train, X_test, y_train, y_test = train_test_split(independent_variables, dependent_variable, test_size=0.2, random_state=42)

# Assuming you have your training and test data split into X_train, X_test, y_train, and y_test

# Calculate the proportions
proportion_train = len(X_train) / (len(X_train) + len(X_test))
proportion_test = len(X_test) / (len(X_train) + len(X_test))




# Print the proportions
print(f"Proportion of Training Data: {proportion_train:.2%}")
print(f"Proportion of Test Data: {proportion_test:.2%}")

train_indices = X_train.index
train_data = df.loc[train_indices]
print(train_data)

test_indices = X_test.index
test_data = df.loc[test_indices]
print(test_data)

model = LogisticRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate a classification report
print(classification_report(y_test, y_pred))

#Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)



independent_variables= sm.add_constant(independent_variables)

# Create and fit the Logistic Regression model
model = sm.Logit(dependent_variable, independent_variables)
result = model.fit()
print(result.summary())



