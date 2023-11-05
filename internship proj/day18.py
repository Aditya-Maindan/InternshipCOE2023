from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/Users/adityamaindan/Desktop/internship proj/instagram.csv')
y = df['fake']
X = df[['profile pic', 'nums/length username', '#posts', '#followers', '#follows']]

k = 12

# Create a KFold object
kf = KFold(n_splits=k)

# Initialize a list to store accuracy scores
accuracy_scores = []

# Convert DataFrame to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()

# Loop through the K-fold splits
for train_index, test_index in kf.split(X):
    # Split the data into train and test sets for this iteration
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create and train a classifier (e.g., Decision Tree)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate accuracy for this fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate the average accuracy across all folds
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)

# Print the accuracy scores for each fold and the average accuracy
print("Accuracy Scores for Each Fold:", accuracy_scores)
print("Average Accuracy:", average_accuracy)
