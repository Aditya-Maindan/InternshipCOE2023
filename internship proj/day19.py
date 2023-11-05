from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/adityamaindan/Desktop/internship proj/instagram.csv')
y = df['fake']
X = df[['profile pic', 'nums/length username', '#posts', '#followers', '#follows']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

TP = conf_matrix[1, 1]  # True Positives
FP= conf_matrix[0, 1]  # False Positives
TN = conf_matrix[0, 0]  # True Negatives
FN = conf_matrix[1, 0]  # False Negatives

precision = TP / (TP + FP)

# Calculate sensitivity (recall)
sensitivity = TP / (TP + FN)

# Calculate specificity
specificity = TN / (TN + FP)

print(f'Precision: {precision:.2f}')
print(f'Sensitivity (Recall): {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')