import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve, roc_auc_score
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE

df = pd.read_csv('/Users/adityamaindan/Desktop/internship proj/instagram.csv')

dependent_variable = df['fake']
independent_variables = df[['profile pic', 'nums/length username','#posts', '#followers', '#follows']]


smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(independent_variables, dependent_variable)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

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

model = RandomForestClassifier(n_estimators=100)  # Adjust the model as needed
model.fit(X_train, y_train)

# Predict probabilities on the test set
y_probabilities = model.predict_proba(X_test)[:, 1]

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)

# Calculate the AUC (Area Under the Curve)
roc_auc = roc_auc_score(y_test, y_probabilities)

# Find the threshold that maximizes the KS statistic
ks_statistic = np.max(tpr - fpr)
best_threshold = thresholds[np.argmax(tpr - fpr)]

# Print the KS statistic and best threshold
print(f'KS Statistic: {ks_statistic:.4f}')
print(f'Best Threshold: {best_threshold:.4f}')

# Create a table to present the results
ks_table = pd.DataFrame({'Threshold': thresholds, 'True Positive Rate (TPR)': tpr, 'False Positive Rate (FPR)': fpr})
ks_table['KS Statistic'] = ks_table['True Positive Rate (TPR)'] - ks_table['False Positive Rate (FPR)']

# Display the KS statistics table
print(ks_table)


# Create and train a machine learning model
model = RandomForestClassifier(n_estimators=100)  # Adjust the model as needed
model.fit(X_train, y_train)

# Predict probabilities on the test set
y_probabilities = model.predict_proba(X_test)

# Calculate the ROC curve and AUC for each independent variable
plt.figure(figsize=(10, 8))

fpr, tpr, thresholds = roc_curve(y_test, y_probabilities[:, 1])
roc_auc = roc_auc_score(y_test, y_probabilities[:, 1])
plt.plot(fpr, tpr, lw=2, label=f'ROC Curve for Variable 1 (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves for Independent Variables')
plt.legend(loc='lower right')
plt.show()
