import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import seaborn as sns
import matplotlib.pyplot as plt


# TODO:
# [] Data Scaling on Features w/ StandardScaler
# [x] ROC scores

# Kaggle: https://www.kaggle.com/datasets/iamtanmayshukla/cardiac-arrest-dataset
# Data is based on Berkley UCI data
data = pd.read_csv("./data/data.csv").to_numpy()

# X = features, y = target
X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=7, shuffle=True
)

# TODO: hyperparameter tuning instead of hardcoding the scale C
# NOTE: possible that class weights could manually improve the accuracy - would need to determine the class weights
svcl = svm.LinearSVC(C=0.25, class_weight="balanced", verbose=1)
svcl.fit(X_train, y_train)

# Make predictions
y_pred = svcl.predict(X_test)

print(y_pred)


# Evaluate the model

# 1. Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 2. Classification Report (precision, recall, F1 score)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# 3. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# ROC-AUC for binary classification
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC: {roc_auc:.4f}")

# Precision-Recall AUC\
precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
