import numpy as np                  
import pandas as pd
import json
import os
import catboost as cb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("Data/diabetes_prediction_dataset.csv")

# Save the column names to a JSON file for later use during inference
with open("Model/feature_columns.json", "w") as f:
    json.dump(data.columns.tolist(), f)

# Split the dataset into features and Labels
X = data.drop(columns=["diabetes"])
y = data["diabetes"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define categorical features
cat_features = ['gender', 'smoking_history']

# Train model using CatBoost using optimized for ~100K data
model = cb.CatBoostClassifier(
    iterations=1000,        #Balance between training time and performance
    learning_rate=0.08,      # Slightly lower for better generalization
    depth=8,                # Deeper trees can capture more complex patterns, but may overfit
    l2_leaf_reg=5,         # Regularization to prevent overfitting
    border_count=128,       # Good for continuous features like age, bmi
    random_strength=1,      # Helps with generalization
    bagging_temperature=0.5, # Adds randomness for better generalization
    verbose=100,            # Print training progress every 100 iterations
    eval_metric="AUC",      # AUC is a good metric for binary classification, especially with imbalanced data
    random_seed=42,         # For reproducibility
    early_stopping_rounds=50,    # Stop training if no improvement in AUC for 50 rounds, Prevents overfit
    cat_features=cat_features  # Specify categorical features
    )

# Train the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

# Model prediction 
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

# Lower the threshold to 0.3 to increase recall (catch more positives)
threshold = 0.3
y_pred = (y_proba >= threshold).astype(int)

# Evaluate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model AUC: {auc:.4f}")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
model.save_model("Model/diabetes_model.cbm")

print("\nModel training complete and saved to Model/diabetes_model.cbm")

# Create evaluation directory if it doesn't exist
os.makedirs("Evaluation", exist_ok=True)

# Save classiification report as a text file
report = classification_report(y_test, y_pred)
with open("Evaluation/classification_report.txt", "w") as f:
    f.write("Model Evaluation Report\n")
    f.write("=======================\n\n")
    f.write(f"Model Accuracy: {accuracy:.4f}\n")
    f.write(f"Model AUC: {auc:.4f}\n")
    f.write(report)
    print("Classification report saved in Evaluation/classification_report.txt")

# Generate confusion matrix and save it as an image
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('Evaluation/confusion_matrix.png')

print("Confusion matrix saved in Evaluation/confusion_matrix.png")

# Save metrics as a JSON file
metrics_dict = {
    "accuracy": float(accuracy),
    "auc": float(auc)
}

with open("Evaluation/metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=4)

print("Metrics saved in Evaluation/metrics.json")

print("All evaluation artifacts saved in the /Evaluation folder")