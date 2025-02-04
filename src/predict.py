import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import os
from helpers import logger

EXAMPLES_DIR = os.path.join("src", "regression_data", "rome", "counterfact")

# Load JSON data
data = []
logger.info(f"Searching for files in {EXAMPLES_DIR}...")
for f in os.listdir(EXAMPLES_DIR):
    if f.endswith("gpt2-xl.json"):
        logger.info(f"Loading data from {os.path.join(EXAMPLES_DIR, f)}...")
        with open(os.path.join(EXAMPLES_DIR, f), "r") as fl:
            fl_data = json.load(f)
        data.extend(fl_data.values())

# Log the data type and first few samples
logger.info(f"Data type: {type(data)}")
logger.info(f"Number of samples: {len(data)}")
logger.info(f"First 3 samples (raw): {data[:3]}")

# Convert JSON array of arrays to NumPy array
data = np.array(data)

# Log dataset shape
logger.info(f"Dataset shape: {data.shape}")

# Ensure correct shape
if data.shape[1] != 49:
    logger.error(f"Unexpected number of columns: {data.shape[1]}, expected 49.")
    raise ValueError("Feature size mismatch.")

# Extract features (first 48 columns) and labels (last column)
X = data[:, :-1]
y = data[:, -1]

logger.info(f"Feature matrix shape: {X.shape}, Label vector shape: {y.shape}")
logger.info(
    f"Label distribution: {np.bincount(y.astype(int))}"
)  # Check class balance

# Split into train (first 324) and test (rest)
X_train, X_test = X[:324], X[324:]
y_train, y_test = y[:324], y[324:]

logger.info(
    f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}"
)

# Train logistic regression model
clf = LogisticRegression(
    max_iter=500
)  # Increase iterations to ensure convergence

logger.info("Training logistic regression model...")
clf.fit(X_train, y_train)

logger.info("Model training completed.")

# Make predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[
    :, 1
]  # Probability estimates for positive class (1)

# Log predictions for debugging
logger.info(f"First 10 predictions: {y_pred[:10]}")
logger.info(f"First 10 probabilities: {y_prob[:10]}")

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print Results
logger.info(f"Accuracy: {accuracy:.4f}")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"F1 Score: {f1:.4f}")
logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
logger.info("Confusion Matrix:")
logger.info(conf_matrix)
