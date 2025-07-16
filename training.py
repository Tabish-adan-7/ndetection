import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)


# 1. Load Preprocessed Data
def load_data(filepath):
    """Load preprocessed TF-IDF data and splits"""
    with open(filepath, 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data('D:/projects/fakenews/train_test_splits.pkl')
# Convert labels: 'fake' → 0, 'real' → 1
y_train = y_train.map({'fake': 0, 'real': 1})
y_test = y_test.map({'fake': 0, 'real': 1})

# 2. Initialize and Train Logistic Regression
model = LogisticRegression(
    max_iter=1000,  # Ensure convergence
    random_state=42,  # Reproducibility
    class_weight='balanced',  # Handle class imbalance
    solver='liblinear',  # Good for small-to-medium datasets
    C=1.0  # Regularization strength
)
model.fit(X_train, y_train)

# ✅ NEW: Calculate Training Accuracy
from sklearn.metrics import accuracy_score
train_preds = model.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)
print(f"✅ Training Accuracy: {train_acc:.4f}")

# 3. Evaluation Functions
def evaluate_model(model, X_test, y_test):
    """Generate comprehensive evaluation metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Key metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.title('Confusion Matrix')
    plt.savefig('D:/projects/fakenews/confusion_matrix.png')  # ✅ Save to file
    plt.close()  # Close the plot to avoid overlapping

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'LogReg (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('D:/projects/fakenews/roc_curve.png')  # ✅ Save to file
    plt.close()

    return metrics


# 4. Evaluate the model
metrics = evaluate_model(model, X_test, y_test)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Average CV score:", scores.mean())


# 5. Save the trained model
joblib.dump(model, 'logistic_regression_fakenews_model.pkl')
print("Model saved successfully!")