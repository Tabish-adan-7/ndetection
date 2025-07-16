# File: train_test_split.py

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle


# 1. Load your preprocessed data (from Step 3)
def load_preprocessed_data():
    """Load the text data after preprocessing"""
    # Example - adjust to your actual file path/format
    df = pd.read_csv('D:/projects/fakenews/processed_text.csv')
    return df['clean_text'], df['label']


# 2. Load your TF-IDF features (from Step 4)
def load_tfidf_features():
    """Load the saved vectorizer and features"""
    # Example - adjust paths as needed
    with open('D:/projects/fakenews/tfidf_features.pkl', 'rb') as f:
        X = pickle.load(f)
    return X


# 3. Split the data
def split_dataset(X, y, test_size=0.2, random_state=42):
    """Perform train-test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintains class balance
    )
    return X_train, X_test, y_train, y_test


# 4. Main execution flow
if __name__ == "__main__":
    # Load the data
    texts, labels = load_preprocessed_data()
    X = load_tfidf_features()

    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_dataset(X, labels)

    # Print summary
    print(f"\nDataset split complete:")
    print(f"- Training samples: {(X_train.shape[0]):,}")
    print(f"- Testing samples: {(X_test.shape[0]):,}")
    print(f"- Feature dimension: {X_train.shape[1]}")

    # Save the splits (optional)
    with open('D:/projects/fakenews/train_test_splits.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
    print("Split data saved to disk.")


'''# Save sparse matrices X_train, X_test as DataFrames
X_train_df = pd.DataFrame(X_train.toarray())
X_test_df = pd.DataFrame(X_test.toarray())

# Save labels y_train, y_test as DataFrames
y_train_df = pd.DataFrame(y_train.reset_index(drop=True), columns=["label"])
y_test_df = pd.DataFrame(y_test.reset_index(drop=True), columns=["label"])

# Save to CSV
X_train_df.to_csv("D:/projects/fakenews/X_train.csv", index=False)
X_test_df.to_csv("D:/projects/fakenews/X_test.csv", index=False)
y_train_df.to_csv("D:/projects/fakenews/y_train.csv", index=False)
y_test_df.to_csv("D:/projects/fakenews/y_test.csv", index=False)

print("✅ Split data also saved as CSV.")'''


