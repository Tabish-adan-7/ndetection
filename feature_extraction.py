import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as p
#Load the preprocessed file

df = pd.read_csv(r"D:\projects\fakenews\processed_text.csv")

#fix missing values
df['clean_text'] = df['clean_text'].fillna('')

#Setup the TF-ID Vectorizer

tfid_vectorizer = TfidfVectorizer(
    max_features= 5000,
    stop_words='english',
    ngram_range = (1,2),
    min_df=5,
    max_df=0.7
)

#Convert the text to numbers
x_tfidf = tfid_vectorizer.fit_transform(df['clean_text'])

#Get the features names
features_names= tfid_vectorizer.get_feature_names_out()

#convert sparse Matrix to DataFrame

tfidf_df = pd.DataFrame(
    x_tfidf.toarray(),
    columns=features_names
)

#print some information
print(f"Shape of TF-IDF matrix: {x_tfidf.shape}")
print("\nSample features (words/phrases):")
print(features_names[:20])
print("\nSample TF-IDF values for first document:")
print(tfidf_df.iloc[0].sort_values(ascending=False).head(10))

#saving
import pickle
import os

# Define the folder path
project_folder = r"D:\projects\fakenews"

# Save the TF-IDF features matrix
with open(os.path.join(project_folder, 'tfidf_features.pkl'), 'wb') as f:
    pickle.dump(x_tfidf, f)

# Save the TF-IDF vectorizer object
with open(os.path.join(project_folder, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfid_vectorizer, f)

