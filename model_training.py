
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load data
df = pd.read_csv("fake_reviews_dataset.csv")

# Features and Labels
X = df['review']
y = df['label']

# Vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy & Confusion Matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("review_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
