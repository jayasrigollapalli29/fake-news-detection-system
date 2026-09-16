import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Fake news:", len(fake))
print("Real news:", len(true))

print("\nFake news columns:")
print(fake.columns)

print("\nReal news columns:")
print(true.columns)

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")

data = data[["content", "label"]]

print("\nTotal news:", len(data))
print("\nColumns:")
print(data.columns)

print("\nExample news:")
print(data["content"].iloc[0])
print("\nLabel:", data["label"].iloc[0])
from sklearn.model_selection import train_test_split

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF conversion completed!")
print("Training features:", X_train_tfidf.shape)
print("Testing features:", X_test_tfidf.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import pickle
import os

os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("model/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel and vectorizer saved successfully!")