import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocessing import clean_text


MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# LOAD DATASETS
# =========================

true_df = pd.read_csv("dataset/True.csv")
fake_df = pd.read_csv("dataset/Fake.csv")


# =========================
# ADD LABELS
# =========================

true_df["label"] = 1      # REAL
fake_df["label"] = 0      # FAKE


# =========================
# COMBINE DATASETS
# =========================

df = pd.concat([true_df, fake_df], ignore_index=True)


# =========================
# REMOVE MISSING DATA
# =========================

df = df.dropna(subset=["text"])


# =========================
# COMBINE TITLE + ARTICLE
# =========================

df["content"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
)


# =========================
# CLEAN TEXT
# =========================

df["clean_text"] = df["content"].astype(str).apply(clean_text)


# =========================
# SHUFFLE DATA
# =========================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)


print("===================================")
print("      FAKE NEWS DETECTION MODEL")
print("===================================")

print(f"Total articles: {len(df)}")
print(f"Real news: {(df['label'] == 1).sum()}")
print(f"Fake news: {(df['label'] == 0).sum()}")


# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)


print(f"\nTraining articles: {len(X_train)}")
print(f"Testing articles: {len(X_test)}")


# =========================
# TF-IDF
# =========================

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)


print("\nCreating TF-IDF features...")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# =========================
# LOGISTIC REGRESSION
# =========================

print("Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    C=2.0
)

model.fit(X_train_vec, y_train)


# =========================
# PREDICTION
# =========================

predictions = model.predict(X_test_vec)


# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, predictions)

print("\n===================================")
print("          MODEL RESULTS")
print("===================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fake News", "Real News"]
))


# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    f"{MODEL_DIR}/model.pkl"
)

joblib.dump(
    vectorizer,
    f"{MODEL_DIR}/vectorizer.pkl"
)


print("\n===================================")
print("Model saved successfully!")
print("===================================")