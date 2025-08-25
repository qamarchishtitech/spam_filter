"""
---------------------------------
SMS Spam Classifier (VS Code version)
Author: Qamar Chishti
---------------------------------
"""

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re, string

# Step 2: Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# Step 3: Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Apply cleaning
data['clean_message'] = data['message'].apply(clean_text)

# Step 4: Train/Test Split
X = data['clean_message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 6: Train Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 8: Real-life Test
sample_msgs = [
    "Congratulations! You have won a free iPhone. Click here to claim your prize!",
    "Hey Qamar, kal office meeting kab hai?",
    "Limited offer! Win cash now by sending SMS to 12345!"
]

sample_clean = [clean_text(msg) for msg in sample_msgs]
sample_tfidf = tfidf.transform(sample_clean)
predictions = model.predict(sample_tfidf)

for msg, label in zip(sample_msgs, predictions):
    print(f"Message: {msg} → Prediction: {label}")

import joblib   # save/load ke liye library

# Model train hone ke baad save karo
joblib.dump(model, "sms_spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("✅ Model aur Vectorizer local folder me save ho gaye!")
