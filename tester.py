"""
---------------------------------
Use Pre-trained Spam Classifier (VS Code)
---------------------------------
"""

import joblib
import re, string

# Cleaning function (same as Colab)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Step 1: Load trained model + vectorizer
model = joblib.load("sms_spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

print("✅ Model aur Vectorizer successfully load ho gaye!")

# Step 2: Real-life test messages
sample_msgs = [
    "Congratulations! You won a free iPhone. Claim now!",
    "Hey Qamar bhai, kal chai pe milte hain?",
    "Limited offer!!! Send SMS to 12345 and win cash."
]

# Step 3: Clean + Vectorize + Predict
sample_clean = [clean_text(msg) for msg in sample_msgs]
sample_tfidf = tfidf.transform(sample_clean)
predictions = model.predict(sample_tfidf)

# Step 4: Results print karo
for msg, label in zip(sample_msgs, predictions):
    print(f"📩 Message: {msg}")
    print(f"🔮 Prediction: {label}")
    print("-" * 40)
