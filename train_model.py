import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    "email": [
        "Win money now",
        "Limited offer buy now",
        "Hello how are you",
        "Let's meet tomorrow",
        "Claim your free prize",
        "Important meeting tomorrow"
    ],
    "label": [1,1,0,0,1,0]
}

df = pd.DataFrame(data)

X = df["email"]
y = df["label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vector, y)

# Save model
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Model trained and saved successfully!")