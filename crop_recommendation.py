# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load Kaggle Crop Recommendation Dataset
# Columns: N, P, K, temperature, humidity, ph, rainfall, label
df = pd.read_csv("Crop_recommendation.csv")

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model weights for FastAPI backend
with open("crop_recommendation.pkl", "wb") as f:
    pickle.dump(model, f)