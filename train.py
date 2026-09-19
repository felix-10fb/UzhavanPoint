import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Load Data (or Generate Synthetic Data for Hackathon)
# ---------------------------------------------------------
DATA_FILE = "Crop_recommendation.csv"

if os.path.exists(DATA_FILE):
    print(f"Loading dataset from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
else:
    print(f"'{DATA_FILE}' not found! Generating dummy agricultural data...")
    # Generating fallback dummy data so your backend runs immediately
    np.random.seed(42)
    num_samples = 1100
    crops = ['rice', 'maize', 'chickpea', 'cotton', 'coffee', 'watermelon', 'banana', 'mango', 'jute', 'apple', 'papaya']
    
    data = {
        'N': np.random.randint(10, 140, num_samples),
        'P': np.random.randint(5, 145, num_samples),
        'K': np.random.randint(5, 205, num_samples),
        'temperature': np.random.uniform(10.0, 40.0, num_samples),
        'humidity': np.random.uniform(15.0, 99.0, num_samples),
        'ph': np.random.uniform(3.5, 9.5, num_samples),
        'rainfall': np.random.uniform(20.0, 300.0, num_samples),
        'label': np.random.choice(crops, num_samples)
    }
    df = pd.DataFrame(data)

print(f"Dataset Shape: {df.shape}")

# ---------------------------------------------------------
# 2. Preprocess & Split Features
# ---------------------------------------------------------
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 3. Train Model
# ---------------------------------------------------------
print("Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=12, 
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. Evaluate Model
# ---------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 5. Save Trained Model to Pickle File
# ---------------------------------------------------------
MODEL_FILE = "crop_recommendation.pkl"
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"Model successfully saved as '{MODEL_FILE}'!")