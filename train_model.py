import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# =========================
# Load dataset
# =========================
data = pd.read_csv("data/weatherAUS.csv")

# Drop leakage + categorical columns
data = data.drop(columns=[
    "Date", "Location", "RainToday",
    "WindDir9am", "WindDir3pm", "WindGustDir",
    "RISK_MM"
])

# Target
y = data["RainTomorrow"].map({"No": 0, "Yes": 1})
X = data.drop(columns=["RainTomorrow"])

# =========================
# Impute + scale
# =========================
num_imputer = SimpleImputer(strategy="mean")
X = num_imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Train model
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =========================
# Save artifacts
# =========================
with open("model/rainfall_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/imputer.pkl", "wb") as f:
    pickle.dump(num_imputer, f)

print("Model saved successfully")
