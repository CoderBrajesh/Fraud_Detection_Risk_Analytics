import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/transactions.csv")

print("==========================================")
print("FRAUD DETECTION - ML MODEL TRAINING")
print("==========================================")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "transaction_amount",
    "account_age_days",
    "previous_transactions",
    "failed_transactions",
    "transaction_hour",
    "is_international",
    "device_change",
]

X = df[features]
y = df["is_fraud"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 6. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(classification_report(y_test, y_pred, zero_division=0))


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame(
    {"Feature": features, "Importance": model.feature_importances_}
)

importance = importance.sort_values(by="Importance", ascending=False)

print("\n==========================================")
print("FEATURE IMPORTANCE")
print("==========================================")

print(importance)


# ==========================================
# 11. SAVE MODEL
# ==========================================

joblib.dump(model, "model/fraud_model.pkl")

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY!")
print("==========================================")

print("\nSaved file:")
print("model/fraud_model.pkl")
