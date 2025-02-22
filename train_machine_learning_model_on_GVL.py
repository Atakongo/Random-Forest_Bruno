import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load the saved data
df = pd.read_excel("cosine_similarity_GVL_train_results.xlsx")

print(df.head())

# Assume 'match' column is the target
X = df.drop(columns=["label"])
y = df["label"]

#handle missing values
X = X.fillna(0)  # Replace NaN with 0 (or use another strategy)

#normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="precision")
print(f"Cross-validated Accuracy: {cv_scores.mean():.2f}")


# Predictions
y_pred = rf_model.predict(X_test)

# Get misclassified indices
misclassified_indices = (y_test != y_pred)
# Convert X_test and y_test back to DataFrame for easier manipulation
X_test_df = X_test.copy()  # Ensure you have the feature set
y_test_df = pd.Series(y_test, index=X_test.index, name="true_label")  # Align indices

# Add predicted labels to the DataFrame
X_test_df["true_label"] = y_test_df
X_test_df["predicted_label"] = y_pred

# Filter for misclassified rows
misclassified_data = X_test_df[misclassified_indices]
misclassified_data.to_excel("misclassified_data.xlsx", index=False)


# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

import matplotlib.pyplot as plt

# Get feature importance
importances = rf_model.feature_importances_
feature_names = X.columns

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Random Forest")
plt.show()


#joblib.dump(rf_model, "random_forest_model_GVL_trained.pkl")