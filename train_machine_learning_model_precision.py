import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score

# Load the Excel file
df = pd.read_excel("cosine_similarity_results.xlsx")
print(df)

# Separate features and target
X = df.drop(columns=["match"])  # Features
y = df["match"]  # Target

# Outer cross-validation loop
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner cross-validation loop for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define the model and hyperparameter grid
model = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# Store results
outer_results = []

for train_idx, test_idx in outer_cv.split(X, y):
    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Perform inner cross-validation for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="precision",
        cv=inner_cv,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Evaluate the best model on the outer test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    outer_results.append(precision)

    print(f"Outer Fold Precision: {precision:.4f}")
    print(classification_report(y_test, y_pred))

# Report overall performance
print("\nNested Cross-Validation Results:")
print(f"Mean precision: {np.mean(outer_results):.4f}")
print(f"Standard Deviation: {np.std(outer_results):.4f}")

# Train the final model using the best hyperparameters on the full dataset
best_params = grid_search.best_params_
final_model = RandomForestClassifier(random_state=42, **best_params)
final_model.fit(X, y)

# Save the trained model for future use
import joblib
#joblib.dump(final_model, "final_model_precision.pkl")


# Get the predictions for the test set
y_pred = best_model.predict(X_test)

# Compare predictions with true labels to find incorrect classifications
incorrect_indices = np.where(y_pred != y_test)[0]

# Retrieve rows with incorrect classifications
incorrect_classifications = X_test.iloc[incorrect_indices].copy()
incorrect_classifications["Actual"] = y_test.iloc[incorrect_indices].values
incorrect_classifications["Predicted"] = y_pred[incorrect_indices]

# Display incorrect classifications
print("\nIncorrect Classifications:")
print(incorrect_classifications)

# Get feature importance from the trained Random Forest model
feature_importances = final_model.feature_importances_

# Create a DataFrame for better visualization
importances_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Display feature importances
print("\nFeature Importances:")
print(importances_df)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importances_df)
plt.title("Feature Importances from Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

