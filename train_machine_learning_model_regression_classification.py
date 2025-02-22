import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

print("Up and running")

scaler = StandardScaler()

# Load the saved data
df_train = pd.read_excel("cosine_similarity_GVL_train_results.xlsx")
df_val = pd.read_excel("cosine_similarity_GVL_validation_results.xlsx")
df_test = pd.read_excel("cosine_similarity_GVL_test_results.xlsx")


df_test_pop_rock = pd.read_excel("cosine_similarity_GVL_test_pop_rock_results.xlsx")
df_test_klassik = pd.read_excel("cosine_similarity_GVL_test_klassik_results.xlsx")


# 'match' column is the renamed to 'label' as the target column
X_train = df_train.drop(columns=["label"])
y_train = df_train["label"]

X_val = df_val.drop(columns=["label"])
y_val = df_val["label"]

X_test = df_test.drop(columns=["label"])
y_test = df_test["label"]

X_test_klassik = df_test_klassik.drop(columns=["label"])
y_test_klassik = df_test_klassik["label"]

X_test_pop_rock = df_test_pop_rock.drop(columns=["label"])
y_test_pop_rock = df_test_pop_rock["label"]

def handleMissingValues():
    X_train = X_train.fillna(0)  # Replace NaN with 0 
    X_val = X_val.fillna(0)  # Replace NaN with 0 
    X_test = X_test.fillna(0)  # Replace NaN with 0 
    X_test_klassik = X_test_klassik.fillna(0) # Replace NaN with 0 
    X_test_pop_rock = X_test_pop_rock.fillna(0) # Replace NaN with 0 

#handle missing values
handleMissingValues()

#nomalize the data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.fit_transform(X_test)
X_test_klassik_scaled = scaler.fit_transform(X_test_klassik)
X_test_pop_rock_scaled = scaler.fit_transform(X_test_pop_rock)


# Initialize the model
rf_model = RandomForestClassifier(n_estimators=400, random_state=42, max_depth= None, max_features= "sqrt", min_samples_leaf= 1, min_samples_split= 2)

# Train the modelss
rf_model.fit(X_train_scaled, y_train)

cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=10, scoring="precision") # cv was 10 but for xgboost 5 was tested
print(f"Cross-validated Accuracy: {cv_scores.mean():.2f}")


# Fit the model to the validation data
rf_model.fit(X_val_scaled, y_val)

# Predictions
y_pred = rf_model.predict(X_test_scaled)
y_pred_klassik = rf_model.predict(X_test_klassik_scaled)
y_pred_pop_rock = rf_model.predict(X_test_pop_rock_scaled)



def evaluate(real_label, predictions):
   accuracy = accuracy_score(real_label, predictions)
   precision = precision_score(real_label, predictions)
   recall = recall_score(real_label,predictions)
   f1Score = f1_score(real_label, predictions)
   
   print(f"Accuracy Test General: {accuracy:.4f}")
   print(f"Precision Test General: {precision:.4f}")
   print(f"Recall Test General: {recall:.4f}")
   print(f"F1 Score Test General: {f1Score:.4f}")
   
def generateConfusionMatrix(real_label, predictions):
    # Confusion matrix
    conf_matrix_general = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix_general)
    
    
# Metrics general
print("General metrics")
evaluate(y_test, y_pred)
print("General Confusion Matrix")
generateConfusionMatrix(y_test, y_pred)



# Metrics klassik
print("Classical Music  metrics")
evaluate(y_test_klassik, y_pred_klassik)
print("Classical Music Confusion Matrix")
generateConfusionMatrix(y_test_klassik, y_pred_klassik)


# Metrics pop rock
print("Pop Rock Music  metrics")
evaluate(y_test_pop_rock, y_pred_pop_rock)
print("Pop Rock Music Confusion Matrix")
generateConfusionMatrix(y_test_pop_rock, y_pred_pop_rock)

def getMissclassifiedData():
    # Get misclassified indices
    misclassified_indices = (y_test != y_pred)

    # Convert X_test and y_test back to DataFrame for easier manipulation
    X_test_df = X_test.copy()  # Ensure having the identical feature set
    y_test_df = pd.Series(y_test, index=X_test.index, name="true_label")  # Align indices

    # Add predicted labels to the DataFrame
    X_test_df["true_label"] = y_test_df
    X_test_df["predicted_label"] = y_pred

    # Filter for misclassified rows
    misclassified_data = X_test_df[misclassified_indices]
    misclassified_data.to_excel("misclassified_data.xlsx", index=True)



