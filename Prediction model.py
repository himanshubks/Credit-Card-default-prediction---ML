# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Data Acquisition

# Load your credit card default dataset (replace 'credit_card_data.csv' with your data file)
data = pd.read_csv('credit_card_data.csv')

# Step 2: Data Preprocessing

# Handle missing values (if any)
data.dropna(inplace=True)

# Define your features (X) and target (y)
X = data[['column1', 'column2', 'column3', 'column4', 'column5', 'column6', 'column7', 'column8']]  # Replace with your relevant column names
y = data['default']  # Replace 'default' with your target column name

# Step 3: Data Splitting

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (KNN and SVM require feature scaling)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training and Evaluation (Logistic Regression)

# Initialize and train a Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Make predictions on the test data
lr_y_pred = lr_model.predict(X_test)

# Evaluate the Logistic Regression model
lr_accuracy = accuracy_score(y_test, lr_y_pred)
print("Logistic Regression Accuracy:", lr_accuracy)

# Generate a classification report for more detailed evaluation
lr_report = classification_report(y_test, lr_y_pred)
print("Logistic Regression Classification Report:\n", lr_report)

# Step 6: Model Training and Evaluation (Random Forest)

# Initialize and train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test data
rf_y_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", rf_accuracy)

# Generate a classification report for more detailed evaluation
rf_report = classification_report(y_test, rf_y_pred)
print("Random Forest Classification Report:\n", rf_report)

# Step 7: Model Training and Evaluation (Support Vector Machine - SVM)

# Initialize and train a Support Vector Machine (SVM) model
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
svm_y_pred = svm_model.predict(X_test_scaled)

# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print("SVM Accuracy:", svm_accuracy)

# Generate a classification report for more detailed evaluation
svm_report = classification_report(y_test, svm_y_pred)
print("SVM Classification Report:\n", svm_report)

# Step 8: Model Training and Evaluation (K-Nearest Neighbors - KNN)

# Initialize and train a K-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed
knn_model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
knn_y_pred = knn_model.predict(X_test_scaled)

# Evaluate the K-Nearest Neighbors model
knn_accuracy = accuracy_score(y_test, knn_y_pred)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)

# Generate a classification report for more detailed evaluation
knn_report = classification_report(y_test, knn_y_pred)
print("K-Nearest Neighbors Classification Report:\n", knn_report)
