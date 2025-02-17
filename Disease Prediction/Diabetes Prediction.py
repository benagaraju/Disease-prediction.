# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset, specifying the separator and handling errors
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns, header=0)

# Check the first few rows of the dataset to understand its structure
print(data.head())

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Separate the features and the target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the classifier (Logistic Regression)
classifier = LogisticRegression(max_iter=1000) 
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Make prediction for new data
new_data = np.array([[0,137,40,35,168,43.1,2.288,33]])  

new_data_scaled = scaler.transform(new_data)
prediction = classifier.predict(new_data_scaled)

if prediction[0] == 1:
    print("The person is diabetic.")
else:
    print("The person is not diabetic.")
