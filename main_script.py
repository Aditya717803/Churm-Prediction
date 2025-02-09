import numpy as np
import joblib  # or use pickle if you saved with pickle
import warnings 
warnings.filterwarnings('ignore')

# Load the trained model
lr_model = joblib.load(filename="linear_regression_churn_model.joblib")  # Replace "model.pkl" with your actual model file

#load clv model
clv_model = joblib.load(filename="randomforestregressor_clv_model.joblib")  # Replace "model.pkl" with your actual model file

# Define the input features
global feature_names
feature_names = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "MonthlyCharges"
]

global StreamingTV, StreamingMovies, Partner, MultipleLines, tenure, DeviceProtection
StreamingTV = 0
StreamingMovies = 0
Partner = 0 
MultipleLines = 0
tenure = 1
DeviceProtection = 0

# Take user input for each feature excluding globally defined ones
user_input = []
print("Enter customer details for churn prediction:")
for feature in feature_names:
    if feature not in ["StreamingTV", "StreamingMovies", "Partner", "MultipleLines", "tenure", "DeviceProtection"]:
        value = input(f"Enter {feature}: ")
        if '.' in value:  # Convert to float if needed
            user_input.append(float(value))
        else:
            user_input.append(int(value))
    else:
        user_input.append(globals()[feature])

# Convert input to NumPy array and reshape
user_input_array = np.array(user_input).reshape(1, -1)

# Make prediction
churn_prediction = lr_model.predict(user_input_array)[0]
churn_proba = lr_model.predict_proba(user_input_array)[0]
clv_prediction = clv_model.predict(user_input_array)[0]

# Display results
print("\n--- Churn Prediction Results ---")
print(f"Churn Probability: {churn_proba[1] * 100:.2f}%")
print(f"Prediction: {'Churn' if churn_prediction == 1 else 'No Churn'}")
print(f"Clv Prediction: {clv_prediction:.2f}")