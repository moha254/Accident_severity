# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('/content/drive/MyDrive/RTA Dataset.csv')


label_encoder = LabelEncoder()
data['Accident_severity_encoded'] = label_encoder.fit_transform(data['Accident_severity'])

# Independent variables (X) and dependent variable (y)
features = ['Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Driving_experience',
            'Type_of_vehicle', 'Vehicle_movement', 'Cause_of_accident']


X = pd.get_dummies(data[features], drop_first=True)
y = data['Accident_severity_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict accident severity on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model for future use
import joblib
joblib.dump(model, 'accident_severity_model.pkl')


example_data = pd.DataFrame({
    'Age_band_of_driver_31-50': [1],
    'Sex_of_driver_Male': [1],
    'Educational_level_High school': [1],
    'Driving_experience_Above 10yr': [1],
    'Type_of_vehicle_Automobile': [1],
    'Vehicle_movement_Going straight': [1],
    'Cause_of_accident_Overtaking': [1]
    
})


for col in X_train.columns:
    if col not in example_data.columns:
        example_data[col] = 0

# Reorder columns to match the training set
example_data = example_data[X_train.columns]
model = joblib.load('accident_severity_model.pkl')
# Predict accident severity for the hypothetical case
severity_prediction = model.predict(example_data)
print(f"Predicted Accident Severity: {severity_prediction}")