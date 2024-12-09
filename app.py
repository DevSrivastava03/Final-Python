import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset from a local file
local_file_path = "Student Depression Dataset.csv"

# Load the dataset
data = pd.read_csv(local_file_path)


# Fill missing values for 'Financial Stress' with mean
data['Financial Stress'] = data['Financial Stress'].fillna(data['Financial Stress'].mean())

# Map 'Gender' to numerical values
gender_mapping = {'Male': 0, 'Female': 1}
data['Gender'] = data['Gender'].map(gender_mapping)

# Map 'City' to numerical values
city_mapping = {city: idx for idx, city in enumerate(data['City'].unique())}
data['City'] = data['City'].map(city_mapping)

# Map 'Degree' to numerical values
degree_mapping = {degree: idx for idx, degree in enumerate(data['Degree'].unique())}
data['Degree'] = data['Degree'].map(degree_mapping)

# Map 'Have you ever had suicidal thoughts ?' to numerical values
suicidal_mapping = {'Yes': 1, 'No': 0}
data['Have you ever had suicidal thoughts ?'] = data[
    'Have you ever had suicidal thoughts ?'
].map(suicidal_mapping)

# Map 'Family History of Mental Illness' to numerical values
family_history_mapping = {'Yes': 1, 'No': 0}
data['Family History of Mental Illness'] = data['Family History of Mental Illness'].map(
    family_history_mapping
)

# Map 'Sleep Duration' text values to numeric values
sleep_duration_mapping = {
    'Less than 5 hours': 4,
    '5-6 hours': 5.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 9
}
data['Sleep Duration'] = data['Sleep Duration'].map(sleep_duration_mapping)
data['Sleep Duration'] = data['Sleep Duration'].fillna(data['Sleep Duration'].mean())

# Map 'Dietary Habits' text values to numeric values
dietary_habits_mapping = {
    'Healthy': 8,
    'Moderate': 5,
    'Unhealthy': 2
}
data['Dietary Habits'] = data['Dietary Habits'].map(dietary_habits_mapping)
data['Dietary Habits'] = data['Dietary Habits'].fillna(data['Dietary Habits'].mean())

# Handle missing values in CGPA if present
data['CGPA'] = data['CGPA'].fillna(data['CGPA'].mean())

# Ensure all other numeric columns are filled with appropriate defaults
for col in ['Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours']:
    data[col] = data[col].fillna(data[col].mean())

# Drop irrelevant or non-numeric columns (e.g., 'id', 'Profession')
X = data.drop(columns=["Depression", "id", "Profession"])
y = data["Depression"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# # Model evaluation
# y_pred = model.predict(X_test_scaled)
# st.write("Model Performance:")
# st.text(classification_report(y_test, y_pred))

# User input for new data
st.write("Enter details for a new data batch:")
new_data_batch = {
    'Gender': st.selectbox("Gender", ['Male', 'Female']),
    'Age': st.number_input("Age", min_value=18, max_value=100, value=20),
    'City': st.selectbox("City", list(city_mapping.keys())),
    'Academic Pressure': st.slider("Academic Pressure", 0.0, 5.0, 3.0),
    'Work Pressure': st.slider("Work Pressure", 0.0, 5.0, 2.0),
    'CGPA': st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0),
    'Study Satisfaction': st.slider("Study Satisfaction", 0.0, 10.0, 7.0),
    'Job Satisfaction': st.slider("Job Satisfaction", 0.0, 10.0, 5.0),
    'Sleep Duration': st.selectbox("Sleep Duration", ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']),
    'Dietary Habits': st.selectbox("Dietary Habits", ['Healthy', 'Moderate', 'Unhealthy']),
    'Degree': st.selectbox("Degree", list(degree_mapping.keys())),
    'Have you ever had suicidal thoughts ?': st.selectbox("Suicidal Thoughts", ['Yes', 'No']),
    'Work/Study Hours': st.slider("Work/Study Hours", 0.0, 24.0, 4.0),
    'Financial Stress': st.slider("Financial Stress", 0.0, 10.0, 5.0),
    'Family History of Mental Illness': st.selectbox("Family History of Mental Illness", ['Yes', 'No']),
}

# Convert user input to numeric
new_data_batch['Gender'] = gender_mapping[new_data_batch['Gender']]
new_data_batch['City'] = city_mapping[new_data_batch['City']]
new_data_batch['Degree'] = degree_mapping[new_data_batch['Degree']]
new_data_batch['Have you ever had suicidal thoughts ?'] = suicidal_mapping[
    new_data_batch['Have you ever had suicidal thoughts ?']
]
new_data_batch['Family History of Mental Illness'] = family_history_mapping[
    new_data_batch['Family History of Mental Illness']
]
new_data_batch['Sleep Duration'] = sleep_duration_mapping[new_data_batch['Sleep Duration']]
new_data_batch['Dietary Habits'] = dietary_habits_mapping[new_data_batch['Dietary Habits']]

# Prepare the new data batch for prediction
new_data_batch_df = pd.DataFrame(new_data_batch, index=[0])

# Scale the new data
new_data_scaled_batch = scaler.transform(new_data_batch_df)

# Make predictions
predictions_batch = model.predict(new_data_scaled_batch)
prediction_proba = model.predict_proba(new_data_scaled_batch)


st.write("Prediction Probabilities:")
st.write(pd.DataFrame(prediction_proba, columns=model.classes_))
