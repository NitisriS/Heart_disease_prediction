import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/heart_disease_data.csv')

print(df.isnull().sum())

df.head()

df.tail()

df.shape

df.info()

df.describe()

df['target'].value_counts()

"""1 - defective heart
0 - healthy heart
"""

x = df.drop(columns='target', axis=1)

x

y = df['target']

y

"""Splitting dataset into training and testing"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=23)

print(x_test.shape, x_train.shape, y_test.shape, y_train.shape)

"""feature scaling"""

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

"""model training - logistic regression"""

model = LogisticRegression(solver='liblinear')

model.fit(x_train, y_train)

prediction = model.predict(x_test)

y_train

score = accuracy_score(prediction, y_test)

"""Building a prediction system"""

def predict_heart_disease():
    print("\nEnter the following details to predict heart disease risk:")

    features = [
        "Age", "Sex (0: Female, 1: Male)", "Chest Pain Type (0-3)", "Resting BP",
        "Cholesterol", "Fasting Blood Sugar (0: False, 1: True)", "Resting ECG (0-2)",
        "Max Heart Rate", "Exercise Induced Angina (0: No, 1: Yes)", "ST Depression",
        "Slope of ST Segment (0-2)", "Number of Major Vessels (0-3)", "Thalassemia (0-3)"
    ]

    user_input = []
    for feature in features:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
                user_input.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    input_values = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_values)  # Scale the input

    result = model.predict(input_scaled)

    if result[0] == 0:
        print("\nYou're healthy and have no risk of heart attack.")
    else:
        print("\nYou're unhealthy, and the risk factor for a heart attack is high.")

predict_heart_disease()