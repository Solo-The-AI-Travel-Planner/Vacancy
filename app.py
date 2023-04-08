from flask import Flask, render_template

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# Load the data
df = pd.read_csv('hotel_booking.csv')

# Create num_rooms_available column
df['num_rooms_available'] = df['reserved_room_type'].apply(lambda x: ord('G') - ord(x) + 1)

# Create num_rooms_booked column
df['num_rooms_booked'] = df['assigned_room_type'].apply(lambda x: ord('G') - ord(x) + 1)

# Create num_cancellations column
df['num_cancellations'] = df['previous_cancellations'] + df['previous_bookings_not_canceled']

# Create avg_length_of_stay column
df['avg_length_of_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

# Create location column
df['location'] = df['hotel'].apply(lambda x: 'City Hotel' if x == 'City Hotel' else 'Resort Hotel')

# Prepare the data
X = df[['num_rooms_available', 'num_rooms_booked', 'num_cancellations', 'avg_length_of_stay', 'location']]
X = pd.get_dummies(X)

df['vacancy_rate'] = (df['num_rooms_available'] - df['num_rooms_booked']) / df['num_rooms_available']
y = df['vacancy_rate']

df['vacancy_rate'].fillna(0, inplace=True)
df = df[~df['vacancy_rate'].isin([np.nan, np.inf, -np.inf])]
df['vacancy_rate'] = df['vacancy_rate'].replace([np.nan, np.inf, -np.inf], df['vacancy_rate'].median())

y = df['vacancy_rate'].to_numpy().reshape(-1, 1)
scaler = MinMaxScaler()
y = scaler.fit_transform(y).flatten()

df = df[df.index.isin(X.index)]
X = X[X.index.isin(df.index)]

assert len(df) == len(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train2 = np.where(np.isnan(y_train), 0.0, y_train)
y_test2 = np.where(np.isnan(y_test), 0.0, y_test)
model = LinearRegression()
model.fit(X_train, y_train2)

app = Flask(__name__)

@app.route('/')
def home():
    # Make a prediction
    prediction = model.predict(X_test)
    # Store the predictions in a list
    prediction_list = prediction.tolist()
    # Print only the first prediction value
    return str(prediction_list[0]*100)

if __name__ == '__main__':
    app.run(debug=True)