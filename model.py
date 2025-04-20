import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import joblib

# Load dataset
df = pd.read_csv('your_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

# Convert date to days
start_date = df['date'].min()
df['days'] = (df['date'] - start_date).dt.days

# Define features and labels
X = df[['days']]
y = df['sales']

# Polynomial Transformation (degree 2 is good enough to start)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# Save everything
joblib.dump(model, 'model.pkl')
joblib.dump(start_date, 'start_date.pkl')
joblib.dump(poly, 'poly_transform.pkl')  # NEW!
