import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("your_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

# Get the earliest date
start_date = df["date"].min()

# Save the start date as a pickle file
joblib.dump(start_date, "start_date.pkl")

print(f"start_date.pkl saved with date: {start_date}")
