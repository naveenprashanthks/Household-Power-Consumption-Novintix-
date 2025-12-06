import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
df = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    low_memory=False
)

df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df.set_index("timestamp", inplace=True)

df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

power_hourly = df["Global_active_power"].resample("H").mean().fillna(method="ffill")

for i in range(1, 25):
    power_hourly[f"prev_{i}h"] = power_hourly.shift(i)

power_hourly.dropna(inplace=True)

X = power_hourly.drop("Global_active_power", axis=1)
y = power_hourly["Global_active_power"]

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
