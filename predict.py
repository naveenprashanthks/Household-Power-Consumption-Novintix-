import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
dataf = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    low_memory=False
)

dataf["timestamp"] = pd.to_datetime(dataf["Date"] + " " + dataf["Time"])
dataf.set_index("timstamp", inplace=True)

dataf["Global_active_power"] = pd.to_numeric(dataf["Global_active_power"], errors="coerce")

powerHourly = dataf["Global_active_power"].resample("H").mean().fillna(method="ffill")
range = 25
for i in range(1, range):
    powerHourly[f"prev_{i}h"] = powerHourly.shift(i)

powerHourly.dropna(inplace=True)

X = powerHourly.drop("Global_active_power", axis=1)
y = powerHourly["Global_active_power"]

train_size = int(len(X) * 0.4)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(mae)
print(rmse)
