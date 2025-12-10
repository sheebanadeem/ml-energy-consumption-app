import numpy as np
import pandas as pd
import datetime as dt

np.random.seed(42)

# Time range for 2 years hourly
start = pd.Timestamp("2023-01-01 00:00:00")
end   = pd.Timestamp("2024-12-31 23:00:00")
rng = pd.date_range(start, end, freq="H")
n = len(rng)

# Seasonal temperature (Delhi-like)
day_of_year = rng.dayofyear.values
hour_of_day = rng.hour.values
month = rng.month.values

yearly_temp = 25 + 12 * np.sin(2 * np.pi * (day_of_year - 170) / 365.25)
daily_temp  = 6 * np.sin(2 * np.pi * (hour_of_day - 14) / 24)
temp_c = yearly_temp + daily_temp + np.random.normal(0, 1.8, n)

# Humidity
monsoon_factor = ((month >= 6) & (month <= 9)).astype(float)
humidity = 55 - 0.6 * (temp_c - temp_c.mean()) + 20 * monsoon_factor + np.random.normal(0, 6, n)
humidity = np.clip(humidity, 10, 100)

# Heat index (approx)
heat_index = temp_c + 0.33 * humidity - 0.7 * np.cos((hour_of_day / 24) * 2 * np.pi)

# Base load patterns
base = 3500 + 400 * np.sin(2 * np.pi * (day_of_year - 200) / 365.25)
daily_load = 200 * np.sin(2 * np.pi * (hour_of_day - 18) / 24)
weekday_effect = np.where(rng.dayofweek < 5, 1.03, 0.94)

# Zone-wise baseline loads
north = (base * 0.27) + (daily_load * 0.95)
south = (base * 0.25) + (daily_load * 1.05)
east  = (base * 0.23) + (daily_load * 0.9)
west  = (base * 0.22) + (daily_load * 1.0)

# Add heat-based multiplier
heat_factor = 1 + 0.004 * (heat_index - heat_index.mean())
north *= heat_factor * weekday_effect
south *= heat_factor * weekday_effect
east  *= heat_factor * weekday_effect
west  *= heat_factor * weekday_effect

# Add random noise
north += np.random.normal(0, 30, n)
south += np.random.normal(0, 28, n)
east  += np.random.normal(0, 26, n)
west  += np.random.normal(0, 24, n)

north = np.clip(north, 60, None)
south = np.clip(south, 40, None)
east  = np.clip(east, 30, None)
west  = np.clip(west, 20, None)

# Total demand
total = north + south + east + west
total = total * (1 + np.random.normal(0, 0.005, n))

# Create DataFrame
df = pd.DataFrame({
    "timestamp": rng,
    "north_mw": north,
    "south_mw": south,
    "east_mw": east,
    "west_mw": west,
    "total_mw": total,
    "temp_c": temp_c,
    "humidity_pct": humidity,
    "heat_index": heat_index,
})

# Calendar features
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day

def season_from_month(m):
    if m in [12,1,2]: return "winter"
    if m in [3,4]: return "spring"
    if m in [5,6,7,8]: return "summer"
    return "autumn"

df["season"] = df["month"].apply(season_from_month)

# Holidays
holidays = [
    "2023-01-26","2023-08-15","2023-10-02","2023-11-12","2023-12-25",
    "2024-01-26","2024-08-15","2024-10-02","2024-11-01","2024-12-25"
]
df["holiday_flag"] = df["timestamp"].dt.strftime("%Y-%m-%d").isin(holidays).astype(int)

# Lag features
df["lag_1"] = df["total_mw"].shift(1)
df["lag_24"] = df["total_mw"].shift(24)
df["lag_168"] = df["total_mw"].shift(168)

# Rolling features
df["roll_mean_24"] = df["total_mw"].shift(1).rolling(24, min_periods=1).mean()
df["roll_std_24"] = df["total_mw"].shift(1).rolling(24, min_periods=1).std().fillna(0)

# Final cleanup
df.fillna(method="bfill", inplace=True)

# Save file
df.to_csv("delhi_energy_2yr_zone.csv", index=False)

print("Dataset generated successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Saved as: delhi_energy_2yr_zone.csv")
