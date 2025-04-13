import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Setup ------------------
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ------------------ Load and Clean Data ------------------
data = pd.read_csv("Air_Quality.csv")
data['last_update'] = pd.to_datetime(data['last_update'], errors='coerce')
data = data.dropna(subset=['pollutant_avg'])

# ------------------ Reshape and Feature Engineering ------------------
pivot = data.pivot_table(index=['country', 'state', 'city', 'station', 'last_update', 'latitude', 'longitude'],
                         columns='pollutant_id', values='pollutant_avg').reset_index()
pivot.columns.name = None
pivot['month'] = pivot['last_update'].dt.month
pivot['hour'] = pivot['last_update'].dt.hour
pivot['pm25_high'] = pivot['PM2.5'] > 60
pivot['pm25_cat'] = pd.qcut(pivot['PM2.5'], q=2, labels=['Low', 'High'])
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']

# ------------------ Correlation Heatmap ------------------
plt.figure(figsize=(10,6))
sns.heatmap(pivot[pollutants].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation between Pollutants")
plt.show()

# ------------------ Scatter Plot (PM10 vs PM2.5) ------------------
sns.scatterplot(data=pivot, x='PM10', y='PM2.5')
plt.title("PM10 vs PM2.5")
plt.grid(True)
plt.show()

# ------------------ Boxplot: PM2.5 Category ------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=pivot, x='pm25_cat', y='PM2.5')
plt.title("PM2.5 Levels by Category")
plt.grid(True)
plt.show()

# ------------------ Boxplot: PM2.5 by Top States ------------------
plt.figure(figsize=(14, 6))
top_states = pivot['state'].value_counts().head(8).index
sns.boxplot(data=pivot[pivot['state'].isin(top_states)], x='state', y='PM2.5')
plt.title("PM2.5 in Different States (Top 8)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# ------------------ Heatmap: Top 10 Polluted Cities ------------------
avg_pollution = pivot.groupby('city')[pollutants].mean()
top_cities = avg_pollution.sort_values('PM2.5', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.heatmap(top_cities, annot=True, cmap='YlOrRd')
plt.title("Avg Pollutant Levels in Top 10 Polluted Cities")
plt.tight_layout()
plt.show()

# ------------------ NEW INSIGHT: Pollutant Contribution by City ------------------
avg_by_city = pivot.groupby('city')[pollutants].mean()
top5_cities = avg_by_city.sort_values('PM2.5', ascending=False).head(5)

top5_cities.plot(kind='bar', stacked=True, colormap='Set2', figsize=(12, 6))
plt.title("Pollutant Contribution in Top 5 Polluted Cities")
plt.ylabel("Average Pollutant Levels (µg/m³)")
plt.xlabel("City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()


# ------------------ NEW INSIGHT: Top 10 Polluted Stations ------------------
station_pm25 = pivot.groupby('station')['PM2.5'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
station_pm25.plot(kind='bar', color='salmon')
plt.title("Top 10 Most Polluted Monitoring Stations (PM2.5)")
plt.ylabel("PM2.5 (µg/m³)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ PM2.5 Compliance Check ------------------
unsafe_percent = pivot['pm25_high'].mean() * 100
print(f"Percentage of PM2.5 values above safe limit (60 µg/m³): {unsafe_percent:.2f}%")

# ------------------ Function: Pollutant Trend by City ------------------
def plot_trend(df, city, pollutant):
    city_data = df[df['city'] == city]
    if city_data.empty or pollutant not in df.columns:
        print("Data not available.")
        return
    daily_avg = city_data.resample('D', on='last_update')[pollutant].mean()
    daily_avg.plot(marker='o')
    plt.title(f"{pollutant} Daily Trend in {city}")
    plt.ylabel(pollutant)
    plt.xlabel("Date")
    plt.grid(True)
    plt.show()

# plot_trend(pivot, "Delhi", "PM2.5")
