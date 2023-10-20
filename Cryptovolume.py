import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Load the data
btc = pd.read_csv('BTCUSDT.csv', skiprows=1)
eth = pd.read_csv('ETHUSDT.csv', skiprows=1)
doge = pd.read_csv('DOGEUSDT.csv', skiprows=1)

# Convert the 'Date' column to datetime format
btc['Date'] = pd.to_datetime(btc['Date'])
eth['Date'] = pd.to_datetime(eth['Date'])
doge['Date'] = pd.to_datetime(doge['Date'])

# Store the dataframes in a dictionary
dataframes = {'BTCUSDT': btc, 'ETHUSDT': eth, 'DOGEUSDT': doge}

# Find the common date range
start_date = max(df['Date'].min() for df in dataframes.values())
end_date = min(df['Date'].max() for df in dataframes.values())

# Plot the data
plt.figure(figsize=(10,6))

for symbol, df in dataframes.items():
    plt.plot(df['Date'], df['Volume USDT'], label=symbol, linewidth=0.7)

# Set x-axis limits to the common date range
plt.xlim(start_date, end_date)

# Define a custom formatter function for the x-axis
def quarter_formatter(x, pos=None):
    date = mdates.num2date(x)
    quarter_number = (date.month - 1) // 3 + 1
    return f'Q{quarter_number} {date.year}'

# Format the x-axis to show quarters and years
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))
plt.gcf().autofmt_xdate()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.ylabel('Volume (USDT)')
plt.title()
plt.legend()
plt.grid(False)

# Save the plot to a file
plt.savefig('crypto_volume.png')

# Show the plot
plt.show()