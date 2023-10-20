
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

btc = pd.read_csv('BTCUSDT.csv', skiprows=1)
eth = pd.read_csv('ETHUSDT.csv', skiprows=1)
doge = pd.read_csv('DOGEUSDT.csv', skiprows=1)

btc['Date'] = pd.to_datetime(btc['Date'])
eth['Date'] = pd.to_datetime(eth['Date'])
doge['Date'] = pd.to_datetime(doge['Date'])
dataframes = {'BTCUSDT': btc, 'ETHUSDT': eth, 'DOGEUSDT': doge}


start_date = max(df['Date'].min() for df in dataframes.values())
end_date = min(df['Date'].max() for df in dataframes.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 3))

def quarter_formatter(x, pos=None):
    date = mdates.num2date(x)
    quarter_number = (date.month - 1) // 3 + 1
    return f'Q{quarter_number} {date.year}'

for ax, (symbol, df) in zip(axs, dataframes.items()):
    ax.plot(df['Date'], df['Close'], label=symbol, linewidth=0.7)

    ax.set_xlim(start_date, end_date)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))
    
    ax.set_ylabel('Price (USD)')
    ax.set_title(symbol)
    
    for label in ax.get_xticklabels():
        label.set_ha('right')
        label.set_rotation(45)
    
    ax.tick_params(axis='x', labelsize=9)
    ax.legend()
    ax.grid(False)

plt.tight_layout()

plt.show()