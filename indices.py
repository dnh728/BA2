import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Define the start and end date
start_date = datetime.strptime('10/01/2019', '%m/%d/%Y')  # Start of Q4 2019
end_date = datetime.strptime('09/30/2023', '%m/%d/%Y')  # End of Q3 2023

# Load the data
rut = pd.read_csv('RUT.csv')
spx = pd.read_csv('SPX.csv')
comp = pd.read_csv('COMP.csv')
djia = pd.read_csv('DJIA.csv')

# Convert the 'Date' column to datetime format
rut['Date'] = pd.to_datetime(rut['Date'], format='%m/%d/%Y')
spx['Date'] = pd.to_datetime(spx['Date'], format='%m/%d/%Y')
comp['Date'] = pd.to_datetime(comp['Date'], format='%m/%d/%Y')
djia['Date'] = pd.to_datetime(djia['Date'], format='%m/%d/%Y')

# Store the dataframes in a dictionary
dataframes = {'RUT': rut, 'SPX': spx, 'COMP': comp, 'DJIA': djia}

# Create a figure and a grid of subplots
fig, axs = plt.subplots(1, 4, figsize=(15, 8))

# Flatten the array of subplots
axs = axs.flatten()

# Define a custom formatter function for the x-axis
def quarter_formatter(x, pos=None):
    date = mdates.num2date(x)
    quarter_number = (date.month - 1) // 3 + 1
    return f'Q{quarter_number} {date.year}'

# Iterate over the dataframes and axes to plot each graph
for ax, (symbol, df) in zip(axs, dataframes.items()):
    # Filter the dataframe based on the date range
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    ax.plot(df['Date'], df['Close/Last'], label=symbol, linewidth=1)
    
    # Set x-axis limits to the defined date range
    ax.set_xlim(start_date, end_date)
    
    # Format the x-axis to show quarters and years
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(quarter_formatter))
    
    # Set labels and title
    ax.set_ylabel('Price (USD)')
    ax.set_title(symbol)  # Set title to the symbol name
    
    # Rotate x-axis labels for better readability and adjust their position
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    plt.gcf().autofmt_xdate()  # Automatically adjust x-axis labels
    
    ax.legend()
    ax.grid(False)

# Adjust layout for better readability
plt.tight_layout()

# Save the plot to a file
plt.savefig('indices_price_individual.png')

# Show the plot
plt.show()

