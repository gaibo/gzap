import pandas as pd
import matplotlib.pyplot as plt
from options_futures_expirations_v3 import month_to_quarter_shifter
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

USE_DATE = '2021-03-29'
PRODUCTS = ['VXM', 'IBHY', 'IBIG', 'VX']  # Default ['VXM', 'IBHY', 'IBIG']

# Quick scratch
# Load
product = 'VX'
new_accounts_data = pd.read_csv(DOWNLOADS_DIR + f'{USE_DATE}_{product}_new_accounts.csv',
                                parse_dates=['Trade Date'])
daily_volume = new_accounts_data.groupby('Trade Date')['Size'].sum()/2
daily_volume.name = 'Volume'

# Allow daily volumes to be groupable with additional columns
yearmonth_col = pd.to_datetime(daily_volume.index.strftime('%Y-%m'))
yearquarter_col = \
    pd.to_datetime(daily_volume.index.to_series()
                   .apply(lambda ts: f'{ts.year}-{month_to_quarter_shifter(ts.month):02}'))
daily_volume_df = pd.DataFrame({'Volume': daily_volume,
                                'Month': yearmonth_col, 'Quarter': yearquarter_col})
# No aggregation - daily
# NOTE: at daily level, "volume" and "ADV" is the same
daily_percentile = daily_volume.rank(pct=True)    # Percentile over full history
def lookback_rank(ser):
    return ser.rank(pct=True)[-1]   # Take in limited series and return percentile rank of last element
daily_percentile_1_year = daily_volume.rolling(256).apply(lookback_rank, raw=False)    # Percentile rolling 1-year
# Aggregate to monthly
# NOTE: ADV is what should be ranked - different months have different numbers of days, so sum is not good
monthly_volume = daily_volume_df.groupby('Month')['Volume'].sum()
monthly_adv = daily_volume_df.groupby('Month')['Volume'].mean()
monthly_percentile = monthly_adv.rank(pct=True)
monthly_percentile_1_year = monthly_adv.rolling(12).apply(lookback_rank, raw=False)    # Percentile rolling 1-year
# Aggregate to quarterly
quarterly_volume = daily_volume_df.groupby('Quarter')['Volume'].sum()
quarterly_adv = daily_volume_df.groupby('Quarter')['Volume'].mean()
quarterly_percentile = quarterly_adv.rank(pct=True)
quarterly_percentile_1_year = quarterly_adv.rolling(4).apply(lookback_rank, raw=False)    # Percentile rolling 1-year
