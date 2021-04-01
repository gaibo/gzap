import pandas as pd
import matplotlib.pyplot as plt
from options_futures_expirations_v3 import month_to_quarter_shifter
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

USE_DATE = '2021-03-30'
PRODUCTS = ['IBHY', 'IBIG', 'VX', 'VXM']  # Default ['VXM', 'IBHY', 'IBIG']

# Quick scratch

# Load
DASHBOARD_DOWNLOAD_FILE = f'Unified_Data_Table_data_{USE_DATE}.csv'
settle_data = pd.read_csv(DOWNLOADS_DIR + DASHBOARD_DOWNLOAD_FILE,
                          parse_dates=['Date', 'Expiry'], thousands=',')
settle_data_trim = settle_data.drop(['Block and Standard', 'Block and TAS',
                                     'ECRP and Standard', 'ECRP and TAS'], axis=1)
# settle_data_df = settle_data_trim.pivot(index=['Product', 'Date', 'Expiry', 'Symbol', 'Term', 'DTE'],
#                                         columns='Measure Names', values='Measure Values')
settle_data_df = settle_data_trim.pivot_table(index=['Product', 'Date', 'Expiry', 'Symbol', 'Term', 'DTE'],
                                              columns='Measure Names', values='Measure Values')
# settle_data_df = (settle_data_trim.set_index(['Product', 'Date', 'Expiry', 'Symbol', 'Term', 'DTE', 'Measure Names'])
#                   .squeeze().unstack())
SETTLE_COLUMN_ORDER = ['Settle', 'Volume', 'Standard', 'TAS',
                       'Block', 'ECRP', 'Spreads',
                       'OI', 'Open', 'High', 'Low', 'Close']
settle_data_df = settle_data_df[SETTLE_COLUMN_ORDER]    # Enforce column order
settle_data_dict = {product: settle_data_df.xs(product) for product in PRODUCTS}

# Select product
# NOTE: the 6 Order Fill fields - Volume, Standard, TAS, Block, ECRP, Spreads - only go back to 2018-03-20;
#       the 6 Settlement+OI fields - Settle, OI, Open, High, Low, Close - go all the way back to 2013-05-20
product = 'VX'
product_data = settle_data_dict[product]
product_orderfill = product_data[['Volume', 'Standard', 'TAS', 'Block', 'ECRP', 'Spreads']]
product_settleoi = product_data[['Settle', 'OI', 'Open', 'High', 'Low', 'Close']]
# Crop NaNs from legacy data clash
modern_start = product_orderfill['Volume'].first_valid_index()[0]
product_orderfill = product_orderfill.loc[modern_start:]
# Extract daily volumes
daily_volume = product_orderfill.groupby(['Date'])['Volume'].sum()

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
    """ Apply Helper: Take in limited series and return percentile rank of last (target) element
    :param ser: pd.Series framed by .rolling() framework
    :return: rank (in percent) of last element of series within the series
    """
    return ser.rank(pct=True)[-1]


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
