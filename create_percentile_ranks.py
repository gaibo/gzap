import pandas as pd
import matplotlib.pyplot as plt
from options_futures_expirations_v3 import month_to_quarter_shifter
plt.style.use('cboe-fivethirtyeight')

# General workflow:
# 1) Go to https://bi.cboe.com/#/views/SettlementDataPull/Dashboard
#    and Download->Data->Summary tab->Download all rows as a text file.
#    Save it in DOWNLOADS_DIR as f'Unified_Data_Table_data_{USE_DATE}.csv'.
# 2) Run this script and check DOWNLOADS_DIR for f'xxxx.csv' (probably multiple CSVs).

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
USE_DATE = '2021-03-30'
PRODUCTS = ['IBHY', 'IBIG', 'VX', 'VXM']  # Default ['IBHY', 'IBIG', 'VX', 'VXM']

###############################################################################

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


def lookback_rank(ser):
    """ Apply Helper: Take in limited series and return percentile rank of last (target) element
    :param ser: pd.Series framed by .rolling() framework
    :return: rank (in percent) of last element of series within the series
    """
    return ser.rank(pct=True)[-1]


###############################################################################

# Select product
# NOTE: the 6 Order Fill fields - Volume, Standard, TAS, Block, ECRP, Spreads - only go back to 2018-03-20;
#       the 6 Settlement+OI fields - Settle, OI, Open, High, Low, Close - go all the way back to 2013-05-20
product = 'VX'
product_data = settle_data_dict[product]
ORDERFILL_FIELDS = ['Volume', 'Standard', 'TAS', 'Block', 'ECRP', 'Spreads']
SETTLEOI_FIELDS = ['Settle', 'OI', 'Open', 'High', 'Low', 'Close']
product_orderfill = product_data[ORDERFILL_FIELDS]
product_settleoi = product_data[SETTLEOI_FIELDS]
# Crop NaNs from legacy data clash
modern_start = product_orderfill['Volume'].first_valid_index()[0]   # Volume field used as representative
product_orderfill = product_orderfill.loc[modern_start:]

# Run percentiles on each field
for field in ORDERFILL_FIELDS:
    # Extract volume-based field up to daily level - aggregate volume by summing
    daily_field = product_orderfill.groupby(['Date'])[field].sum()
    # Attach month, quarter, and year to each row to allow groupby()
    yearmonth_col = pd.to_datetime(daily_field.index.strftime('%Y-%m'))
    yearquarter_col = \
        pd.to_datetime(daily_field.index.to_series()
                       .apply(lambda ts: f'{ts.year}-{month_to_quarter_shifter(ts.month):02}'))
    year_col = pd.to_datetime(daily_field.index.strftime('%Y'))
    daily_field_df = pd.DataFrame({field: daily_field, 'Month': yearmonth_col,
                                   'Quarter': yearquarter_col, 'Year': year_col})
    # No aggregation - daily
    # NOTE: at daily level, "volume" and "ADV" are conceptually the same
    daily_percentile = daily_field.rank(pct=True)    # Percentile over full history
    daily_percentile_1_year = daily_field.rolling(256).apply(lookback_rank, raw=False)    # Percentile rolling 1-year
    # Aggregate to monthly
    # NOTE: this was originally written with field='Volume', and I've generalized it to work
    #       with any field. "average daily volume" (ADV) is therefore too specific, but I won't
    #       change that for fear of confusion. so think of it as "average daily value" instead.
    # NOTE: ADV is what should be ranked - different months have different numbers of days, so sum doesn't work
    monthly_field = daily_field_df.groupby('Month')[field].sum()
    monthly_adv = daily_field_df.groupby('Month')[field].mean()
    monthly_percentile = monthly_adv.rank(pct=True)
    monthly_percentile_1_year = monthly_adv.rolling(12).apply(lookback_rank, raw=False)    # Percentile rolling 1-year
    # Aggregate to quarterly
    quarterly_field = daily_field_df.groupby('Quarter')[field].sum()
    quarterly_adv = daily_field_df.groupby('Quarter')[field].mean()
    quarterly_percentile = quarterly_adv.rank(pct=True)
    quarterly_percentile_1_year = quarterly_adv.rolling(4).apply(lookback_rank, raw=False)  # Percentile rolling 1-year
    # Aggregate to yearly
    yearly_field = daily_field_df.groupby('Year')[field].sum()
    yearly_adv = daily_field_df.groupby('Year')[field].mean()
    yearly_percentile = yearly_adv.rank(pct=True)
    yearly_percentile_2_year = yearly_adv.rolling(2).apply(lookback_rank, raw=False)  # Percentile rolling 2-year
