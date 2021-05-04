import pandas as pd
import matplotlib.pyplot as plt
from options_futures_expirations_v3 import month_to_quarter_shifter
from pathlib import Path
plt.style.use('cboe-fivethirtyeight')

# General workflow:
# 1) Go to https://bi.cboe.com/#/views/SettlementDataPull/Dashboard
#    and Download->Data->Summary tab->Download all rows as a text file.
#    Save it in DOWNLOADS_DIR as f'Unified_Data_Table_data_{USE_DATE}.csv'.
# 2) Run this script and check DOWNLOADS_DIR for f'xxxx.csv' (probably multiple CSVs).

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
USE_DATE = '2021-04-30'
PRODUCTS = ['IBHY', 'IBIG', 'VX', 'VXM']  # Default ['IBHY', 'IBIG', 'VX', 'VXM']
MULTIPLIER_DICT = {'IBHY': 1000, 'IBIG': 1000, 'VX': 1000, 'VXM': 100}

###############################################################################

# Load
DASHBOARD_DOWNLOAD_FILE = f'Unified_Data_Table_data_{USE_DATE}.csv'
settle_data = pd.read_csv(DOWNLOADS_DIR + DASHBOARD_DOWNLOAD_FILE,
                          parse_dates=['Date', 'Expiry'], thousands=',')
settle_data['Weekly'] = settle_data['Symbol'].apply(lambda s: s[-5:-3].isnumeric())     # Test for week number
settle_data_trim = settle_data.drop(['Block and Standard', 'Block and TAS',
                                     'ECRP and Standard', 'ECRP and TAS'], axis=1)
# settle_data_df = settle_data_trim.pivot(index=['Product', 'Date', 'Expiry', 'Symbol', 'Term', 'DTE'],
#                                         columns='Measure Names', values='Measure Values')
settle_data_df = settle_data_trim.pivot_table(index=['Product', 'Date', 'Expiry', 'Symbol', 'Term', 'DTE', 'Weekly'],
                                              columns='Measure Names', values='Measure Values')
# settle_data_df = (settle_data_trim.set_index(['Product', 'Date', 'Expiry', 'Symbol', 'Term', 'DTE', 'Measure Names'])
#                   .squeeze().unstack())
SETTLE_COLUMN_ORDER = ['Settle', 'Volume', 'Standard', 'TAS',
                       'Block', 'ECRP', 'Spreads',
                       'OI', 'Open', 'High', 'Low', 'Close']
settle_data_df = settle_data_df[SETTLE_COLUMN_ORDER].copy()     # Enforce column order
# Create 'Notional' column by combining settle and volume
for product in PRODUCTS:
    settle_data_df.loc[product, 'Notional'] = \
        (settle_data_df.loc[product, 'Settle'] * settle_data_df.loc[product, 'Volume']
         * MULTIPLIER_DICT[product]).values
# Split DataFrame by product
settle_data_dict = {product: settle_data_df.xs(product) for product in PRODUCTS}


def lookback_rank(ser):
    """ Apply Helper: Take in limited series and return percentile rank of last (target) element
    :param ser: pd.Series framed by .rolling() framework
    :return: rank (in percent) of last element of series within the series
    """
    return ser.rank(pct=True)[-1]


def generate_volume_percentiles(data, field):
    # Extract volume-based field up to daily level - aggregate volume by summing
    daily_field = data.groupby(['Date'])[field].sum()
    if field in ['Standard', 'TAS', 'Block', 'ECRP', 'Spreads']:
        daily_volume = data.groupby(['Date'])['Volume'].sum()
    else:
        daily_volume = None     # Not applicable/useful
    # Attach month, quarter, and year to each row to allow groupby()
    yearmonth_col = pd.to_datetime(daily_field.index.strftime('%Y-%m'))
    yearquarter_col = \
        pd.to_datetime(daily_field.index.to_series()
                       .apply(lambda ts: f'{ts.year}-{month_to_quarter_shifter(ts.month):02}'))
    year_col = pd.to_datetime(daily_field.index.strftime('%Y'))
    daily_field_df = pd.DataFrame({field: daily_field, 'Total Volume': daily_volume,
                                   'Month': yearmonth_col, 'Quarter': yearquarter_col, 'Year': year_col})

    # No aggregation - daily
    # NOTE: at daily level, "volume" and "ADV" are conceptually the same
    daily_percentile = daily_field.rank(pct=True)  # Percentile over full history
    daily_percentile_rolling = daily_field.rolling(252).apply(lookback_rank, raw=False)     # Percentile rolling 1-year
    # Percent change stats
    daily_change = daily_field.pct_change()
    daily_change_pos = daily_change[daily_change > 0]
    daily_change_pos_percentile = daily_change_pos.rank(pct=True)
    daily_change_neg = daily_change[daily_change < 0]
    daily_change_neg_percentile = (-daily_change_neg).rank(pct=True)
    daily_change_posneg_percentile_df = pd.DataFrame({'Increase': daily_change_pos_percentile,
                                                      'Decrease': daily_change_neg_percentile})
    if field in ['Standard', 'TAS', 'Block', 'ECRP', 'Spreads']:
        # Do the same analysis on % total volume if field is a subset of volume
        daily_percent_total = daily_field / daily_volume
        daily_percent_total_percentile = daily_percent_total.rank(pct=True)
        daily_percent_total_percentile_rolling = daily_percent_total.rolling(252).apply(lookback_rank, raw=False)
        daily_percent_total_change = daily_percent_total.pct_change()
        daily_percent_total_change_pos = daily_percent_total_change[daily_percent_total_change > 0]
        daily_percent_total_change_pos_percentile = daily_percent_total_change_pos.rank(pct=True)
        daily_percent_total_change_neg = daily_percent_total_change[daily_percent_total_change < 0]
        daily_percent_total_change_neg_percentile = (-daily_percent_total_change_neg).rank(pct=True)
        daily_percent_total_change_posneg_percentile_df = \
            pd.DataFrame({'Increase': daily_percent_total_change_pos_percentile,
                          'Decrease': daily_percent_total_change_neg_percentile})
    else:
        daily_percent_total = None
        daily_percent_total_percentile = None
        daily_percent_total_percentile_rolling = None
        daily_percent_total_change = None
        daily_percent_total_change_posneg_percentile_df = None
    # Initialize storage dict
    if field in ['Standard', 'TAS', 'Block', 'ECRP', 'Spreads']:
        field_percentile_dict = {
            'Daily': {
                'Sum': daily_field,
                'ADV': daily_field,
                'Percentile (All)': daily_percentile,
                'Percentile (Last 252)': daily_percentile_rolling,
                'ADV Change': daily_change,
                'ADV Change Percentile +': daily_change_posneg_percentile_df['Increase'],
                'ADV Change Percentile -': daily_change_posneg_percentile_df['Decrease'],
                '% Total': daily_percent_total,
                '% Total Percentile (All)': daily_percent_total_percentile,
                '% Total Percentile (Last 252)': daily_percent_total_percentile_rolling,
                '% Total Change': daily_percent_total_change,
                '% Total Change Percentile +': daily_percent_total_change_posneg_percentile_df['Increase'],
                '% Total Change Percentile -': daily_percent_total_change_posneg_percentile_df['Decrease']
            }
        }
    else:
        field_percentile_dict = {
            'Daily': {
                'Sum': daily_field,
                'ADV': daily_field,
                'Percentile (All)': daily_percentile,
                'Percentile (Last 252)': daily_percentile_rolling,
                'ADV Change': daily_change,
                'ADV Change Percentile +': daily_change_posneg_percentile_df['Increase'],
                'ADV Change Percentile -': daily_change_posneg_percentile_df['Decrease']
            }
        }

    # Aggregate daily volumes to monthly/quarterly/yearly
    for agg_level, lookback_n in zip(['Month', 'Quarter', 'Year'], [12, 4, 2]):
        agg = f'{agg_level}ly'
        field_percentile_dict[agg] = {}
        # NOTE: this was originally written with field='Volume', and I've generalized it to work
        #       with any field. "average daily volume" (ADV) is therefore too specific, but I won't
        #       change that for fear of confusion. so think of it as "average daily value" instead.
        # NOTE: ADV is what should be ranked - different months have different numbers of days, so sum doesn't work
        field_percentile_dict[agg]['Sum'] = daily_field_df.groupby(agg_level)[field].sum()
        field_percentile_dict[agg]['ADV'] = daily_field_df.groupby(agg_level)[field].mean()
        field_percentile_dict[agg]['Percentile (All)'] = \
            (field_percentile_dict[agg]['ADV']
             .rank(pct=True))
        field_percentile_dict[agg][f'Percentile (Last {lookback_n})'] = \
            (field_percentile_dict[agg]['ADV']
             .rolling(lookback_n).apply(lookback_rank, raw=False))  # Percentile rolling
        # Percent change stats
        agg_change = field_percentile_dict[agg]['ADV'].pct_change()
        field_percentile_dict[agg]['ADV Change'] = agg_change
        agg_change_pos, agg_change_neg = agg_change[agg_change > 0], agg_change[agg_change < 0]
        agg_change_posneg_percentile_df = \
            pd.DataFrame({'Increase': agg_change_pos.rank(pct=True),
                          'Decrease': (-agg_change_neg).rank(pct=True)})
        field_percentile_dict[agg]['ADV Change Percentile +'], field_percentile_dict[agg]['ADV Change Percentile -'] = \
            agg_change_posneg_percentile_df['Increase'], agg_change_posneg_percentile_df['Decrease']
        if field in ['Standard', 'TAS', 'Block', 'ECRP', 'Spreads']:
            # Do the same analysis on % total volume if field is a subset of volume
            field_percentile_dict[agg]['% Total'] = \
                field_percentile_dict[agg]['Sum'] / daily_field_df.groupby(agg_level)['Total Volume'].sum()
            field_percentile_dict[agg]['% Total Percentile (All)'] = \
                (field_percentile_dict[agg]['% Total']
                 .rank(pct=True))
            field_percentile_dict[agg][f'% Total Percentile (Last {lookback_n})'] = \
                (field_percentile_dict[agg]['% Total']
                 .rolling(lookback_n).apply(lookback_rank, raw=False))  # Percentile rolling
            # Percent change stats
            agg_change = field_percentile_dict[agg]['% Total'].pct_change()
            field_percentile_dict[agg]['% Total Change'] = agg_change
            agg_change_pos, agg_change_neg = agg_change[agg_change > 0], agg_change[agg_change < 0]
            agg_change_posneg_percentile_df = \
                pd.DataFrame({'Increase': agg_change_pos.rank(pct=True),
                              'Decrease': (-agg_change_neg).rank(pct=True)})
            field_percentile_dict[agg]['% Total Change Percentile +'], \
                field_percentile_dict[agg]['% Total Change Percentile -'] = \
                agg_change_posneg_percentile_df['Increase'], agg_change_posneg_percentile_df['Decrease']
    return field_percentile_dict


###############################################################################

product_dict = {}
for product in PRODUCTS:
    # Select product and run percentiles on each volume-related field
    # NOTE: the 6 Order Fill fields - Volume, Standard, TAS, Block, ECRP, Spreads - only go back to 2018-03-20;
    #       the 6 Settlement+OI fields - Settle, OI, Open, High, Low, Close - go all the way back to 2013-05-20
    # NOTE: 1 additional constructed field - Notional - uses Volume and Settle; informationally it is closer to Volume
    product_data = settle_data_dict[product]
    ORDERFILL_FIELDS = ['Volume', 'Standard', 'TAS', 'Block', 'ECRP', 'Spreads']
    SETTLEOI_FIELDS = ['Settle', 'OI', 'Open', 'High', 'Low', 'Close']
    CONSTRUCTED_FIELDS = ['Notional']
    product_orderfill = product_data[ORDERFILL_FIELDS]
    product_settleoi = product_data[SETTLEOI_FIELDS]
    product_constructed = product_data[CONSTRUCTED_FIELDS]
    # Crop NaNs from legacy data clash
    modern_start = product_orderfill['Volume'].first_valid_index()[0]   # Volume field used as representative
    product_orderfill = product_orderfill.loc[modern_start:]
    product_constructed = product_constructed.loc[modern_start:]    # Constructed field limited by Order Fill history

    # Initialize storage dictionary
    percentile_dict = {}

    # Run percentiles on each volume-based field from Order Fill
    for i_field in ORDERFILL_FIELDS:
        percentile_dict[i_field] = generate_volume_percentiles(product_orderfill, i_field)
    # product_orderfill_subsplits = product_orderfill.copy()
    # for i_field in ['Standard', 'TAS', 'Block', 'ECRP', 'Spreads']:
    #     product_orderfill_subsplits[i_field] = product_orderfill[i_field] / product_orderfill['Volume']
    #     percentile_dict[i_field + ' Percent'] = generate_volume_percentiles(product_orderfill_subsplits, i_field)
    #     for i_agg in ['Monthly', 'Quarterly', 'Yearly']:
    #         percentile_dict[i_field + ' Percent'][i_agg]['Sum'] = None  # This field makes no sense for percent splits

    # Run percentiles on OI field from Settlement+OI. OI's aggregation is very specific: it must be summed up to
    # daily level (since the roll drives OI at expiry-level), then averaged up to monthly/quarterly/yearly
    percentile_dict['OI'] = generate_volume_percentiles(product_settleoi, 'OI')
    for i_agg in ['Monthly', 'Quarterly', 'Yearly']:
        percentile_dict['OI'][i_agg]['Sum'] = None    # This field makes no sense for OI

    # Run percentiles on constructed Notionals
    percentile_dict['Notional'] = generate_volume_percentiles(product_constructed, 'Notional')

    # Store in higher level storage dictionary
    product_dict[product] = percentile_dict

###############################################################################

# Write to disk for general public access
for product in PRODUCTS:
    export_path = Path(f'P:/PrdDevSharedDB/Cboe Futures Volume Historical Percentiles/{USE_DATE}/{product}/')
    export_path.mkdir(parents=True, exist_ok=True)
    print(f"{export_path} created (if not before).")
    for i_field in ['Volume', 'Standard', 'TAS', 'Block', 'ECRP', 'Spreads', 'OI', 'Notional']:
        with pd.ExcelWriter(export_path / f'{i_field}.xlsx', datetime_format='YYYY-MM-DD') as writer:
            for i_agg in ['Daily', 'Monthly', 'Quarterly', 'Yearly']:
                agg_df = pd.DataFrame(product_dict[product][i_field][i_agg])
                agg_df.to_excel(writer, sheet_name=i_agg, freeze_panes=(1, 1))
