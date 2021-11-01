import pandas as pd
import matplotlib.pyplot as plt
from options_futures_expirations_v3 import month_to_quarter_shifter
from pathlib import Path
plt.style.use('cboe-fivethirtyeight')

# General workflow:
# 1) Go to https://bi.cboe.com/#/views/SettlementDataPull/Dashboard
#    and Download->Data->Summary tab->Download all rows as a text file.
#    Save it in DOWNLOADS_DIR as f'Unified_Data_Table_data_{USE_DATE}.csv'.
# 2) Run this script and check EXPORT_DIR for folder named USE_DATE which contains XLSXs.

DOWNLOADS_DIR = 'C:/Users/gzhang/OneDrive - CBOE/Downloads/'
USE_DATE = '2021-10-29'
# Default: ['IBHY', 'IBIG', 'VX', 'VXM', 'AMW', 'AMB1', 'AMB3', 'AMT1']
PRODUCTS = ['IBHY', 'IBIG', 'VX', 'VXM', 'AMW', 'AMB1', 'AMB3', 'AMT1']
MULTIPLIER_DICT = {'IBHY': 1000, 'IBIG': 1000, 'VX': 1000, 'VXM': 100, 'AMW': 35, 'AMB1': 50, 'AMB3': 25, 'AMT1': 25}
EXPORT_DIR = 'P:/PrdDevSharedDB/Cboe Futures Historical Percentiles/'

###############################################################################

# Load
DASHBOARD_DOWNLOAD_FILE = f'Unified_Data_Table_data_{USE_DATE}.csv'
settle_data = pd.read_csv(DOWNLOADS_DIR + DASHBOARD_DOWNLOAD_FILE,
                          parse_dates=['Date', 'Expiry'], thousands=',')
# Label Weeklys (VIX futures that start with VX01, VX49, etc. instead of VX; AMW futures)
WEEKLY_PRODUCTS = ['VX', 'AMW']
symbol_start = settle_data['Symbol'].apply(lambda s: s.split('/')[0])   # Cut off end, which designates month
symbol_vx = symbol_start.apply(lambda s: s[:2]) == 'VX'
symbol_vx_weeknumbers = symbol_start.apply(lambda s: s[-2:]).str.isnumeric()
symbol_amw = symbol_start.apply(lambda s: s[:3]) == 'AMW'
settle_data['Weekly'] = (symbol_vx & symbol_vx_weeknumbers) | symbol_amw
# Remove Tableau tooltip data (more than we need, for now at least)
settle_data_trim = settle_data.drop(['Block and Standard', 'Block and TAS',
                                     'ECRP and Standard', 'ECRP and TAS'], axis=1)
# Pivot and clean columns into DataFrame
INDEX_COLS = ['Product', 'Date', 'Expiry', 'Symbol', 'Term', 'DTE', 'Weekly']
# .pivot() same as .pivot_table() but strict on unique indexes - if ever there are duplicates in data, data is broken
settle_data_df = settle_data_trim.pivot(index=INDEX_COLS, columns='Measure Names', values='Measure Values')
CONTENT_COLS = ['Settle', 'Volume', 'Standard', 'TAS',
                'Block', 'ECRP', 'Spreads', 'Customer', 'RTH', 'GTH',
                'OI', 'Open', 'High', 'Low', 'Close']
settle_data_df = settle_data_df[CONTENT_COLS].copy()    # Enforce column order
# Create 'Notional' column by combining settle and volume
for product in PRODUCTS:
    settle_data_df.loc[product, 'Notional'] = \
        (settle_data_df.loc[product, 'Settle'] * settle_data_df.loc[product, 'Volume']
         * MULTIPLIER_DICT[product]).values
# Establish groupings for different analyses
VOLUME_REFERENCE_COLS = ['Volume', 'Notional', 'OI']    # Most commonly referenced stats
VOLUME_SUBCAT_COLS = ['Standard', 'TAS', 'Block', 'ECRP',
                      'Spreads', 'Customer', 'RTH', 'GTH']  # Subcategories; consider % of total
VOLUME_COLS = VOLUME_REFERENCE_COLS + VOLUME_SUBCAT_COLS
PRICE_COLS = ['Settle', 'Open', 'High', 'Low', 'Close']
# Split DataFrame by product
settle_data_dict = {product: settle_data_df.xs(product) for product in PRODUCTS}


def lookback_rank(ser):
    """ Apply Helper: Take in limited series and return percentile rank of last (target) element
    :param ser: pd.Series framed by .rolling() framework
    :return: rank (in percent) of last element of series within the series
    """
    return ser.rank(pct=True, method='min')[-1]


def generate_volume_percentiles(data, field):
    # Extract volume-based field up to daily level - aggregate volume by summing
    daily_field = data.groupby(['Date'])[field].sum()
    if field in VOLUME_SUBCAT_COLS:
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
    daily_percentile = daily_field.rank(pct=True, method='min')  # Percentile over full history
    daily_percentile_rolling = daily_field.rolling(252).apply(lookback_rank, raw=False)     # Percentile rolling 1-year
    # Percent change stats
    daily_change = daily_field.pct_change()
    daily_change_pos = daily_change[daily_change > 0]
    daily_change_pos_percentile = daily_change_pos.rank(pct=True, method='min')
    daily_change_neg = daily_change[daily_change < 0]
    daily_change_neg_percentile = (-daily_change_neg).rank(pct=True, method='min')
    daily_change_posneg_percentile_df = pd.DataFrame({'Increase': daily_change_pos_percentile,
                                                      'Decrease': daily_change_neg_percentile})
    if field in VOLUME_SUBCAT_COLS:
        # Do the same analysis on % total volume if field is a subset of volume
        daily_percent_total = daily_field / daily_volume
        daily_percent_total_percentile = daily_percent_total.rank(pct=True, method='min')
        daily_percent_total_percentile_rolling = daily_percent_total.rolling(252).apply(lookback_rank, raw=False)
        daily_percent_total_change = daily_percent_total.pct_change()
        daily_percent_total_change_pos = daily_percent_total_change[daily_percent_total_change > 0]
        daily_percent_total_change_pos_percentile = daily_percent_total_change_pos.rank(pct=True, method='min')
        daily_percent_total_change_neg = daily_percent_total_change[daily_percent_total_change < 0]
        daily_percent_total_change_neg_percentile = (-daily_percent_total_change_neg).rank(pct=True, method='min')
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
    if field in VOLUME_SUBCAT_COLS:
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
             .rank(pct=True, method='min'))
        field_percentile_dict[agg][f'Percentile (Last {lookback_n})'] = \
            (field_percentile_dict[agg]['ADV']
             .rolling(lookback_n).apply(lookback_rank, raw=False))  # Percentile rolling
        # Percent change stats
        agg_change = field_percentile_dict[agg]['ADV'].pct_change()
        field_percentile_dict[agg]['ADV Change'] = agg_change
        agg_change_pos, agg_change_neg = agg_change[agg_change > 0], agg_change[agg_change < 0]
        agg_change_posneg_percentile_df = \
            pd.DataFrame({'Increase': agg_change_pos.rank(pct=True, method='min'),
                          'Decrease': (-agg_change_neg).rank(pct=True, method='min')})
        field_percentile_dict[agg]['ADV Change Percentile +'], field_percentile_dict[agg]['ADV Change Percentile -'] = \
            agg_change_posneg_percentile_df['Increase'], agg_change_posneg_percentile_df['Decrease']
        if field in VOLUME_SUBCAT_COLS:
            # Do the same analysis on % total volume if field is a subset of volume
            field_percentile_dict[agg]['% Total'] = \
                field_percentile_dict[agg]['Sum'] / daily_field_df.groupby(agg_level)['Total Volume'].sum()
            field_percentile_dict[agg]['% Total Percentile (All)'] = \
                (field_percentile_dict[agg]['% Total']
                 .rank(pct=True, method='min'))
            field_percentile_dict[agg][f'% Total Percentile (Last {lookback_n})'] = \
                (field_percentile_dict[agg]['% Total']
                 .rolling(lookback_n).apply(lookback_rank, raw=False))  # Percentile rolling
            # Percent change stats
            agg_change = field_percentile_dict[agg]['% Total'].pct_change()
            field_percentile_dict[agg]['% Total Change'] = agg_change
            agg_change_pos, agg_change_neg = agg_change[agg_change > 0], agg_change[agg_change < 0]
            agg_change_posneg_percentile_df = \
                pd.DataFrame({'Increase': agg_change_pos.rank(pct=True, method='min'),
                              'Decrease': (-agg_change_neg).rank(pct=True, method='min')})
            field_percentile_dict[agg]['% Total Change Percentile +'], \
                field_percentile_dict[agg]['% Total Change Percentile -'] = \
                agg_change_posneg_percentile_df['Increase'], agg_change_posneg_percentile_df['Decrease']
    return field_percentile_dict


###############################################################################

# Independently of "volume fields vs. volume subcategory fields vs. price fields", we also must account for
# "Tableau Order Fill fields vs. Settlement+OI fields", as the latter's data goes back further in history
# NOTE: must manually make this split because currently code relies on Tableau dashboard download to determine trade
#       days; where we run into issue is that for VX the code would think since OI goes to 2013, there was 0 volume for
#       every month from 2013 to 2018. I should patch this so we can take out this clunky manual distinction
# NOTE: the 9 Order Fill fields - Volume, Standard, TAS, Block, ECRP, Spreads, Customer, RTH, GTH -
#         only go back to 2018-02-26;
#       the 6 Settlement+OI fields - Settle, OI, Open, High, Low, Close -
#         go all the way back to 2013-05-20;
#       the 1 constructed field - Notional -
#         uses Volume and Settle so only goes back to 2018-02-26; informationally it is grouped with Volume
ORDERFILL_FIELDS = ['Volume', 'Standard', 'TAS', 'Block', 'ECRP', 'Spreads', 'Customer', 'RTH', 'GTH']
SETTLEOI_FIELDS = ['Settle', 'OI', 'Open', 'High', 'Low', 'Close']
CONSTRUCTED_FIELDS = ['Notional']

product_dict = {}
for product in PRODUCTS:
    # Select product and run percentiles on each volume-related field
    product_data = settle_data_dict[product]
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

    # Run percentiles on OI field from Settlement+OI. OI's aggregation is very specific: it must be summed up to
    # daily level (since the roll drives OI at expiry-level), then averaged up to monthly/quarterly/yearly
    percentile_dict['OI'] = generate_volume_percentiles(product_settleoi, 'OI')
    for i_agg in ['Monthly', 'Quarterly', 'Yearly']:
        percentile_dict['OI'][i_agg]['Sum'] = None    # This field makes no sense for OI

    # Run percentiles on constructed Notional field
    percentile_dict['Notional'] = generate_volume_percentiles(product_constructed, 'Notional')

    # Store in higher level storage dictionary
    product_dict[product] = percentile_dict

###############################################################################

# Write to disk for general public access
# NOTE: optimize for Excel usage - can't open multiple files with same name, so do VX_Volume.xlsx
export_root_path = Path(EXPORT_DIR) / USE_DATE
for product in PRODUCTS:
    export_path = export_root_path / product
    export_path_sub = export_path / 'Volume Subcategories'
    export_path_sub.mkdir(parents=True, exist_ok=True)
    print(f"{export_path} established as export path.")
    for i_field in VOLUME_REFERENCE_COLS:
        with pd.ExcelWriter(export_path / f'{product}_{i_field}.xlsx', datetime_format='YYYY-MM-DD') as writer:
            for i_agg in ['Daily', 'Monthly', 'Quarterly', 'Yearly']:
                agg_df = pd.DataFrame(product_dict[product][i_field][i_agg])
                agg_df.to_excel(writer, sheet_name=i_agg, freeze_panes=(1, 1))
    for i_field in VOLUME_SUBCAT_COLS:
        with pd.ExcelWriter(export_path_sub / f'{product}_{i_field}.xlsx', datetime_format='YYYY-MM-DD') as writer:
            for i_agg in ['Daily', 'Monthly', 'Quarterly', 'Yearly']:
                agg_df = pd.DataFrame(product_dict[product][i_field][i_agg])
                agg_df.to_excel(writer, sheet_name=i_agg, freeze_panes=(1, 1))
