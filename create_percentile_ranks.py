import pandas as pd
import matplotlib.pyplot as plt
from options_futures_expirations_v3 import month_to_quarter_shifter
from pathlib import Path
import numpy as np
plt.style.use('cboe-fivethirtyeight')

# General workflow:
# 1) Go to https://bi.cboe.com/#/views/SettlementDataPull/Dashboard
#    and Download->Data->Summary tab->Download all rows as a text file.
#    Save it in DOWNLOADS_DIR as f'Unified_Data_Table_data_{USE_DATE}.csv'.
# 2) Run this script and check EXPORT_DIR for folder named USE_DATE which contains XLSXs.

DOWNLOADS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/')
USE_DATE = '2022-08-31'
# Default: ['IBHY', 'IBIG', 'VX', 'VXM', 'AMW', 'AMB1', 'AMB3', 'AMT1', 'AMT3']
PRODUCTS = ['IBHY', 'IBIG', 'VX', 'VXM', 'AMW', 'AMB1', 'AMB3', 'AMT1', 'AMT3']
MULTIPLIER_DICT = {'IBHY': 1000, 'IBIG': 1000, 'VX': 1000, 'VXM': 100,
                   'AMW': 35, 'AMB1': 50, 'AMB3': 25, 'AMT1': 25, 'AMT3': 25}
EXPORT_DIR = Path('P:/PrdDevSharedDB/Cboe Futures Historical Percentiles/')
EXPORT_DIR_ALLCAT = Path('P:/PrdDevSharedDB/Cboe Futures All Categories Tables/')

###############################################################################

# Load
DASHBOARD_DOWNLOAD_FILE = f'Unified_Data_Table_data_{USE_DATE}.csv'
settle_data = pd.read_csv(DOWNLOADS_DIR / DASHBOARD_DOWNLOAD_FILE,
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


def generate_volume_percentiles_for_field(data, field):
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
                       .apply(lambda ts: f'{ts.year}-{month_to_quarter_shifter(ts.month, left_quarter=True):02}'))
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


def process_fields_for_product(data):
    # Treat different data sources' fields differently
    product_orderfill = data[ORDERFILL_FIELDS]
    product_settleoi = data[SETTLEOI_FIELDS]
    product_constructed = data[CONSTRUCTED_FIELDS]
    # Crop NaNs from legacy data clash
    modern_start = product_orderfill['Volume'].first_valid_index()[0]  # Volume field used as representative
    product_orderfill = product_orderfill.loc[modern_start:]
    product_constructed = product_constructed.loc[modern_start:]  # Constructed field limited by Order Fill history

    # Initialize storage dictionary
    percentile_dict = {}
    # Run percentiles on each volume-based field from Order Fill
    for field in ORDERFILL_FIELDS:
        percentile_dict[field] = generate_volume_percentiles_for_field(product_orderfill, field)
    # Run percentiles on OI field from Settlement+OI. OI's aggregation is very specific: it must be summed up to
    # daily level (since the roll drives OI at expiry-level), then averaged up to monthly/quarterly/yearly
    percentile_dict['OI'] = generate_volume_percentiles_for_field(product_settleoi, 'OI')
    for agg in ['Monthly', 'Quarterly', 'Yearly']:
        percentile_dict['OI'][agg]['Sum'] = None  # This field makes no sense for OI
    # Run percentiles on constructed Notional field
    percentile_dict['Notional'] = generate_volume_percentiles_for_field(product_constructed, 'Notional')
    return percentile_dict


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

PRODUCT_DICT = {}
for product in PRODUCTS:
    # Select product and run percentiles on each volume-related field
    product_data = settle_data_dict[product]
    if product == 'VX':
        # Must do 1) Monthly, 2) Weekly, 3) both together
        vx_monthly_product_data = product_data.xs(False, level='Weekly')
        vx_weekly_product_data = product_data.xs(True, level='Weekly')
        # Process separately and store in higher level storage dictionary
        PRODUCT_DICT[f'{product}_Monthly'] = process_fields_for_product(vx_monthly_product_data)
        PRODUCT_DICT[f'{product}_Weekly'] = process_fields_for_product(vx_weekly_product_data)
        PRODUCT_DICT[product] = process_fields_for_product(product_data)
    elif product == 'AMW':
        # Different from VIX - AMW is a Weekly-only product
        PRODUCT_DICT[f'{product}_Weekly'] = process_fields_for_product(product_data)
    else:
        PRODUCT_DICT[product] = process_fields_for_product(product_data)

###############################################################################

# Write to disk for general public access
# NOTE: optimize for Excel usage - can't open multiple files with same name, so do VX_Volume.xlsx
export_root_path = EXPORT_DIR / USE_DATE
for product_name in PRODUCT_DICT.keys():
    export_path = export_root_path / product_name
    export_path_sub = export_path / 'Volume Subcategories'
    export_path_sub.mkdir(parents=True, exist_ok=True)
    print(f"{export_path} established as export path.")
    for i_field in VOLUME_REFERENCE_COLS:
        with pd.ExcelWriter(export_path / f'{product_name}_{i_field}.xlsx', datetime_format='YYYY-MM-DD') as writer:
            for i_agg in ['Daily', 'Monthly', 'Quarterly', 'Yearly']:
                agg_df = pd.DataFrame(PRODUCT_DICT[product_name][i_field][i_agg])
                agg_df.to_excel(writer, sheet_name=i_agg, freeze_panes=(1, 1))
    for i_field in VOLUME_SUBCAT_COLS:
        with pd.ExcelWriter(export_path_sub / f'{product_name}_{i_field}.xlsx', datetime_format='YYYY-MM-DD') as writer:
            for i_agg in ['Daily', 'Monthly', 'Quarterly', 'Yearly']:
                agg_df = pd.DataFrame(PRODUCT_DICT[product_name][i_field][i_agg])
                agg_df.to_excel(writer, sheet_name=i_agg, freeze_panes=(1, 1))

###############################################################################

# Generate Trade Advisory Committee (TAC) style report

# [CONFIGURE] Manually set month of interest
tac_month = '2022-07'    # Default: '2022-01'
tac_next_incomplete_month = '2022-08'   # Default: '2022-02'

# Piece together report
tac_df = pd.DataFrame(columns=['Month Volume', 'ADV', 'Rank (Last 12)',
                               'Prev Month ADV', 'Month over Month',
                               'Prev Quarter ADV', 'Month over Quarter', 'Prev Year ADV', 'Month over Year',
                               'Current (Incomplete) Month ADV'])
tac_df.index.name = tac_month + ' Stats'
for product_name in ['VX_Monthly', 'VXM', 'IBHY', 'IBIG', 'VX_Weekly', 'AMW_Weekly', 'AMB1', 'AMB3', 'AMT1', 'AMT3']:
    prod_monthly_df = pd.DataFrame(PRODUCT_DICT[product_name]['Volume']['Monthly'])
    prod_quarterly_df = pd.DataFrame(PRODUCT_DICT[product_name]['Volume']['Quarterly'])
    prod_yearly_df = pd.DataFrame(PRODUCT_DICT[product_name]['Volume']['Yearly'])
    # Get full stats for month of interest
    try:
        month_stats = prod_monthly_df.loc[tac_month].squeeze()  # Must .squeeze() because using approx date indexing
    except KeyError:
        # Example: 2022-07 AMB3 futures delisted
        print(f"WARNING: Month {tac_month} not found for {product_name}... delisted?\n"
              f"Product excluded from report!")
        continue
    # Get ADVs for previous periods
    try:
        prev_month_stats = prod_monthly_df.loc[:tac_month].iloc[-2]     # On 2021-12, it will grab 2021-11
        prev_month_adv = prev_month_stats['ADV']
    except IndexError:
        prev_month_adv = np.NaN
    try:
        prev_quarter_stats = prod_quarterly_df.loc[:tac_month].iloc[-2]     # On 2021-12, it will grab 2021Q3 (7, 8, 9)
        prev_quarter_adv = prev_quarter_stats['ADV']
    except IndexError:
        prev_quarter_adv = np.NaN
    try:
        prev_year_stats = prod_yearly_df.loc[:tac_month].iloc[-2]   # On 2021-12, it will grab 2020
        prev_year_adv = prev_year_stats['ADV']
    except IndexError:
        prev_year_adv = np.NaN
    moq = (month_stats['ADV']-prev_quarter_adv)/prev_quarter_adv if prev_quarter_adv != 0 else np.NaN
    moy = (month_stats['ADV']-prev_year_adv)/prev_year_adv if prev_year_adv != 0 else np.NaN
    # Get ADV for current, incomplete month for reference
    try:
        next_month_stats = prod_monthly_df.loc[tac_next_incomplete_month].squeeze()
        next_month_adv = next_month_stats['ADV']
    except KeyError:
        # Example: 2022-07 AMB3 futures delisted
        print(f"WARNING: \"Next\" month {tac_next_incomplete_month} not found for {product_name}... delisted?\n"
              f"Product still included in report, but NaN for \"next\" month ADV!")
        next_month_adv = np.NaN
    last_12_rank = 13 - np.round(month_stats['Percentile (Last 12)'] * 12)
    tac_df.loc[product_name] = (month_stats['Sum'], month_stats['ADV'], last_12_rank,
                                prev_month_adv, month_stats['ADV Change'],
                                prev_quarter_adv, moq, prev_year_adv, moy,
                                next_month_adv)

# Export
tac_df.to_csv(DOWNLOADS_DIR / f'tac_df_{tac_month}.csv')

# Piece together supplement report for Scott Manziano - YTD, YoY, QoQ, high-level
tac2_df = pd.DataFrame(columns=['YTD Volume', 'YTD ADV',
                                'Prev YTD ADV', 'YTD over Prev YTD',
                                'Prev Year ADV', 'YTD over Prev Y', 'Prev Prev Year ADV', 'YTD over Prev Prev Y',
                                'QTD Volume', 'QTD ADV',
                                'Prev Quarter ADV', 'QTD over Prev Q'])
tac2_df.index.name = tac_month + ' Stats'
tac_month_ts = pd.to_datetime(tac_month)
year_start_ts = tac_month_ts.replace(month=1)
tac_month_prev_year_ts = tac_month_ts - pd.DateOffset(months=12)
prev_year_start_ts = tac_month_prev_year_ts.replace(month=1)
for product_name in ['VX_Monthly', 'VXM', 'IBHY', 'IBIG', 'VX_Weekly', 'AMW_Weekly', 'AMB1', 'AMB3', 'AMT1', 'AMT3']:
    prod_monthly_df = pd.DataFrame(PRODUCT_DICT[product_name]['Volume']['Monthly'])
    prod_quarterly_df = pd.DataFrame(PRODUCT_DICT[product_name]['Volume']['Quarterly'])
    prod_yearly_df = pd.DataFrame(PRODUCT_DICT[product_name]['Volume']['Yearly'])

    # Get current year (careful - up to TAC month only, no incomplete month), prev year, prev prev year
    ytd_df = prod_monthly_df.loc[year_start_ts:tac_month_ts]
    n_days_ytd = (ytd_df['Sum'] / ytd_df['ADV']).sum()
    ytd_volume = ytd_df['Sum'].sum()
    ytd_adv = ytd_volume / n_days_ytd if n_days_ytd != 0 else np.NaN
    # try:
    #     curr_year_stats = prod_yearly_df.loc[:tac_month].iloc[-1]   # On 2021-12, it will grab 2021
    #     ytd_volume, ytd_adv = curr_year_stats['Sum'], curr_year_stats['ADV']
    # except IndexError:
    #     ytd_volume = ytd_adv = np.NaN
    try:
        prev_year_stats = prod_yearly_df.loc[:tac_month].iloc[-2]  # On 2021-12, it will grab 2020
        prev_year_adv = prev_year_stats['ADV']
    except IndexError:
        prev_year_adv = np.NaN
    try:
        prevprev_year_stats = prod_yearly_df.loc[:tac_month].iloc[-3]  # On 2021-12, it will grab 2019
        prevprev_year_adv = prevprev_year_stats['ADV']
    except IndexError:
        prevprev_year_adv = np.NaN
    # Get YTD of prev year for seasonal comparison
    # NOTE: this implementation only works for whole months... could improve to be exact to date but shouldn't need
    prev_ytd_df = prod_monthly_df.loc[prev_year_start_ts:tac_month_prev_year_ts]
    n_days_prev_ytd = (prev_ytd_df['Sum']/prev_ytd_df['ADV']).sum()
    prev_ytd_adv = prev_ytd_df['Sum'].sum() / n_days_prev_ytd if n_days_prev_ytd != 0 else np.NaN
    # Calculate percents up or down
    ytd_o_prev_ytd = (ytd_adv - prev_ytd_adv) / prev_ytd_adv if prev_ytd_adv != 0 else np.NaN
    ytd_o_prev_y = (ytd_adv - prev_year_adv) / prev_year_adv if prev_year_adv != 0 else np.NaN
    ytd_o_prevprev_y = (ytd_adv - prevprev_year_adv) / prevprev_year_adv if prevprev_year_adv != 0 else np.NaN

    # Get current quarter, prev quarter
    try:
        curr_quarter_stats = prod_quarterly_df.loc[:tac_month].iloc[-1]     # On 2021-12, it will grab 2021Q4
        curr_quarter_volume, curr_quarter_adv = curr_quarter_stats['Sum'], curr_quarter_stats['ADV']
    except IndexError:
        curr_quarter_volume = curr_quarter_adv = np.NaN
    try:
        prev_quarter_stats = prod_quarterly_df.loc[:tac_month].iloc[-2]     # On 2021-12, it will grab 2021Q3 (7, 8, 9)
        prev_quarter_adv = prev_quarter_stats['ADV']
    except IndexError:
        prev_quarter_adv = np.NaN
    # Calculate percents up or down
    qtd_o_prev_q = (curr_quarter_adv - prev_quarter_adv) / prev_quarter_adv if prev_quarter_adv != 0 else np.NaN

    # Stitch
    tac2_df.loc[product_name] = (ytd_volume, ytd_adv,
                                 prev_ytd_adv, ytd_o_prev_ytd,
                                 prev_year_adv, ytd_o_prev_y, prevprev_year_adv, ytd_o_prevprev_y,
                                 curr_quarter_volume, curr_quarter_adv,
                                 prev_quarter_adv, qtd_o_prev_q)

# Export
tac2_df.to_csv(DOWNLOADS_DIR / f'tac2_df_{tac_month}.csv')

###############################################################################

# Generate per-product all-categories report

# [CONFIGURE] Manually set product and month of interest
allcat_month = USE_DATE[:-3]    # Default: '2022-03'

# Piece together report for each product
allcat_df_dict = {}
for allcat_product in ['VX', 'VX_Monthly', 'VX_Weekly', 'VXM',
                       'IBHY', 'IBIG',
                       'AMW_Weekly', 'AMB1', 'AMB3', 'AMT1', 'AMT3']:
    # Format product table
    allcat_df = \
        pd.DataFrame(columns=['Month Total', 'Avg Daily (ADV)', 'Last 12 Months Rank', 'Percentile (Since 2018-02)',
                              'Prev Month', 'MoM', 'Prev Quarter', 'MoQ', 'Prev Year', 'MoY',
                              '% Total', '% Total Last 12 Months Rank',
                              '% Total Percentile (Since 2018-02)', '% Total Change'])
    allcat_df.index.name = f"{allcat_product}: {allcat_month}"

    # Each field will be row in table
    for i_field in ['Volume', 'Notional', 'OI', 'Customer', 'TAS', 'Spreads', 'Block', 'ECRP', 'GTH']:
        prod_monthly_df = pd.DataFrame(PRODUCT_DICT[allcat_product][i_field]['Monthly'])
        prod_quarterly_df = pd.DataFrame(PRODUCT_DICT[allcat_product][i_field]['Quarterly'])
        prod_yearly_df = pd.DataFrame(PRODUCT_DICT[allcat_product][i_field]['Yearly'])
        # Get full stats for month of interest
        try:
            # Must .squeeze() because using approx date indexing
            month_stats = prod_monthly_df.loc[allcat_month].squeeze()
        except KeyError:
            print(f"WARNING, All-Categories Report: {allcat_product} {allcat_month} {i_field} not found")
            continue
        # Get ADVs for previous peridos
        try:
            prev_month_stats = prod_monthly_df.loc[:allcat_month].iloc[-2]  # On 2021-12, it will grab 2021-11
            prev_month_adv = prev_month_stats['ADV']
        except IndexError:
            prev_month_adv = np.NaN
        try:
            # On 2021-12, it will grab 2021Q3 (7, 8, 9)
            prev_quarter_stats = prod_quarterly_df.loc[:allcat_month].iloc[-2]
            prev_quarter_adv = prev_quarter_stats['ADV']
        except IndexError:
            prev_quarter_adv = np.NaN
        try:
            prev_year_stats = prod_yearly_df.loc[:allcat_month].iloc[-2]    # On 2021-12, it will grab 2020
            prev_year_adv = prev_year_stats['ADV']
        except IndexError:
            prev_year_adv = np.NaN
        # Set some shorthand names to fill in columns
        moq = (month_stats['ADV'] - prev_quarter_adv) / prev_quarter_adv if prev_quarter_adv != 0 else np.NaN
        moy = (month_stats['ADV'] - prev_year_adv) / prev_year_adv if prev_year_adv != 0 else np.NaN
        last_12_rank = 13 - np.round(month_stats['Percentile (Last 12)']*12)
        try:
            perc_total = month_stats['% Total']
            perc_total_last_12_rank = 13 - np.round(month_stats['% Total Percentile (Last 12)']*12)
            perc_total_percentile = month_stats['% Total Percentile (All)']
            perc_total_change = month_stats['% Total Change']
        except KeyError:
            perc_total, perc_total_last_12_rank, perc_total_percentile, perc_total_change = (np.NaN,)*4

        # Fill in row by column
        allcat_df.loc[i_field] = \
            (month_stats['Sum'], month_stats['ADV'], last_12_rank, month_stats['Percentile (All)'],
             prev_month_adv, month_stats['ADV Change'], prev_quarter_adv, moq, prev_year_adv, moy,
             perc_total, perc_total_last_12_rank, perc_total_percentile, perc_total_change)

    # Store completed product table
    allcat_df_dict[allcat_product] = allcat_df

# Export


def get_col_widths(data_df):
    # Get max width of index (1st column of Excel output)
    idx_max = max([len(str(s)) for s in data_df.index.values] + [len(str(data_df.index.name))])
    # Prepend this to the list of max widths of the rest of the columns
    rest_max_list = [max([len(str(s)) for s in data_df[col_name].values] + [len(col_name)])
                     for col_name in data_df.columns]
    return [idx_max] + rest_max_list


export_path_allcat = EXPORT_DIR_ALLCAT / USE_DATE
export_path_allcat.mkdir(parents=True, exist_ok=True)
print(f"{export_path_allcat} established as export path.")
for allcat_product in ['VX', 'VX_Monthly', 'VX_Weekly', 'VXM',
                       'IBHY', 'IBIG',
                       'AMW_Weekly', 'AMB1', 'AMB3', 'AMT1', 'AMT3']:
    allcat_df = allcat_df_dict[allcat_product]
    with pd.ExcelWriter(export_path_allcat / f'allcat_df_{allcat_product}_{allcat_month}.xlsx',
                        datetime_format='YYYY-MM-DD') as writer:
        # Create Excel object?
        allcat_df.to_excel(writer, sheet_name='All-Categories Table', freeze_panes=(1, 1))
        # Get xlsxwriter objects and create formats
        workbook = writer.book
        worksheet = writer.sheets['All-Categories Table']
        format_roundcomma = workbook.add_format({'num_format': '#,##0'})
        format_percentile = workbook.add_format({'num_format': '#0%'})
        format_perc = workbook.add_format({'num_format': '#0.0%'})
        format_percplusminus = workbook.add_format({'num_format': '+#0.0%;-#0.0%'})     # '+* #0.0%;-* #0.0%' spaced
        format_money = workbook.add_format({'num_format': '$#,##0_);($#,##0)'})
        # Set column formats and widths; TODO: 1) fix autofit and 2) add conditional formatting highlights and greys
        worksheet.set_column(1, 2, None, format_roundcomma)
        worksheet.set_column(4, 4, None, format_percentile)
        worksheet.set_column(5, 5, None, format_roundcomma)
        worksheet.set_column(6, 6, None, format_percplusminus)
        worksheet.set_column(7, 7, None, format_roundcomma)
        worksheet.set_column(8, 8, None, format_percplusminus)
        worksheet.set_column(9, 9, None, format_roundcomma)
        worksheet.set_column(10, 10, None, format_percplusminus)
        worksheet.set_column(11, 11, None, format_perc)
        worksheet.set_column(13, 13, None, format_percentile)
        worksheet.set_column(14, 14, None, format_percplusminus)
        # worksheet.set_row(2, None, format_money)    # Can't use because xlsxwriter gives absolute precedence to row
        # worksheet.conditional_format('B3', {'type': 'no_errors', 'format': format_money})   # Not very clean
        money_cells_list = [(1, 0), (1, 1), (1, 4), (1, 6), (1, 8)]     # Note it's pandas DF coordinates
        for row, col in money_cells_list:
            try:
                worksheet.write(row+1, col+1, allcat_df.iloc[row, col], format_money)   # +1 for index/header
            except TypeError:
                pass    # Just don't write NaN or INF
            except IndexError:
                break   # allcat_df is empty, i.e. product did not trade!
        # autofit_widths = get_col_widths(allcat_df)  # Won't work because the formatting changes the widths!
        # for i, width in enumerate(autofit_widths):
        #     worksheet.set_column(i, i, width)

###############################################################################

# Create mega data DF to use as Tableau data source
data_source_df_list = []    # Collect, then use pd.concat()
for product_name in PRODUCT_DICT.keys():
    for i_field in VOLUME_REFERENCE_COLS+VOLUME_SUBCAT_COLS:
        for i_agg in ['Daily', 'Monthly', 'Quarterly', 'Yearly']:
            agg_df = pd.DataFrame(PRODUCT_DICT[product_name][i_field][i_agg])
            agg_df['Product'] = product_name
            agg_df['Field'] = i_field
            agg_df['Aggregation'] = i_agg
            if i_field in VOLUME_SUBCAT_COLS:
                agg_df['Field Class'] = 'Volume Subcategory Field'
            else:
                agg_df['Field Class'] = 'Overview Field'
            data_source_df_list.append(agg_df)
data_source_df = pd.concat(data_source_df_list)
data_source_df.index.name = 'Date/Month/Quarter/Year'
data_source_df = \
    data_source_df.reset_index().set_index(['Product', 'Field Class', 'Field', 'Aggregation',
                                            'Date/Month/Quarter/Year'])

# Export
data_source_df.to_csv(DOWNLOADS_DIR / f'data_source_df_{USE_DATE}.csv')
