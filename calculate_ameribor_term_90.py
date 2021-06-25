import pandas as pd
from pathlib import Path
import os
# from options_futures_expirations_v3 import DAY_OFFSET, ensure_bus_day
# from utility.universal_tools import chop_segments_off_string

# [MANUAL] Configure
OFFICIAL_START = pd.Timestamp('2016-06-01')
OUR_DATA_FORMAT_TRANSITION = pd.Timestamp('2021-03-19')     # Up to and including this date is dataset 1, then dataset 2
OUR_START = pd.Timestamp('2016-01-15')  # We have enough DTCC and AFX data to try calculating further than 2016-06-01
OUR_END = pd.Timestamp('2021-06-08')    # Set this to the latest date for which we have both DTCC and AFX data

####
# Load DTCC commercial paper and commercial deposit data

DTCC_DATA_USECOLS = ['Principal Amount', 'Dated/ Issue Date', 'Settlement Date',
                     'Duration (Days)', 'Interest Rate Type', 'Country Code', 'Sector Code',
                     'Product Type', 'CUSIP', 'Issuer Name',
                     'Maturity Date', 'Settlement Amount', 'Parties to Transaction Classification',
                     'Interest Rate', 'Days To Maturity']

# DTCC dataset 1 (2016 to 2021-03-23)
DTCC_DATA_DIR_1 = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/Ameribor/For CBOE/')
dtcc_data_list_1 = []
for year in range(2016, 2022):
    print(f"Loading DTCC dataset 1 {year}")
    if year == 2020:
        # Account for irregular column formatting
        year_dtcc_data_raw = pd.read_csv(DTCC_DATA_DIR_1 / f'{str(year)}DTCCfile.csv')
        year_dtcc_data_raw = year_dtcc_data_raw.rename(
            {'Product.Type': 'Product Type', 'Issuer.Name': 'Issuer Name', 'Settlement.Date': 'Settlement Date',
             'Dated..Issue.Date': 'Dated/ Issue Date', 'Maturity.Date': 'Maturity Date',
             'Principal.Amount': 'Principal Amount', 'Settlement.Amount': 'Settlement Amount',
             'Interest.Rate': 'Interest Rate', 'Interest.Rate.Type': 'Interest Rate Type',
             'Parties.to.Transaction.Classification': 'Parties to Transaction Classification',
             'Sector.Code': 'Sector Code', 'Duration..Days.': 'Duration (Days)', 'Days.To.Maturity': 'Days To Maturity',
             'Country.Code': 'Country Code'}, axis=1)
        year_dtcc_data_raw = year_dtcc_data_raw[DTCC_DATA_USECOLS]
        year_dtcc_data_raw = year_dtcc_data_raw.drop(year_dtcc_data_raw.index[-1])
        for date_col in ['Dated/ Issue Date', 'Settlement Date', 'Maturity Date']:
            year_dtcc_data_raw[date_col] = pd.to_datetime(year_dtcc_data_raw[date_col].astype(str), format='%Y%m%d')
    else:
        year_dtcc_data_raw = pd.read_csv(DTCC_DATA_DIR_1 / f'{str(year)}DTCCfile.csv',
                                         usecols=DTCC_DATA_USECOLS,
                                         parse_dates=['Dated/ Issue Date', 'Settlement Date', 'Maturity Date'])
    dtcc_data_list_1.append(year_dtcc_data_raw)
dtcc_data_1_raw = pd.concat(dtcc_data_list_1, sort=False)
dtcc_data_1_raw = dtcc_data_1_raw[DTCC_DATA_USECOLS]    # Enforce column order
dtcc_data_1 = dtcc_data_1_raw.copy()
# Filter 1: principal >= $1MM
dtcc_data_1 = dtcc_data_1[dtcc_data_1['Principal Amount'] >= 1e6]
# Filter 2: issue date == settle date
dtcc_data_1 = dtcc_data_1[dtcc_data_1['Dated/ Issue Date'] == dtcc_data_1['Settlement Date']]
# Filter 3: duration >= 2, <=40 days
dtcc_data_1 = dtcc_data_1[(dtcc_data_1['Duration (Days)'] >= 2) & (dtcc_data_1['Duration (Days)'] <= 40)]
# Filter 4: fixed interest rate delivery
dtcc_data_1 = dtcc_data_1[dtcc_data_1['Interest Rate Type'] == 'F']
# Filter 5: American company, financial sector
dtcc_data_1 = dtcc_data_1[(dtcc_data_1['Country Code'] == 'USA') & (dtcc_data_1['Sector Code'] == 'FIN')]

# DTCC dataset 2 (2021-02-16 to present)
DTCC_DATA_DIR_2 = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/Ameribor/AFX DTCC/')
dtcc_data_list_2 = []
dtcc_trading_days_2 = pd.to_datetime([f[1:7] for f in os.listdir(DTCC_DATA_DIR_2) if f.startswith('D')],
                                     yearfirst=True).sort_values()
for day in dtcc_trading_days_2:
    day_str = day.strftime('%y%m%d')
    print(f"Loading DTCC dataset 2 {day_str}")
    try:
        day_filename = sorted([f for f in os.listdir(DTCC_DATA_DIR_2) if f.startswith(f'D{day_str}')])[-1]
        if day_filename[-4:] != '.csv':
            os.rename(DTCC_DATA_DIR_2/day_filename, DTCC_DATA_DIR_2/(day_filename+'.csv'))
            day_filename += '.csv'
    except IndexError:
        raise ValueError(f"ERROR: IMPOSSIBLE. "
                         f"\"{day_str}\" DTCC dataset 2 file not read despite being found in directory.")
    # Load day's trades
    day_trades = pd.read_csv(DTCC_DATA_DIR_2 / day_filename,
                             skiprows=1, usecols=DTCC_DATA_USECOLS,
                             parse_dates=['Dated/ Issue Date', 'Settlement Date', 'Maturity Date'])
    dtcc_data_list_2.append(day_trades)
dtcc_data_2_raw = pd.concat(dtcc_data_list_2)
dtcc_data_2 = dtcc_data_2_raw.copy()
# Filter 1: principal >= $1MM
dtcc_data_2 = dtcc_data_2[dtcc_data_2['Principal Amount'] >= 1e6]
# Filter 2: issue date == settle date
dtcc_data_2 = dtcc_data_2[dtcc_data_2['Dated/ Issue Date'] == dtcc_data_2['Settlement Date']]
# Filter 3: duration >= 2, <=40 days
dtcc_data_2 = dtcc_data_2[(dtcc_data_2['Duration (Days)'] >= 2) & (dtcc_data_2['Duration (Days)'] <= 40)]
# Filter 4: fixed interest rate delivery
dtcc_data_2 = dtcc_data_2[dtcc_data_2['Interest Rate Type'] == 'F']
# Filter 5: American company, financial sector
dtcc_data_2 = dtcc_data_2[(dtcc_data_2['Country Code'] == 'USA') & (dtcc_data_2['Sector Code'] == 'FIN')]

####
# Re-format dataset columns for consistency
# DTCC data vs. AFX data
# Crucial to both: ['Date', 'Product Type', 'Principal Amount', 'Maturity Date',
#                   'Duration (Days)', 'Days To Maturity', 'Interest Rate']
# DTCC filter process: ['Dated/ Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code']
# DTCC nice to have: ['Dated/ Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code',
#                     'Issuer Name', 'CUSIP', 'Settlement Amount', 'Parties to Transaction Classification']
# AFX filter process: ['instrument', 'busted_at', 'funded_at']
# AFX nice to have: ['borrower_name', 'lender_name', 'matched_at',
#                    'price', 'quantity', 'trade_id', 'borrower_id', 'lender_id']

# DTCC dataset 1
dtcc_combine_1 = pd.DataFrame()
# Crucial
dtcc_combine_1['Date'] = dtcc_data_1['Dated/ Issue Date']
dtcc_combine_1[['Product Type', 'Principal Amount', 'Maturity Date',
                'Duration (Days)', 'Days To Maturity', 'Interest Rate']] = \
    dtcc_data_1[['Product Type', 'Principal Amount', 'Maturity Date',
                 'Duration (Days)', 'Days To Maturity', 'Interest Rate']]
# Nice (default ['Issuer Name', 'CUSIP'])
dtcc_combine_1[['Dated/Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code',
                'Issuer Name', 'CUSIP', 'Parties to Transaction Classification']] = \
    dtcc_data_1[['Dated/ Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code',
                 'Issuer Name', 'CUSIP', 'Parties to Transaction Classification']]
# Combine DTCC+AFX
combined_data_1 = dtcc_combine_1.copy()     # Note: still no sorting
combined_data_1 = combined_data_1[(combined_data_1['Date'] >= OUR_START)
                                  & (combined_data_1['Date'] <= OUR_DATA_FORMAT_TRANSITION)]

# DTCC dataset 2
dtcc_combine_2 = pd.DataFrame()
# Create combine version
dtcc_combine_2['Date'] = dtcc_data_2['Dated/ Issue Date']
dtcc_combine_2[['Product Type', 'Principal Amount', 'Maturity Date',
                'Duration (Days)', 'Days To Maturity', 'Interest Rate']] = \
    dtcc_data_2[['Product Type', 'Principal Amount', 'Maturity Date',
                 'Duration (Days)', 'Days To Maturity', 'Interest Rate']]
dtcc_combine_2[['Dated/Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code',
                'Issuer Name', 'CUSIP', 'Parties to Transaction Classification']] = \
    dtcc_data_2[['Dated/ Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code',
                 'Issuer Name', 'CUSIP', 'Parties to Transaction Classification']]
# Combine DTCC+AFX
combined_data_2 = dtcc_combine_2.copy()     # Note: still no sorting
combined_data_2 = combined_data_2[(combined_data_2['Date'] > OUR_DATA_FORMAT_TRANSITION)
                                  & (combined_data_2['Date'] <= OUR_END)]

# Combine datasets 1 and 2
combined_data = pd.concat([combined_data_1, combined_data_2], sort=False)

####
# Calculate derived field - DBPV

# Create "Dollar Basis Point Value"
combined_data['DBPV'] = combined_data['Principal Amount'] * combined_data['Days To Maturity']
# Sort for readability
combined_data = combined_data.sort_values(['Date', 'DBPV', 'Principal Amount'],
                                          ascending=[True, False, False]).reset_index(drop=True)

####
# Run the AMBOR30T calculation process

# Determine dataset
test = combined_data.set_index('Date').loc[OUR_START:OUR_END]   # '2016-01-15':'2021-03-19'
test_dates = test.index.unique()    # Essentially Federal Reserve K.8 business calendar


def filter_outliers_from_rate(data_df, filter_rate):
    # Let through 1) all AFX transactions
    #             2) DTCC transactions that are NOT outside 250bp bounds
    filtered_data_df = \
        data_df[~((data_df['Product Type'] == 'CP')
                  | (data_df['Product Type'] == 'CD'))
                | ~((data_df['Interest Rate'] > filter_rate + 2.5)
                    | (data_df['Interest Rate'] < filter_rate - 2.5))].copy()
    return filtered_data_df


# For each date:
#   1) Use previous day's term AMERIBOR rate (calculated from 5+ days lookback ending on previous date)
#      +-250bp to filter latest date's transactions.
#      Record eligible transactions into permanent dictionary for use during next 5+ days.
#   2) Apply minimum principal threshold ($25bn for term 30, $10bn for term 90) to determine number of
#      additional lookback days (if any needed beyond 5).
#   3) Calculate latest day's term AMERIBOR rate with the 5+ days lookback.
#   4) [OPTIONAL] Record "isolated" daily total DBPV and "isolated" daily weighted rate;
#      these deconstructed, intermediate numbers can be combined in sets of 5+ to re-construct actual term rate.
AMBOR30T_DICT = {}
ELIGIBLE_TRANSACTIONS_DF = pd.DataFrame()   # DataFrame for easy range lookup
OUTLIER_FILTER_IMPOSSIBLES = []
OUTLIER_FILTER_APPLIEDS = []
VOLUME_THRESHOLD_NOT_METS = []
AMBOR30T_IMPOSSIBLES = []
AMBOR30T_INPUT_DF_DICT = {}
ISOLATED_DAILY_TOTAL_VOLUME_DICT = {}
ISOLATED_DAILY_TOTAL_DBPV_DICT = {}
ISOLATED_DAILY_RATE_DICT = {}
for i, today in enumerate(test_dates):
    print(f"Calculating {today.strftime('%Y-%m-%d')}:")

    # Pull up latest date's transaction data
    today_data = test.loc[today].copy()
    print("\tData loaded")

    # 1) Look up previous AMBOR30T rate and filter latest date's transactions
    yesterday_i = i - 1
    if yesterday_i < 0 or test_dates[yesterday_i] not in AMBOR30T_DICT:
        # Edge case: no previous AMBOR30T rate - cannot apply filter
        OUTLIER_FILTER_IMPOSSIBLES.append(today)
        print("\tWARNING: Cannot apply outlier filter - no yesterday AMBOR30T")
    else:
        yesterday = test_dates[yesterday_i]
        yesterday_rate = AMBOR30T_DICT[yesterday]
        today_data_filtered = filter_outliers_from_rate(today_data, yesterday_rate)
        n_filtered = today_data.shape[0] - today_data_filtered.shape[0]
        if n_filtered != 0:
            OUTLIER_FILTER_APPLIEDS.append((today, n_filtered))
            print(f"\tOutlier filter effective! {n_filtered} transactions removed")
        today_data = today_data_filtered
    ELIGIBLE_TRANSACTIONS_DF = ELIGIBLE_TRANSACTIONS_DF.append(today_data)

    # 2) Apply $25bn (or $10bn) moveable left bound date
    left_bound_i = i - 4    # Initial planned lookback is 5 days
    if left_bound_i < 0:
        print("\tWARNING: Don't have 5 days of data yet... aborting")
        AMBOR30T_IMPOSSIBLES.append(today)
        continue    # Edge case at beginning of history: don't have 5 days of data yet
    left_bound_date = test_dates[left_bound_i]
    print(f"\tInitial lookback: {left_bound_date.strftime('%Y-%m-%d')}")
    # Extend if needed
    curr_bounded_data = ELIGIBLE_TRANSACTIONS_DF.loc[left_bound_date:today].copy()
    curr_bounded_volume = curr_bounded_data['Principal Amount'].sum()
    if curr_bounded_volume < 25e9:
        print("\tVolume threshold extension effective!")
        failed_five_day_volume = curr_bounded_volume
        n_days_extended = 0
        extend_success = False
        while curr_bounded_volume < 25e9:
            n_days_extended += 1
            new_left_bound_i = left_bound_i - n_days_extended
            if new_left_bound_i < 0:
                # Fail: ran out of dates to extend
                VOLUME_THRESHOLD_NOT_METS.append((today, failed_five_day_volume, None, None))
                print("\tWARNING: Principal volume could not be extended to $25bn... aborting")
                break
            new_left_bound_date = test_dates[new_left_bound_i]
            curr_bounded_data = ELIGIBLE_TRANSACTIONS_DF.loc[new_left_bound_date:today].copy()
            curr_bounded_volume = curr_bounded_data['Principal Amount'].sum()
        else:
            # Success: reaching here means volume was successfully extended
            extend_success = True
            VOLUME_THRESHOLD_NOT_METS.append((today, failed_five_day_volume, n_days_extended, curr_bounded_volume))
            print(f"\tInitial: ${failed_five_day_volume:,.0f}\n"
                  f"\tDays Extended: {n_days_extended}\n"
                  f"\tFinal: ${curr_bounded_volume:,.0f}")
        if not extend_success:
            AMBOR30T_IMPOSSIBLES.append(today)
            continue  # Just move on to next date, skipping calculation

    # 3) Calculate latest date's AMERIBOR term rate
    # Re-sort for 5+-day window
    curr_bounded_data = curr_bounded_data.sort_values(['DBPV', 'Principal Amount'], ascending=[False, False])
    # Calculate total DBPV
    curr_bounded_data['5+-Day Total DBPV'] = curr_bounded_data['DBPV'].sum()
    # Calculate each transaction's DBPV weight
    curr_bounded_data['Transaction DBPV Weight'] = \
        curr_bounded_data['DBPV'] / curr_bounded_data['5+-Day Total DBPV']
    # Calculate each transaction's weighted interest rate
    curr_bounded_data['Transaction Weighted Interest Rate'] = \
        curr_bounded_data['Transaction DBPV Weight'] * curr_bounded_data['Interest Rate']
    # Calculate AMERIBOR Term-30 rate
    ambor30t_rate = curr_bounded_data['Transaction Weighted Interest Rate'].sum()
    AMBOR30T_INPUT_DF_DICT[today] = curr_bounded_data
    AMBOR30T_DICT[today] = ambor30t_rate

    # 4) For deconstruction/alternate calculation: record "isolated" single-day numbers
    ISOLATED_DAILY_TOTAL_VOLUME_DICT[today] = today_data['Principal Amount'].sum() / 1e9    # Billions $
    today_data['Daily Total DBPV'] = today_data['DBPV'].sum()
    ISOLATED_DAILY_TOTAL_DBPV_DICT[today] = today_data['Daily Total DBPV'] / 360 / 10000  # Bespoke scaling format
    today_data['Daily Transaction DBPV Weight'] = today_data['DBPV'] / today_data['Daily Total DBPV']
    today_data['Daily Transaction Weighted Interest Rate'] = \
        today_data['Daily Transaction DBPV Weight'] * today_data['Interest Rate']
    isolated_daily_term_rate = today_data['Daily Transaction Weighted Interest Rate'].sum()
    ISOLATED_DAILY_RATE_DICT[today] = isolated_daily_term_rate

####
# Organize results into summary DataFrames

test_rates = pd.Series(AMBOR30T_DICT).sort_index()
test_rates.index.name, test_rates.name = 'Date', 'Replicated AMBOR30T'
threshold_info = (pd.DataFrame(VOLUME_THRESHOLD_NOT_METS, columns=['Date', 'Failed 5-Day Volume',
                                                                   'Days Extended', 'Extended Volume'])
                  .set_index('Date'))
outlier_info = (pd.DataFrame(OUTLIER_FILTER_APPLIEDS, columns=['Date', 'Transactions Omitted'])
                .set_index('Date'))
daily_total_dbpv = pd.Series(ISOLATED_DAILY_TOTAL_DBPV_DICT).sort_index()
daily_rate = pd.Series(ISOLATED_DAILY_RATE_DICT).sort_index()
daily_total_volume = pd.Series(ISOLATED_DAILY_TOTAL_VOLUME_DICT).sort_index()
daily_values_breakdown = pd.DataFrame({'BPV Weighted IR': daily_rate,
                                       'Underlying Volume': daily_total_volume,
                                       'Total BPV': daily_total_dbpv})
daily_values_breakdown.index.name = 'Date'

####
# Export results to disk

DOWNLOADS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/')  # For miscellaneous work

# Export rates and helper calculations
test_rates.to_csv(DOWNLOADS_DIR / 'ambor30t_test_rates.csv', header=True)
threshold_info.to_csv(DOWNLOADS_DIR / 'ambor30t_test_thresholds.csv')
outlier_info.to_csv(DOWNLOADS_DIR / 'ambor30t_test_outliers.csv')
daily_values_breakdown.to_csv(DOWNLOADS_DIR / 'ambor30t_test_daily_breakdown.csv')

# # Export input data for full history
# full_dates = ELIGIBLE_TRANSACTIONS_DF.loc[OFFICIAL_START:OUR_END].index.unique()
# for date in full_dates:
#     export_loc = (DOWNLOADS_DIR / 'My AMBOR30T Full History Input Data' /
#                   f'{date.strftime("%Y-%m-%d")}_dtccafx_ambor30t_input.csv')
#     AMBOR30T_INPUT_DF_DICT[date].to_csv(export_loc)

# # Export input data for 6+ months for CFTC legal filing
# CFTC_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/Ameribor/Term-30 Legal Filing/')
# cftc_export_dates = ELIGIBLE_TRANSACTIONS_DF.loc[OFFICIAL_START:OUR_END].index.unique()
# for date in cftc_export_dates:
#     export_loc = (CFTC_DIR / 'Input Data (Identities Scrubbed)' /
#                   f'{date.strftime("%Y-%m-%d")}_dtccafx_ambor30t_input.csv')
#     AMBOR30T_INPUT_DF_DICT[date].drop(['borrower_name', 'lender_name'], axis=1).to_csv(export_loc)

####

# Compare DTCC and AFX contributions for CFTC
cftc_export_transactions = ELIGIBLE_TRANSACTIONS_DF.loc['2020-09-01':'2021-03-19'].copy()
cftc_export_transactions.loc[(cftc_export_transactions['Product Type'] == 'CP')
                             | (cftc_export_transactions['Product Type'] == 'CD'),
                             'DTCC or AFX'] = 'DTCC'
cftc_export_transactions.loc[(cftc_export_transactions['Product Type'] == 'overnight_unsecured_ameribor_loan')
                             | (cftc_export_transactions['Product Type'] == 'thirty_day_unsecured_ameribor_loan')
                             | (cftc_export_transactions['Product Type'] == 'bilateral_loan_overnight')
                             | (cftc_export_transactions['Product Type'] == 'direct_settlement_loan_overnight'),
                             'DTCC or AFX'] = 'AFX'
dtcc_trans = cftc_export_transactions[cftc_export_transactions['DTCC or AFX'] == 'DTCC']
afx_trans = cftc_export_transactions[cftc_export_transactions['DTCC or AFX'] == 'AFX']

# Relative Share - DTCC vs. AFX
# Number of transactions
n_trans_dtcc = dtcc_trans.shape[0]
n_trans_afx = afx_trans.shape[0]
# Total principal amount (volume)
volume_trans_dtcc = dtcc_trans['Principal Amount'].sum()
volume_trans_afx = afx_trans['Principal Amount'].sum()
# Total weighted volume (DBPV)
dbpv_trans_dtcc = dtcc_trans['DBPV'].sum()
dbpv_trans_afx = afx_trans['DBPV'].sum()

# Distribution of Duration - DTCC vs. AFX
dur_n_dtcc = dtcc_trans.groupby('Duration (Days)')['Principal Amount'].count()
dur_n_afx = afx_trans.groupby('Duration (Days)')['Principal Amount'].count()
dur_volume_dtcc = dtcc_trans.groupby('Duration (Days)')['Principal Amount'].sum()
dur_volume_afx = afx_trans.groupby('Duration (Days)')['Principal Amount'].sum()
dur_dbpv_dtcc = dtcc_trans.groupby('Duration (Days)')['DBPV'].sum()
dur_dbpv_afx = afx_trans.groupby('Duration (Days)')['DBPV'].sum()
