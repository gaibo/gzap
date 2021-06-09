import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET, DAY_OFFSET, ensure_bus_day
from pathlib import Path

# Load DTCC commercial paper and commercial deposit data
DTCC_DATA_DIR = Path('C:/Users/gzhang/Downloads/Ameribor/For CBOE/')
DTCC_DATA_USECOLS = ['Principal Amount', 'Dated/ Issue Date', 'Settlement Date',
                     'Duration (Days)', 'Interest Rate Type', 'Country Code', 'Sector Code',
                     'Product Type', 'CUSIP', 'Issuer Name',
                     'Maturity Date', 'Settlement Amount', 'Parties to Transaction Classification',
                     'Interest Rate', 'Days To Maturity']
dtcc_data_list = []
for year in range(2016, 2022):
    if year == 2020:
        # Account for irregular column formatting
        year_dtcc_data_raw = pd.read_csv(DTCC_DATA_DIR / f'{str(year)}DTCCfile.csv')
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
        year_dtcc_data_raw = pd.read_csv(DTCC_DATA_DIR / f'{str(year)}DTCCfile.csv',
                                         usecols=DTCC_DATA_USECOLS,
                                         parse_dates=['Dated/ Issue Date', 'Settlement Date', 'Maturity Date'])
    dtcc_data_list.append(year_dtcc_data_raw)
dtcc_data_raw = pd.concat(dtcc_data_list, sort=False)
dtcc_data_raw = dtcc_data_raw[DTCC_DATA_USECOLS]    # Enforce column order
dtcc_data = dtcc_data_raw.copy()
# Filter 1: principal >= $1MM
dtcc_data = dtcc_data[dtcc_data['Principal Amount'] >= 1e6]
# Filter 2: issue date == settle date
dtcc_data = dtcc_data[dtcc_data['Dated/ Issue Date'] == dtcc_data['Settlement Date']]
# Filter 3: duration >= 2, <=40 days
dtcc_data = dtcc_data[(dtcc_data['Duration (Days)'] >= 2) & (dtcc_data['Duration (Days)'] <= 40)]
# Filter 4: fixed interest rate delivery
dtcc_data = dtcc_data[dtcc_data['Interest Rate Type'] == 'F']
# Filter 5: American company, Financial sector
dtcc_data = dtcc_data[(dtcc_data['Country Code'] == 'USA') & (dtcc_data['Sector Code'] == 'FIN')]

####

# Load AFX loan data
AFX_DATA_DIR = Path('P:/ProductDevelopment/Database/AFX/AFX_data/')
AFX_DATA_USECOLS = ['trade_id', 'matched_at', 'funded_at', 'busted_at',
                    'price', 'quantity',
                    'borrower_id', 'lender_id']
trading_days = pd.date_range('2016-01-15', '2021-03-19', freq=BUSDAY_OFFSET)
afx_data_list = []
MISSING_AFX_DAYS = []
for day in trading_days:
    day_str = day.strftime('%Y%m%d')
    print(day_str)
    # Load day's trades
    try:
        day_trades = pd.read_csv(AFX_DATA_DIR / day_str / 'trades.csv',
                                 usecols=AFX_DATA_USECOLS,
                                 parse_dates=['matched_at', 'funded_at', 'busted_at'])
    except FileNotFoundError:
        MISSING_AFX_DAYS.append(day)
        continue
    # Merge in institution reference info
    day_institutions = pd.read_csv(AFX_DATA_DIR / day_str / 'institutions.csv',
                                   usecols=['id', 'name'])
    day_afx_data = (day_trades.merge(day_institutions, how='left', left_on='borrower_id', right_on='id')
                    .drop('id', axis=1).rename({'name': 'borrower_name'}, axis=1))
    day_afx_data = (day_afx_data.merge(day_institutions, how='left', left_on='lender_id', right_on='id')
                    .drop('id', axis=1).rename({'name': 'lender_name'}, axis=1))
    afx_data_list.append(day_afx_data)
afx_data_raw = pd.concat(afx_data_list)
afx_data = afx_data_raw.copy()


# Create 'instrument' designation to separate overnight from 30-day, etc.
def apply_helper_instrument(s):
    s_list = s.split('_')
    if len(s_list) <= 4:
        # Cannot parse with universal rule
        return s
    else:
        # Concat important parts for identification
        return '_'.join(s_list[:-4])


afx_data['instrument'] = afx_data['trade_id'].apply(apply_helper_instrument)
# Filter 0: drop duplicates - 2 records of some trades
afx_data = afx_data.drop_duplicates('trade_id')
# Filter 1: throw out busted trades, assert funded
afx_data = afx_data[afx_data['busted_at'].isna() & afx_data['funded_at'].notna()]
# Filter 2: overnight or 30-day lending markets only
afx_data = afx_data[(afx_data['instrument'] == 'overnight_unsecured_ameribor_loan')
                    | (afx_data['instrument'] == 'thirty_day_unsecured_ameribor_loan')
                    | (afx_data['instrument'] == 'bilateral_loan_overnight')
                    | (afx_data['instrument'] == 'direct_settlement_loan_overnight')]
# Create crucial columns matching DTCC format
# NOTE: required logic for correct days to maturity: find next business day AFTER incrementing by 1 or 30
afx_data['Trade Date'] = afx_data['funded_at'].apply(lambda tstz: tstz.replace(tzinfo=None)).dt.normalize()
afx_data.loc[(afx_data['instrument'] == 'overnight_unsecured_ameribor_loan'), 'Maturity Date'] = \
    ensure_bus_day(afx_data.loc[(afx_data['instrument'] == 'overnight_unsecured_ameribor_loan'), 'Trade Date']
                   + DAY_OFFSET, shift_to='next', busday_type='SIFMA').values
afx_data.loc[(afx_data['instrument'] == 'thirty_day_unsecured_ameribor_loan'), 'Maturity Date'] = \
    ensure_bus_day(afx_data.loc[(afx_data['instrument'] == 'thirty_day_unsecured_ameribor_loan'), 'Trade Date']
                   + 30*DAY_OFFSET, shift_to='next', busday_type='SIFMA').values
afx_data['Days To Maturity'] = (afx_data['Maturity Date'] - afx_data['Trade Date']).dt.days
afx_data['Duration (Days)'] = afx_data['Days To Maturity']
afx_data['Principal Amount'] = afx_data['quantity'] * 1e6   # Originally in millions
afx_data['Interest Rate'] = afx_data['price'] / 1000 / 100  # Originally in thousands of basis points
# Filter 3: principal >= $1MM (shouldn't be needed for AFX but still)
afx_data = afx_data[afx_data['Principal Amount'] >= 1e6]

####

# Combine DTCC data with AFX data
# Crucial to both: ['Date', 'Product Type', 'Principal Amount', 'Maturity Date',
#                   'Duration (Days)', 'Days To Maturity', 'Interest Rate']
# DTCC filter process: ['Dated/ Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code']
# DTCC nice to have: ['Dated/ Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code',
#                     'Issuer Name', 'CUSIP', 'Settlement Amount', 'Parties to Transaction Classification']
# AFX filter process: ['instrument', 'busted_at', 'funded_at']
# AFX nice to have: ['borrower_name', 'lender_name', 'matched_at',
#                    'price', 'quantity', 'trade_id', 'borrower_id', 'lender_id']
# DTCC
dtcc_combine = pd.DataFrame()
# Crucial
dtcc_combine['Date'] = dtcc_data['Dated/ Issue Date']
dtcc_combine[['Product Type', 'Principal Amount', 'Maturity Date',
              'Duration (Days)', 'Days To Maturity', 'Interest Rate']] = \
    dtcc_data[['Product Type', 'Principal Amount', 'Maturity Date',
               'Duration (Days)', 'Days To Maturity', 'Interest Rate']]
# Nice (default ['Issuer Name', 'CUSIP'])
dtcc_combine[['Dated/Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code',
              'Issuer Name', 'CUSIP', 'Parties to Transaction Classification']] = \
    dtcc_data[['Dated/ Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code',
               'Issuer Name', 'CUSIP', 'Parties to Transaction Classification']]
# AFX
afx_combine = pd.DataFrame()
# Crucial
afx_combine['Date'] = afx_data['Trade Date']
afx_combine[['Product Type', 'Principal Amount', 'Maturity Date',
             'Duration (Days)', 'Days To Maturity', 'Interest Rate']] = \
    afx_data[['instrument', 'Principal Amount', 'Maturity Date',
              'Duration (Days)', 'Days To Maturity', 'Interest Rate']]
# Nice (default ['borrower_name', 'lender_name', 'matched_at', 'funded_at'])
afx_combine[['borrower_name', 'lender_name', 'matched_at', 'funded_at']] = \
    afx_data[['borrower_name', 'lender_name', 'matched_at', 'funded_at']]
# Combin DTCC+AFX
combined_data = pd.concat([dtcc_combine, afx_combine], sort=False)  # Note: still no sorting

####

# Create "Dollar Basis Point Value"
combined_data['DBPV'] = combined_data['Principal Amount'] * combined_data['Days To Maturity']
# Sort for readability
combined_data = combined_data.sort_values(['Date', 'DBPV', 'Principal Amount'],
                                          ascending=[True, False, False]).reset_index(drop=True)

####
# Run the AMBOR30T calculation process

# Determine dataset
test = combined_data.set_index('Date').loc['2016-01-15':'2021-03-19']
test_dates = test.index.unique()    # No idea how this compares to our trading calendars


def filter_outliers_from_rate(data_df, filter_rate):
    # Let through all AFX transactions OR DTCC transactions that are not outside 250bp bounds
    filtered_data_df = \
        data_df[~((data_df['Product Type'] == 'CP')
                  | (data_df['Product Type'] == 'CD'))
                | ~((data_df['Interest Rate'] > filter_rate + 2.5)
                    | (data_df['Interest Rate'] < filter_rate - 2.5))].copy()
    return filtered_data_df


# For each date:
#   1) Use previous AMBOR30T rate +-250bp to filter latest date's transactions
#   2) Record eligible transactions into permanent dictionary;
#      also record "isolated" daily total DBPV for easy lookback weighting purposes
#   3) Calculate "isolated" date rate;
#      apply $25bn minimum principal threshold to determine number of lookback days (if beyond 5) and
#      calculate final latest AMBOR30T rate with 5+ days lookback
AMBOR30T_DICT = {}
ELIGIBLE_TRANSACTIONS_DF = pd.DataFrame()   # DataFrame for easy range lookup
ISOLATED_DAILY_TOTAL_DBPV_DICT = {}
ISOLATED_DAILY_RATE_DICT = {}
ISOLATED_DAILY_TOTAL_VOLUME_DICT = {}
OUTLIER_FILTER_IMPOSSIBLES = []
OUTLIER_FILTER_APPLIEDS = []
VOLUME_THRESHOLD_NOT_METS = []
AMBOR30T_IMPOSSIBLES = []
AMBOR30T_INPUT_DF_DICT = {}
for i, today in enumerate(test_dates):
    print(today.strftime('%Y-%m-%d'))

    # Look up latest date's data
    today_data = test.loc[today].copy()
    print("\tData loaded")

    # Look up previous AMBOR30T rate and filter latest transactions
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
    ISOLATED_DAILY_TOTAL_DBPV_DICT[today] = today_data['DBPV'].sum()/360/10000
    ISOLATED_DAILY_TOTAL_VOLUME_DICT[today] = today_data['Principal Amount'].sum()

    # Apply $25bn moveable left bound date and calculate AMBOR30T rate
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

    # For alternate calculation method: record "isolated" single-day AMBOR30T for weighting
    today_data['Daily Total DBPV'] = today_data['DBPV'].sum()
    today_data['Daily Transaction DBPV Weight'] = today_data['DBPV'] / today_data['Daily Total DBPV']
    today_data['Daily Transaction Weighted Interest Rate'] = \
        today_data['Daily Transaction DBPV Weight'] * today_data['Interest Rate']
    isolated_daily_ambor30t_rate = today_data['Daily Transaction Weighted Interest Rate'].sum()
    ISOLATED_DAILY_RATE_DICT[today] = isolated_daily_ambor30t_rate

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

####

EXPORT_DIR = Path('C:/Users/gzhang/Downloads/Ameribor/Term-30 Legal Filing/')

# Export input data for 6 months
past_six_months_dates = ELIGIBLE_TRANSACTIONS_DF.loc['2020-09-01':'2021-03-19'].index.unique()
for date in past_six_months_dates:
    export_loc = EXPORT_DIR / '6 Months Input Data' / f'{date.strftime("%Y-%m-%d")}_dtccafx_ambor30t_input.csv'
    AMBOR30T_INPUT_DF_DICT[date].drop(['borrower_name', 'lender_name'], axis=1).to_csv(export_loc)

DOWNLOADS_DIR = Path('C:/Users/gzhang/Downloads/')

# Export input data for full history
full_dates = ELIGIBLE_TRANSACTIONS_DF.loc['2016-06-01':'2021-03-19'].index.unique()
for date in full_dates:
    export_loc = (DOWNLOADS_DIR / 'My AMBOR30T Full History Input Data' /
                  f'{date.strftime("%Y-%m-%d")}_dtccafx_ambor30t_input.csv')
    AMBOR30T_INPUT_DF_DICT[date].to_csv(export_loc)
