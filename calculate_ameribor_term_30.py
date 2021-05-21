import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET
from pathlib import Path

# Load DTCC commercial paper and commercial deposit data
DTCC_DATA_DIR = Path('C:/Users/gzhang/Downloads/Ameribor/For CBOE/')
DTCC_DATA_USECOLS = ['Principal Amount', 'Dated/ Issue Date', 'Settlement Date',
                     'Duration (Days)', 'Interest Rate Type', 'Country Code', 'Sector Code',
                     'Product Type', 'CUSIP', 'Issuer Name', 'Maturity Date', 'Settlement Amount',
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
             'Interest.Rate': 'Interest Rate', 'Interest.Rate.Type': 'Interest Rate Type', 'Sector.Code': 'Sector Code',
             'Duration..Days.': 'Duration (Days)', 'Days.To.Maturity': 'Days To Maturity',
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
# Filter 1: throw out busted trades, assert funded
afx_data = afx_data[afx_data['busted_at'].isna() & afx_data['funded_at'].notna()]
# Filter 2: overnight or 30-day lending markets only
afx_data = afx_data[(afx_data['instrument'] == 'overnight_unsecured_ameribor_loan')
                    | (afx_data['instrument'] == 'thirty_day_unsecured_ameribor_loan')]
# Create crucial columns matching DTCC format
afx_data.loc[(afx_data['instrument'] == 'overnight_unsecured_ameribor_loan'),
             'Days To Maturity'] = 1
afx_data.loc[(afx_data['instrument'] == 'thirty_day_unsecured_ameribor_loan'),
             'Days To Maturity'] = 30
afx_data.loc[(afx_data['instrument'] == 'overnight_unsecured_ameribor_loan'),
             'Duration (Days)'] = 1
afx_data.loc[(afx_data['instrument'] == 'thirty_day_unsecured_ameribor_loan'),
             'Duration (Days)'] = 30
afx_data['Principal Amount'] = afx_data['quantity'] * 1e6   # Originally in millions
afx_data['Interest Rate'] = afx_data['price'] / 1000 / 100  # Originally in thousands of basis points
# Filter 3: principal >= $1MM (shouldn't be needed for AFX but still)
afx_data = afx_data[afx_data['Principal Amount'] >= 1e6]

####

# Combine DTCC data with AFX data
# Crucial to both: ['Date', 'Product Type', 'Principal Amount', 'Duration (Days)', 'Days To Maturity', 'Interest Rate']
# DTCC filter process: ['Dated/ Issue Date', 'Settlement Date', 'Interest Rate Type', 'Country Code', 'Sector Code']
# DTCC nice to have: ['Issuer Name', 'CUSIP', 'Maturity Date', 'Settlement Amount']
# AFX filter process: ['instrument', 'busted_at', 'funded_at']
# AFX nice to have: ['borrower_name', 'lender_name', 'matched_at',
#                    'price', 'quantity', 'trade_id', 'borrower_id', 'lender_id']
dtcc_combine = pd.DataFrame()
dtcc_combine['Date'] = dtcc_data['Dated/ Issue Date']
dtcc_combine[['Product Type', 'Principal Amount', 'Duration (Days)', 'Days To Maturity', 'Interest Rate']] = \
    dtcc_data[['Product Type', 'Principal Amount', 'Duration (Days)', 'Days To Maturity', 'Interest Rate']]
dtcc_combine[['Issuer Name', 'CUSIP']] = dtcc_data[['Issuer Name', 'CUSIP']]
afx_combine = pd.DataFrame()
afx_combine['Date'] = afx_data['funded_at'].apply(lambda tstz: tstz.replace(tzinfo=None)).dt.normalize()
afx_combine[['Product Type', 'Principal Amount', 'Duration (Days)', 'Days To Maturity', 'Interest Rate']] = \
    afx_data[['instrument', 'Principal Amount', 'Duration (Days)', 'Days To Maturity', 'Interest Rate']]
afx_combine[['borrower_name', 'lender_name', 'matched_at', 'funded_at']] = \
    afx_data[['borrower_name', 'lender_name', 'matched_at', 'funded_at']]
combined_data = pd.concat([dtcc_combine, afx_combine], sort=False)  # Note: still no sorting

####

# Create "Dollar Basis Point Value"
combined_data['DBPV'] = combined_data['Principal Amount'] * combined_data['Days To Maturity']
# Sort for readability
combined_data = combined_data.sort_values(['Date', 'DBPV', 'Principal Amount'],
                                          ascending=[True, False, False]).reset_index(drop=True)

####

# Temp 2021 test
# test = combined_data.set_index('Date').loc['2021-01-04':'2021-03-19']
test = combined_data.set_index('Date').loc['2016-01-15':'2021-03-19']
test_dates = test.index.unique()
# test_daily_totals = test.groupby('Date')['DBPV'].sum()
# test['Daily Total DBPV'] = test_daily_totals
# test['Transaction DBPV Weight'] = test['DBPV'] / test['Daily Total DBPV']
# test['Transaction Weighted Interest Rate'] = test['Transaction DBPV Weight'] * test['Interest Rate']
VOLUME_THRESHOLD_NOT_METS = []
SAFEGUARD_NEEDEDS = []
TERM_30_RATES_FIRSTPASS = []
TERM_30_RATES = []
DAY_DF_DICT_FIRSTPASS = {}
DAY_DF_DICT = {}
for five_days_ago, today in zip(test_dates[:-4], test_dates[4:]):
    print(today.strftime('%Y-%m-%d'))
    test_curr_five_days = test.loc[five_days_ago:today].copy()
    curr_five_day_volume = test_curr_five_days['Principal Amount'].sum()
    if curr_five_day_volume < 25e9:
        VOLUME_THRESHOLD_NOT_METS.append((today, curr_five_day_volume))
        TERM_30_RATES.append((today, None))
        continue
    # Re-sort for 5-day window
    test_curr_five_days = test_curr_five_days.sort_values(['DBPV', 'Principal Amount'], ascending=[False, False])
    # Calculate total DBPV
    test_curr_five_days['5-Day Total DBPV'] = test_curr_five_days['DBPV'].sum()
    # Calculate each transaction's DBPV weight
    test_curr_five_days['Transaction DBPV Weight'] = \
        test_curr_five_days['DBPV'] / test_curr_five_days['5-Day Total DBPV']
    # Calculate each transaction's weighted interest rate
    test_curr_five_days['Transaction Weighted Interest Rate'] = \
        test_curr_five_days['Transaction DBPV Weight'] * test_curr_five_days['Interest Rate']
    # Calculate AMERIBOR Term-30 rate - FIRST PASS
    term_30_rate_firstpass = test_curr_five_days['Transaction Weighted Interest Rate'].sum()
    # Safeguard - check for transactions outside of 250bp range from first pass rate
    test_curr_five_days_safeguard = \
        test_curr_five_days[~((test_curr_five_days['Interest Rate'] > term_30_rate_firstpass + 2.5)
                              | (test_curr_five_days['Interest Rate'] < term_30_rate_firstpass - 2.5))].copy()
    # Calculate SECOND PASS
    test_curr_five_days_safeguard['5-Day Total DBPV'] = test_curr_five_days_safeguard['DBPV'].sum()
    test_curr_five_days_safeguard['Transaction DBPV Weight'] = \
        test_curr_five_days_safeguard['DBPV'] / test_curr_five_days_safeguard['5-Day Total DBPV']
    test_curr_five_days_safeguard['Transaction Weighted Interest Rate'] = \
        test_curr_five_days_safeguard['Transaction DBPV Weight'] * test_curr_five_days_safeguard['Interest Rate']
    term_30_rate = test_curr_five_days_safeguard['Transaction Weighted Interest Rate'].sum()
    if term_30_rate_firstpass != term_30_rate:
        SAFEGUARD_NEEDEDS.append((today, test_curr_five_days.shape[0]-test_curr_five_days_safeguard.shape[0],
                                  term_30_rate_firstpass, term_30_rate))
    TERM_30_RATES_FIRSTPASS.append((today, term_30_rate_firstpass))
    TERM_30_RATES.append((today, term_30_rate))
    DAY_DF_DICT_FIRSTPASS[today] = test_curr_five_days
    DAY_DF_DICT[today] = test_curr_five_days_safeguard
test_rates_firstpass = (pd.DataFrame(TERM_30_RATES_FIRSTPASS, columns=['Date', 'Term-30 Rate (First Pass)'])
                        .set_index('Date'))
test_rates = pd.DataFrame(TERM_30_RATES, columns=['Date', 'Term-30 Rate']).set_index('Date')
threshold_info = (pd.DataFrame(VOLUME_THRESHOLD_NOT_METS, columns=['Date', '5-Day Volume'])
                  .set_index('Date'))
safeguard_info = (pd.DataFrame(SAFEGUARD_NEEDEDS, columns=['Date', 'Transactions Omitted', 'First Pass Rate', 'Second Pass Rate'])
                  .set_index('Date'))
