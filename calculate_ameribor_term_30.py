import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET
from pathlib import Path

# Load DTCC commercial paper and commercial deposit data
DTCC_DATA_DIR = Path('C:/Users/gzhang/Downloads/Ameribor/For CBOE/')
DTCC_DATA_USECOLS = ['Principal Amount', 'Dated/ Issue Date', 'Settlement Date',
                     'Duration (Days)', 'Interest Rate Type', 'Country Code', 'Sector Code',
                     'Product Type', 'CUSIP', 'Issuer Name', 'Maturity Date', 'Settlement Amount',
                     'Interest Rate', 'Days To Maturity']
dtcc_data_raw = pd.read_csv(DTCC_DATA_DIR / '2021DTCCfile.csv',
                            usecols=DTCC_DATA_USECOLS,
                            parse_dates=['Dated/ Issue Date', 'Settlement Date', 'Maturity Date'])
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

# Combine DTCC data with AFX data
dtcc_combine = dtcc_data.copy()
combined_data = pd.concat([dtcc_data, afx_data], sort=False)


# Create "Dollar Basis Point Value"
combined_data['DBPV'] = combined_data['Principal Amount'] * combined_data['Days To Maturity']
# Sort for readability
combined_data = combined_data.sort_values(['Settlement Date', 'DBPV', 'Principal Amount'],
                                           ascending=[True, False, False]).reset_index(drop=True)
