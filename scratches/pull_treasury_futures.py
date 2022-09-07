import pandas as pd
from futures_reader import create_bloomberg_connection
# from futures_reader import reformat_pdblp
# from options_futures_expirations_v3 import next_expiry, third_friday, vix_thirty_days_before
# from options_futures_expirations_v3 import BUSDAY_OFFSET
from options_futures_expirations_v3 import next_treasury_futures_maturity
import matplotlib.pyplot as plt
plt.style.use('cboe-fivethirtyeight')

con = create_bloomberg_connection()
DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

###############################################################################

START_DATE = pd.Timestamp('2010-01-01')
END_DATE = pd.Timestamp('2020-12-28')

TREASURY_BASE_TICKERS = ['TU', 'FV', 'TY', 'US']

# TEMP_TICKERS = ['TU Index', 'SPVXSP Index']
TEMP_TICKERS = [base + str(number) + ' Comdty' for base in TREASURY_BASE_TICKERS for number in range(1, 4)]

# TEMP_FIELDS = ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'EQY_SH_OUT', 'FUND_NET_ASSET_VAL', 'FUND_TOTAL_ASSETS']
TEMP_FIELDS = ['PX_LAST', 'PX_VOLUME']

###############################################################################

data_raw = con.bdh(TEMP_TICKERS,
                   TEMP_FIELDS,
                   f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}")    # elms=[('currency','USD')]
data_raw = data_raw[TEMP_TICKERS].copy()    # Enforce column order

###############################################################################

# Generate Treasury futures maturities lists of arbitrary length (I should make a function)
tu_futures_mat_list = [next_treasury_futures_maturity(START_DATE-pd.Timedelta(days=30), n_terms=i, tenor=2)
                       for i in range(1, 101)]
fv_futures_mat_list = [next_treasury_futures_maturity(START_DATE-pd.Timedelta(days=30), n_terms=i, tenor=5)
                       for i in range(1, 101)]
ty_futures_mat_list = [next_treasury_futures_maturity(START_DATE-pd.Timedelta(days=30), n_terms=i, tenor=10)
                       for i in range(1, 101)]
us_futures_mat_list = [next_treasury_futures_maturity(START_DATE-pd.Timedelta(days=30), n_terms=i, tenor=30)
                       for i in range(1, 101)]
futures_mat_df = pd.DataFrame({'TU Maturities': tu_futures_mat_list, 'FV Maturities': fv_futures_mat_list,
                               'TY Maturities': ty_futures_mat_list, 'US Maturities': us_futures_mat_list})
normalized_futures_mat_df = futures_mat_df.apply(lambda col: col.dt.normalize())
