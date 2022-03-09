from impala.dbapi import connect
from impala.util import as_pandas
import pandas as pd
from pathlib import Path

DOWNLOADS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/')

# Establish connection to Data Platform with Impala (distributed SQL database that runs on Hadoop cluster)
conn = connect(host='dpc.us.cboe.net', port=21050)

###############################################################################
# Download IBHY futures intraday quote data from Data Platform's
# futures_us.order_summary, joining in some contract reference data

SQL_QUERY = ("SELECT fos.dt, fos.order_id, fos.arrival_time, fos.activation_time, fos.termination_time, "
             "       fos.firm, fos.account, fos.clearing_account, fos.efid, fos.cmta, "
             "       fos.order_entry_operator_id, fos.cti_code, "
             "       fos.symbol_id, fos.side, fos.size, fos.price, fos.simple_complex, fos.capacity, fos.open_close, "
             "       fos.traded_qty, fos.complex_contracts_traded, fos.marketability_on_activation, "
             "       symb.futures_root, symb.symbol_name, symb.expire_date , symb.begin_dt "
             "FROM futures_us.order_summary fos "
             "LEFT JOIN cfe_ref_data.v_symbol symb "
             "ON fos.symbol_id=symb.id "
             "WHERE fos.dt>='2021-09-01' "
             "AND fos.dt<='2022-02-01' "
             "AND symb.futures_root='IBHY'")
ibhy_quotes_df = pd.read_sql(con=conn, sql=SQL_QUERY)
print(f"pd.read_sql(conn, SQL_QUERY):\n{ibhy_quotes_df.head(10)}")
# ibhy_quotes_df.to_csv(DOWNLOADS_DIR/'ibhy_quotes_test_big.csv')

###############################################################################
# Clean and use quote data

# Remove pre-cancelled quotes and sort
clean_quotes = ibhy_quotes_df[ibhy_quotes_df['marketability_on_activation'] != ' '].sort_values('arrival_time')

# Define some test data
# Filtered quotes for just 2022-02 IBHY contract
feb_quotes = clean_quotes[clean_quotes['symbol_name'] == 'IBHY/G2']
kcls_quotes = clean_quotes[clean_quotes['efid'] == 'KCLS']
feb_kcls_quotes = feb_quotes[feb_quotes['efid'] == 'KCLS']
# Groupby, all EFIDs
feb_daily_open = \
    (feb_quotes.groupby(['dt', 'efid', 'side']).first()
     [['arrival_time', 'termination_time', 'marketability_on_activation', 'price', 'size', 'traded_qty']])
# Groupby, just Virtu
feb_virtu_daily_open = \
    (feb_kcls_quotes.groupby(['dt', 'side']).first()
     [['arrival_time', 'termination_time', 'marketability_on_activation', 'price', 'size', 'traded_qty']])
# feb_virtu_daily_open.to_csv(DOWNLOADS_DIR/'feb_virtu_daily_open.csv')


def compile_daily_open(quotes_clean, contract_symbol='IBHY/G2', efid='KCLS'):
    # Filter down quotes
    contract_quotes = quotes_clean[quotes_clean['symbol_name'] == contract_symbol]
    efid_contract_quotes = contract_quotes[contract_quotes['efid'] == efid]

    # Groupby to get opening quotes
    daily_open = \
        (efid_contract_quotes.groupby(['dt', 'side']).first()
         [['arrival_time', 'termination_time', 'marketability_on_activation', 'price', 'size', 'traded_qty']])

    # Unstack opening quotes for human-readable table
    unstacked_open_price = daily_open['price'].unstack()
    unstacked_open_size = daily_open['size'].unstack()
    unstacked_open_time = daily_open['arrival_time'].unstack()
    unstacked_open_marketability = daily_open['marketability_on_activation'].unstack()
    daily_open_table = \
        (unstacked_open_price.join(unstacked_open_size, rsuffix='_size')
                             .join(unstacked_open_time, rsuffix='_time')
                             .join(unstacked_open_marketability, rsuffix='_marketability')
         [['B_size', 'B_marketability', 'B_time', 'B', 'S', 'S_time', 'S_marketability', 'S_size']])
    return daily_open_table


feb_virtu_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/G2', 'KCLS')
feb_prime_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/G2', 'PRIM')
