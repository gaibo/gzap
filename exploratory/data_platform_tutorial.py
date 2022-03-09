from impala.dbapi import connect
from impala.util import as_pandas
import pandas as pd
from pathlib import Path

DOWNLOADS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/')

# Establish connection to Data Platform with Impala (distributed SQL database that runs on Hadoop cluster)
conn = connect(host='dpc.us.cboe.net', port=21050)

# Test query: BZX equities order_summary table
# NOTE: as of 2021-01-08, equities_us.order_summary does not have column "exchange";
#       this query no longer works
# TEST_SQL_QUERY = ("SELECT exchange, order_id, modify_sequence, side, price, size "
#                   "FROM equities_us.order_summary "
#                   "WHERE exchange='00' "
#                   "LIMIT 10")
TEST_SQL_QUERY = ("SELECT firm, order_id, modify_sequence, side, price, size "
                  "FROM equities_us.order_summary "
                  "WHERE firm='GSCO' "
                  "LIMIT 10")

###############################################################################
# Method 1: Cursor

# Get a connection cursor and use its execute() function
cursor = conn.cursor()
cursor.execute(TEST_SQL_QUERY)
print(f"cursor.fetchall():\n{cursor.fetchall()}")   # List
cursor.execute(TEST_SQL_QUERY)
print(f"cursor.fetchone():\n{cursor.fetchone()}")   # Iterable
cursor.execute(TEST_SQL_QUERY)
print("for-loop on cursor:")  # Can also just treat cursor as iterable
for row in cursor:
    print(row)

# Cursor can directly be converted into DataFrame
cursor.execute(TEST_SQL_QUERY)  # The .fetch methods on cursor expend it, so reload cursor
cursor_df = as_pandas(cursor)
print(f"as_pandas(cursor):\n{cursor_df}")

###############################################################################
# Method 2: pandas.read_sql()

# Directly pass connection object into pandas function
alt_df = pd.read_sql(con=conn, sql=TEST_SQL_QUERY)
print(f"pd.read_sql(conn, TEST_SQL_QUERY):\n{alt_df}")

###############################################################################
# Test: Get BP VIXTLT data

BPVIXTLT_SQL_QUERY = ("SELECT * "
                      "FROM idx_ref_data.index_values "
                      "WHERE index_symbol='VIXTLT'")
bp_vixtlt_df_raw = pd.read_sql(con=conn, sql=BPVIXTLT_SQL_QUERY)
bp_vixtlt = bp_vixtlt_df_raw.set_index(['dt', 'transact_time']).sort_index()['ls_value']
print(f"pd.read_sql(conn, BPVIXTLT_SQL_QUERY):\n{bp_vixtlt}")

###############################################################################
# Test: Get IBHY futures intraday quote data
# Questions: symbol_live vs. v_symbol?

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

clean_quotes = ibhy_quotes_df[ibhy_quotes_df['marketability_on_activation'] != ' '].sort_values('arrival_time')
feb_quotes = clean_quotes[clean_quotes['symbol_name'] == 'IBHY/G2']
prim_quotes = clean_quotes[clean_quotes['efid'] == 'PRIM']
feb_prim_quotes = feb_quotes[feb_quotes['efid'] == 'PRIM']
feb_prime_daily_open = \
    (feb_prim_quotes.groupby(['dt', 'side']).first()
     [['arrival_time', 'termination_time', 'marketability_on_activation', 'price', 'size', 'traded_qty']])
# feb_prime_daily_open.to_csv(DOWNLOADS_DIR/'feb_prime_daily_open.csv')
feb_daily_open = \
    (feb_quotes.groupby(['dt', 'efid', 'side']).first()
     [['arrival_time', 'termination_time', 'marketability_on_activation', 'price', 'size', 'traded_qty']])

unstacked_open_price = feb_prime_daily_open['price'].unstack()
unstacked_open_size = feb_prime_daily_open['size'].unstack()
feb_prim_daily_open_ba = unstacked_open_price.join(unstacked_open_size, rsuffix='_size')[['B_size', 'B', 'S', 'S_size']]
# feb_prim_daily_open_ba.to_csv(DOWNLOADS_DIR/'feb_prim_daily_open_ba.csv')

###############################################################################
# Cleanup

# Close the established connection
conn.close()    # Empirically we find this doesn't work lol - pd.read_sql() still works
