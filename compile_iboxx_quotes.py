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
             "WHERE fos.dt>='2021-08-01' "
             "AND fos.dt<='2022-03-02' "
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


mar_virtu_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/H2', 'KCLS')
mar_prime_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/H2', 'PRIM')
mar_dv_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/H2', 'DVTR')
mar_mercury_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/H2', 'MDTL')
# mar_tanius_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/H2', 'TANI')     # Tanius out

feb_virtu_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/G2', 'KCLS')
feb_prime_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/G2', 'PRIM')
feb_dv_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/G2', 'DVTR')
feb_mercury_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/G2', 'MDTL')
feb_tanius_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/G2', 'TANI')

jan_virtu_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/F2', 'KCLS')
jan_prime_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/F2', 'PRIM')
jan_dv_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/F2', 'DVTR')
jan_mercury_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/F2', 'MDTL')
jan_tanius_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/F2', 'TANI')

dec_virtu_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/Z1', 'KCLS')
dec_prime_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/Z1', 'PRIM')
dec_dv_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/Z1', 'DVTR')
dec_mercury_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/Z1', 'MDTL')
dec_tanius_daily_open_table = compile_daily_open(clean_quotes, 'IBHY/Z1', 'TANI')

###############################################################################
# Full intraday spot checks

dec_quotes = clean_quotes[clean_quotes['symbol_name'] == 'IBHY/Z1']     # Only simple, no roll
interesting_fields = ['firm', 'efid', 'account', 'order_entry_operator_id', 'cti_code', 'capacity',
                      'side', 'size', 'price', 'marketability_on_activation', 'traded_qty']

# Case 1: normal day, MM interactions
# Notes: DV kicks off with 70 cent market, Prime comes in at midpoint 16 cents wide,
#        Virtu and Tanius join, Mercury comes 20 mins later once everything is settled at 6 cents
#        DV had offers lifted 4 times, raised offer by 80 cents each time lol
intraday_check_1 = dec_quotes[dec_quotes['dt'] == '2021-11-16'].set_index('arrival_time')[interesting_fields]
intraday_check_1 = intraday_check_1.set_index('side', append=True).stack().unstack(1).unstack(1)
# intraday_check_1.to_csv(DOWNLOADS_DIR/'intraday_check_dec_2021-11-16.csv')

# Case 2: Virtu quoting alone
# Notes: maintains same width 40 cents all day
intraday_check_2 = dec_quotes[dec_quotes['dt'] == '2021-09-15'].set_index('arrival_time')[interesting_fields]
intraday_check_2 = intraday_check_2.set_index('side', append=True).stack().unstack(1).unstack(1)
# intraday_check_2.to_csv(DOWNLOADS_DIR/'intraday_check_dec_2021-09-15.csv')

# Case 3: normal day, MM interactions
# Notes: DV kicks off with 40 cent market, Prime and Tanius come in at midpoint 16 cents wide,
#        Mercury comes 6 mins in once everything is settled at 6 cents, Virtu joins 10 minutes in very tight
#        DV had bid lifted for 1 contract, dropped the bid by 70 cents lol
intraday_check_3 = dec_quotes[dec_quotes['dt'] == '2021-11-08'].set_index('arrival_time')[interesting_fields]
intraday_check_3 = intraday_check_3.set_index('side', append=True).stack().unstack(1).unstack(1)
# intraday_check_3.to_csv(DOWNLOADS_DIR/'intraday_check_dec_2021-11-08.csv')

# Case 4: right before maturity, MM interactions
# Notes: market 80 cents wide all day; long stretches of only bids from Prime, only asks from Virtu;
#        Prime had runs where they dropped their bid down to $80;
#        DV did not quote until 10:11am, and only a bid
intraday_check_4 = dec_quotes[dec_quotes['dt'] == '2021-11-30'].set_index('arrival_time')[interesting_fields]
intraday_check_4 = intraday_check_4.set_index('side', append=True).stack().unstack(1).unstack(1)
# intraday_check_4.to_csv(DOWNLOADS_DIR/'intraday_check_dec_2021-11-30.csv')

# Case 5: day where DV was not first to quote, MM interactions
# Notes: not a lot of insight - Prime comes in at 8:30am with 16 cent market, is fine
intraday_check_5 = dec_quotes[dec_quotes['dt'] == '2021-10-28'].set_index('arrival_time')[interesting_fields]
intraday_check_5 = intraday_check_5.set_index('side', append=True).stack().unstack(1).unstack(1)
# intraday_check_5.to_csv(DOWNLOADS_DIR/'intraday_check_dec_2021-10-28.csv')

# Case 6: chaos, MM interactions
# Notes: DV hit at open with IB customer trade, raised offer by a dollar, stayed wide as
#        IB customers kept buying through the morning
mar_quotes = clean_quotes[clean_quotes['symbol_name'] == 'IBHY/H2']
intraday_check_6 = mar_quotes[mar_quotes['dt'] == '2022-01-24'].set_index('arrival_time')[interesting_fields]
intraday_check_6 = intraday_check_6.set_index('side', append=True).stack().unstack(1).unstack(1)
# intraday_check_6.to_csv(DOWNLOADS_DIR/'intraday_check_mar_2022-01-24.csv')

# Case 7: 2021-09 contract in 2021-07 - what changed for DV?
# Notes: turns out Zeke Charlesworth may have transitioned trading to Walter this month
#        DVTR DVZC: pre-2021-06 to 2021-07-27; Zeke
#        DVTR DVTR: 2021-07-27 to present; Walter
#        CQCI DVTR: pre-2021-06 to 2021-10-18; this is the one-man show that couldn't cover costs ("DV2")
sep_quotes = clean_quotes[clean_quotes['symbol_name'] == 'IBHY/U1']
intraday_check_7 = sep_quotes[sep_quotes['dt'] == '2021-07-15'].set_index('arrival_time')[interesting_fields]
intraday_check_7 = intraday_check_7.set_index('side', append=True).stack().unstack(1).unstack(1)
# intraday_check_7.to_csv(DOWNLOADS_DIR/'intraday_check_sep_2021-07-15.csv')
