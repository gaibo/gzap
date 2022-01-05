import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET
from ishares_csv_reader import get_cashflows_from_holdings
from bonds_analytics import get_day_difference_30_360, get_yield_to_maturity, get_duration
from futures_reader import create_bloomberg_connection

# [CONFIGURE]
ETF_NAME = 'TLT'
LIVE_CALC = False  # Set False for historical calc (allows us to get dividend info)
START_DATE = pd.Timestamp('2010-01-05')
END_DATE = pd.Timestamp('now').normalize() - BUSDAY_OFFSET  # Default to using latest date
# =============================================================================
# END_DATE = pd.Timestamp('2020-08-26')   # Manual
# =============================================================================

###############################################################################

# Pull historical ETF closes from Bloomberg
con = create_bloomberg_connection()
BBG_PRICE_HIST = con.bdh(f'{ETF_NAME} US Equity', 'PX_LAST',
                         START_DATE.strftime('%Y%m%d'), END_DATE.strftime('%Y%m%d')).squeeze()
con.stop()  # End the connection

###############################################################################

# Set trade date to change with for-loop
TRADING_DAYS = pd.date_range(start=START_DATE, end=END_DATE, freq=BUSDAY_OFFSET)
TRADE_DATE_LIST = []
ASOF_DATE_LIST = []
SETTLE_DATE_LIST = []
IMPLIED_CASH_DATE_LIST = []
YIELD_LIST = []
MOD_DUR_LIST = []
for TRADE_DATE in TRADING_DAYS:
    # On morning of trade date, CSV files are updated to "as of" previous trade date
    ASOF_DATE = TRADE_DATE - BUSDAY_OFFSET
    SETTLE_DATE = TRADE_DATE + 2*BUSDAY_OFFSET  # Formerly T+3 back in 2016ish
    # Important: "implied cash date" is earliest day for cash arrival, and that includes bond coupon payments!
    IMPLIED_CASH_MATURITY_DATE = SETTLE_DATE + BUSDAY_OFFSET

    # Produce cash flows from holdings
    try:
        # NOTE: if performed just after ex-dividend date (start of month), check if you missed the latest dividend!
        cashflows_inhouse = get_cashflows_from_holdings(ETF_NAME, ASOF_DATE, live_calc=LIVE_CALC, shift_shares=True)
    except FileNotFoundError:
        continue
    except ValueError as e:
        print(e)
        continue
    except AttributeError as e:
        print(e)
        continue
    except TypeError as e:
        print(e)
        continue

    # Calculate number of semiannual periods forward from settlement date to each maturity (30/360)
    to_maturity_cf = cashflows_inhouse['CASHFLOW'].copy()
    settle_mat_periods_30_360 = to_maturity_cf.index.map(lambda m: get_day_difference_30_360(SETTLE_DATE, m)) / 360 * 2
    periods_forward = pd.Series(settle_mat_periods_30_360, index=to_maturity_cf.index, name='SEMIANNUAL PERIODS')

    # Back out yield from ETF price
    etf_price = BBG_PRICE_HIST[TRADE_DATE]
    etf_yield = get_yield_to_maturity(etf_price * 1_000_000, coupon=None,
                                      remaining_coupon_periods=periods_forward, remaining_payments=to_maturity_cf)
    # Calculate modified duration from yield
    etf_mod_dur = get_duration(etf_yield, coupon=None,
                               remaining_coupon_periods=periods_forward, remaining_payments=to_maturity_cf,
                               get_modified=True)

    # Record
    TRADE_DATE_LIST.append(TRADE_DATE)
    ASOF_DATE_LIST.append(ASOF_DATE)
    SETTLE_DATE_LIST.append(SETTLE_DATE)
    IMPLIED_CASH_DATE_LIST.append(IMPLIED_CASH_MATURITY_DATE)
    YIELD_LIST.append(etf_yield)
    MOD_DUR_LIST.append(etf_mod_dur)

# Format MD results
md_hist = pd.DataFrame({'Trade Date': TRADE_DATE_LIST,
                        'As of Date': ASOF_DATE_LIST,
                        'Settlement Date': SETTLE_DATE_LIST,
                        'Implied Cash Maturity Date': IMPLIED_CASH_DATE_LIST,
                        'Yield to Maturity': YIELD_LIST,
                        'Modified Duration': MOD_DUR_LIST}).set_index('Trade Date').sort_index()

###############################################################################

# Validate against past results
EXPORT_DIR = 'P:/PrdDevSharedDB/New Treasury VIX/Historical TLT Modified Durations/'
# DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
import os
latest_md_hist_file = sorted([f for f in os.listdir(EXPORT_DIR) if f.startswith('md_hist_')])[-1]
md_hist_old = pd.read_csv(EXPORT_DIR + latest_md_hist_file, index_col='Trade Date',
                          parse_dates=['Trade Date', 'As of Date', 'Settlement Date', 'Implied Cash Maturity Date'])
shared_idx = md_hist.index.intersection(md_hist_old.index)
# validate_diff = (md_hist.loc[md_hist_old.index] - md_hist_old).sum()    # More strict
validate_diff = (md_hist.loc[shared_idx] - md_hist_old.loc[shared_idx]).abs().sum()     # Less strict
# Literally graph (md_hist.loc[md_hist_old.index] - md_hist_old)['Yield to Maturity'] to check for problem dates
assert validate_diff['Yield to Maturity'] < 1e-13
assert validate_diff['Modified Duration'] < 1e-13

# Export
md_hist.to_csv(EXPORT_DIR + f"md_hist_{END_DATE.strftime('%Y-%m-%d')}.csv")
