import pandas as pd
import numpy as np
from ishares_csv_reader import load_holdings_csv, BUSDAY_OFFSET, coupon_payment_from_holding, face_payment_from_holding, TREASURY_BUSDAY_OFFSET, get_historical_xls_info, to_per_million_shares
from bonds_analytics import create_coupon_schedule
import warnings
from pandas.errors import PerformanceWarning

# ####
#
# test_date = '2020-07-27'
#
# foo = get_cashflows_from_holdings('TLT', test_date)
# foo_shift = get_cashflows_from_holdings('TLT', test_date, shift_shares=True)
# bar = load_cashflows_csv('TLT', test_date)
# bar = bar[bar['CALL_TYPE'] == 'WORST'].set_index('CASHFLOW_DATE')[['INTEREST', 'PRINCIPAL', 'CASHFLOW']]
#
# print(foo - bar)
#
# ####

holdings, extra = load_holdings_csv('HYG', '2021-02-22')
etf_name = 'HYG'
asof_datelike = '2021-02-22'
file_dir = None
file_name = None
live_calc = False
shift_shares = False
verbose = True

def get_cashflows_from_holdings(etf_name='TLT', asof_datelike=None, file_dir=None, file_name=None,
                                live_calc=False, shift_shares=False, verbose=True):
    """ Create aggregated cash flows (ACF) information (in style of iShares ETF cash flows file)
        from local holdings information (Shares ETF holdings file)
        NOTE: the process in this function is highly visual - coupon_flow_df and face_flow_df
              may be useful for visualizing the cash flows contributions of individual notes/bonds
    :param etf_name: 'TLT', 'IEF', etc.
    :param asof_datelike: desired "as of" date of information; set None to get latest file
    :param file_dir: directory to search for data file (overrides default directory)
    :param file_name: exact file name to load from file_dir (overrides default file name)
    :param live_calc: set True if conversion is being performed during trade date just after "as of" date;
                      necessary to trigger alternative method for obtaining ex-dividend date distributions
    :param shift_shares: set True to perform idiosyncratic shift of shares outstanding to day before;
                         useful to account for iShares erroneous file format
    :param verbose: set True for explicit print statements
    :return:
    """
    # Load holdings (and section of additional info) from local CSV
    holdings, extra = load_holdings_csv(etf_name, asof_datelike, file_dir=file_dir, file_name=file_name, verbose=False)
    if holdings.empty:
        raise ValueError(f"ERROR: empty \"as of\" date: {asof_datelike}")
    asof_date = extra['Fund Holdings as of'].squeeze()  # Obtain pd.Timestamp this way, in case asof_datelike is None
    asof_date_str = asof_date.strftime('%Y-%m-%d')
    # Derive trade date and settle date
    trade_date = asof_date + BUSDAY_OFFSET
    settle_date = trade_date + 2*BUSDAY_OFFSET  # Formerly T+3 back in 2016ish
    if verbose:
        print(f"\"As of\" date: {asof_date_str}\n"
              f"Trade date: {trade_date.strftime('%Y-%m-%d')}\n"
              f"Settlement date: {settle_date.strftime('%Y-%m-%d')}")
    # Obtain shares outstanding
    if shift_shares:
        _, next_extra = load_holdings_csv(etf_name, trade_date, file_dir=file_dir, file_name=file_name, verbose=False)
        shares_outstanding = next_extra['Shares Outstanding'].squeeze()
        if verbose:
            print("Purposefully pulling shares outstanding from holdings CSV 1 day after \"as of\" date...")
    else:
        shares_outstanding = extra['Shares Outstanding'].squeeze()
    if verbose:
        print(f"Shares outstanding: {shares_outstanding}")

    # Focus only on Treasury notes/bonds, exclude cash-like assets
    notesbonds = holdings[holdings['Asset Class'] == 'Fixed Income'].reset_index(drop=True)
    # Map out all unique upcoming coupon dates
    # NOTE: coupon stops showing up when "as of" date reaches coupon arrival date, so want coupon dates after "as of"
    coupon_schedules = notesbonds['Maturity'].map(lambda m: list(create_coupon_schedule(m, asof_date)))
    unique_coupon_dates = sorted(set(coupon_schedules.sum()))
    # Map out all unique upcoming maturity dates
    unique_maturity_dates = sorted(set(notesbonds['Maturity']))

    # Initialize empty DataFrame with a column for each unique coupon date
    coupon_flow_df = pd.DataFrame(columns=unique_coupon_dates)
    # Initialize empty DataFrame with a column for each unique maturity date
    face_flow_df = pd.DataFrame(columns=unique_maturity_dates)
    # Fill each holding's cash flows into DataFrames according to schedule
    for i, holding in notesbonds.iterrows():
        # Note/bond's coupon amounts
        scaled_coupon_payment = coupon_payment_from_holding(holding, shares_outstanding)
        coupon_flow_df.loc[holding['ISIN'], coupon_schedules[i]] = scaled_coupon_payment    # To all coupon dates
        # Note/bond's maturity face amount
        scaled_face_payment = face_payment_from_holding(holding, shares_outstanding)
        face_flow_df.loc[holding['ISIN'], holding['Maturity']] = scaled_face_payment    # To maturity date

    # Compress interest (coupon) and principal (face) into a DataFrame
    interest_ser = coupon_flow_df.sum()
    principal_ser = face_flow_df.sum()
    cashflows_df = pd.DataFrame({'INTEREST': interest_ser,
                                 'PRINCIPAL': principal_ser}).replace(np.NaN, 0)

    # Change from raw maturity dates (15th) to cash flows dates (next business dates if 15th is not)
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=PerformanceWarning)     # Following operation is not vectorized
        cashflows_df.index = cashflows_df.index - TREASURY_BUSDAY_OFFSET + TREASURY_BUSDAY_OFFSET
    cashflows_df.index.name = 'CASHFLOW_DATE'

    # Calculate implied cash
    # NOTE: implied cash must be reduced on ex-dividend date by distribution amount
    bond_mv = notesbonds['Market Value'].sum()
    _, nav_per_share, _, _ = get_historical_xls_info('TLT', asof_date, verbose=False)
    if not live_calc:
        # If calculating historically, get dividends from iShares XLS which updates at end of each trade date
        try:
            _, _, div, _ = get_historical_xls_info('TLT', trade_date, verbose=False)    # Allowed to raise ValueError
        except ValueError as e:
            raise ValueError(f"No XLS historical data found for {asof_date_str}; "
                             f"set live_calc=True if day-of dividends are needed.\n"
                             f"{e}")
        if div != 0:
            nav_per_share -= div
            if verbose:
                print(f"Dividend: {div} found for ex-dividend date {asof_date_str}")
    else:
        # Pull latest dividends info from website sidebar (dividend will be available morning of ex-date)
        pass
    nav_mv = nav_per_share * shares_outstanding
    implied_cash = nav_mv - bond_mv
    implied_cash_scaled = to_per_million_shares(implied_cash, shares_outstanding)
    # Add into cash flows as a principal
    implied_cash_maturity_date = settle_date + BUSDAY_OFFSET
    if implied_cash_maturity_date in cashflows_df.index:
        cashflows_df.loc[implied_cash_maturity_date, 'PRINCIPAL'] += implied_cash_scaled
    else:
        cashflows_df.loc[implied_cash_maturity_date] = (0, implied_cash_scaled)
    cashflows_df = cashflows_df.sort_index()
    if verbose:
        print(f"Implied cash maturity date: {implied_cash_maturity_date.strftime('%Y-%m-%d')}\n"
              f"Implied cash per million shares: {implied_cash_scaled}")

    # Create sum of interest and principal column, for convenience like in iShares cash flow CSV
    cashflows_df['CASHFLOW'] = cashflows_df.sum(axis=1)
    return cashflows_df
