import pandas as pd
from futures_reader import create_bloomberg_connection
# from futures_reader import reformat_pdblp
from options_futures_expirations_v3 import next_expiry, third_friday, vix_thirty_days_before, BUSDAY_OFFSET
import matplotlib.pyplot as plt
plt.style.use('cboe-fivethirtyeight')

con = create_bloomberg_connection()
DOWNLOADS_DIR = 'C:/Users/gzhang/OneDrive - CBOE/Downloads/'

###############################################################################

START_DATE = pd.Timestamp('2000-01-01')
END_DATE = pd.Timestamp('2021-07-30')

# NOTE: 1709583D US Equity is old, matured VXX; VXX US Equity is "series B" and only goes back to 2018-01
VIX_ETPS = ['XIV US Equity', 'SVXY US Equity',
            '1709583D US Equity', 'VXX US Equity', 'VIXY US Equity',
            'UVXY US Equity', 'TVIXF US Equity',
            '00677U TT Equity', '1552 JP Equity',
            'PHDG US Equity', 'VQT US Equity', 'ZIVZF US Equity']

PROSHARES_DELEVERED_DATE = pd.Timestamp('2018-02-28')   # First date of v2
VIX_ETPS_LEVERAGE_v1 = {'XIV US Equity': -1, 'SVXY US Equity': -1,
                        '1709583D US Equity': 1, 'VXX US Equity': 1, 'VIXY US Equity': 1,
                        'UVXY US Equity': 2, 'TVIXF US Equity': 2,
                        '00677U TT Equity': 1, '1552 JP Equity': 1,
                        'PHDG US Equity': 1, 'VQT US Equity': 1, 'ZIVZF US Equity': -1}
VIX_ETPS_LEVERAGE_v2 = {'XIV US Equity': -1, 'SVXY US Equity': -0.5,
                        '1709583D US Equity': 1, 'VXX US Equity': 1, 'VIXY US Equity': 1,
                        'UVXY US Equity': 1.5, 'TVIXF US Equity': 2,
                        '00677U TT Equity': 1, '1552 JP Equity': 1,
                        'PHDG US Equity': 1, 'VQT US Equity': 1, 'ZIVZF US Equity': -1}

VIX_FUTURES = ['UX1 Index', 'UX2 Index', 'UX3 Index']

TEMP_TICKERS = ['SPVXSTR Index', 'SPVXSP Index']
TEMP_FIELDS = ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'EQY_SH_OUT', 'FUND_NET_ASSET_VAL', 'FUND_TOTAL_ASSETS']

###############################################################################

data_raw = con.bdh(VIX_ETPS + VIX_FUTURES + TEMP_TICKERS,
                   TEMP_FIELDS,
                   f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}",
                   elms=[('currency', 'USD')])
data_raw = data_raw[VIX_ETPS + VIX_FUTURES + TEMP_TICKERS].copy()    # Enforce column order

###############################################################################

# Generate VIX maturities list of arbitrary length (I should make a function)
vix_expiries_func = vix_thirty_days_before(third_friday)
vix_futures_mat_list = [next_expiry(START_DATE, vix_expiries_func, n_terms=i)
                        for i in range(1, 301)]

# Create a VIX maturities reference DataFrame independent of data_raw
vix_mat_df = pd.DataFrame({'VIX Maturity': vix_futures_mat_list})
vix_mat_df['Prev VIX Maturity'] = vix_mat_df['VIX Maturity'].shift(1)


def busday_between(earlier, later):
    return len(pd.date_range(earlier, later, freq=BUSDAY_OFFSET, closed='left'))


vix_mat_df['Bus Days Since Last Maturity'] = \
    vix_mat_df.loc[1:].apply(lambda row: busday_between(row['Prev VIX Maturity'], row['VIX Maturity']), axis=1)
vix_mat_df = vix_mat_df.dropna().reset_index(drop=True)

# Literally go through each day in data_raw and calculate 1) weighted VIX futures price and 2) the vega per share
vix_etps = data_raw[VIX_ETPS].copy()
vix_futures = data_raw[VIX_FUTURES].copy()
for day in data_raw.index:
    # searchsorted() is always convoluted - day match case with side='right' means return next idx, not matching idx;
    # i.e., on day of maturity, we have already moved onto next "frame" and can consider new frame's number of days.
    # The goal is 100% in "next" term on DAY BEFORE maturity day of maturing term to avoid the settlement.
    # Note then that on maturity day of maturing term, we are already split between ~96% next term and ~4% third term.
    # Consider additionally that price source is Bloomberg's generic xth future (UX1, UX2, etc.), which rolls UX2
    # to UX1 on DAY AFTER UX1's maturity. So on day of UX1's maturity, we must split between ~96% UX2 and ~4% UX3.
    # On DAY BEFORE maturity date,
    #   - UX1 is still on second to last day of the almost obsolete future
    #   - put 0% weight in UX1, 100% weight in UX2
    # On maturity date,
    #   - UX1 is still on last day of the almost obsolete future
    #   - we have moved the "frame" on, despite UX1 and UX2 not transitioning yet
    #   - put 100% - 100%/n_busdays_in_new_frame*(busdays_since_maturity+1) in UX2 (where busdays_since_maturity is 0),
    #     complement in UX3
    # On DAY AFTER maturity date,
    #   - UX2 has become new UX1, UX3 has become new UX2
    #   - put 100% - 100%/n_busdays_in_new_frame*(busdays_since_maturity+1) in UX1 (where busdays_since_maturity is 1),
    #     complement in UX2
    day_vix_context_idx = vix_mat_df['VIX Maturity'].searchsorted(day, side='right')
    day_next_mat, day_prev_mat, n_busdays = vix_mat_df.loc[day_vix_context_idx]
    nth_day_of_frame = busday_between(day_prev_mat, day) + 1
    first_term_weight = 1 - 1/n_busdays*nth_day_of_frame    # = (busday_between(day, day_next_mat) - 1) / n_busdays
    second_term_weight = 1 - first_term_weight
    if nth_day_of_frame == 1:
        # Today is the maturity date - we have to reach into 3rd term because of UX1 transition delay!
        first_term_origin = 'UX2 Index'
        second_term_origin = 'UX3 Index'
    else:
        first_term_origin = 'UX1 Index'
        second_term_origin = 'UX2 Index'
    first_term_vix_future = vix_futures.loc[day, (first_term_origin, 'PX_LAST')]
    second_term_vix_future = vix_futures.loc[day, (second_term_origin, 'PX_LAST')]
    weighted_vix_future = (first_term_weight*first_term_vix_future
                           + second_term_weight*second_term_vix_future)
    # Record every step for posterity
    vix_futures.loc[day, 'Previous Maturity'] = day_prev_mat
    vix_futures.loc[day, 'Next Maturity'] = day_next_mat
    vix_futures.loc[day, 'Days in Current Maturity'] = n_busdays
    vix_futures.loc[day, 'nth Day in Current Maturity'] = nth_day_of_frame
    vix_futures.loc[day, 'First Term Weight'] = first_term_weight
    vix_futures.loc[day, 'Second Term Weight'] = second_term_weight
    vix_futures.loc[day, 'First Term VIX Futures Price'] = first_term_vix_future
    vix_futures.loc[day, 'First Term BBG Origin'] = first_term_origin
    vix_futures.loc[day, 'Second Term VIX Futures Price'] = second_term_vix_future
    vix_futures.loc[day, 'Second Term BBG Origin'] = second_term_origin
    vix_futures.loc[day, 'Weighted VIX Futures Price'] = weighted_vix_future
    # Calculate vega per share with the weighted VIX future
    for etp in VIX_ETPS:
        leverage = VIX_ETPS_LEVERAGE_v1[etp] if day < PROSHARES_DELEVERED_DATE else VIX_ETPS_LEVERAGE_v2[etp]
        vega_per_share = vix_etps.loc[day, (etp, 'PX_LAST')] * leverage / weighted_vix_future
        vix_equiv_volume = vega_per_share * vix_etps.loc[day, (etp, 'PX_VOLUME')] / 1000
        # Record for posterity
        vix_etps.loc[day, (etp, 'Leverage')] = leverage
        vix_etps.loc[day, (etp, 'Weighted VIX Futures Price Used')] = weighted_vix_future
        vix_etps.loc[day, (etp, 'Vega per Share')] = vega_per_share
        vix_etps.loc[day, (etp, 'VIX Futures Equivalent Volume')] = vix_equiv_volume

# Clean up for export
# NOTE: for some reason, VIX futures list 0 volume for MLK 2021 (2021-01-18) instead of NaN; may need to manually clean
pretty_vix_futures = vix_futures.dropna(how='all')
pretty_vix_etps = vix_etps.dropna(how='all')
etp_vix_equiv_volume_df = vix_etps.xs('VIX Futures Equivalent Volume', axis=1, level='field').dropna(how='all')
mega_vix_futures_equiv_df = \
    etp_vix_equiv_volume_df.join(vix_futures[VIX_FUTURES].xs('PX_VOLUME', axis=1, level='field').dropna(how='all'),
                                 how='outer')
export_df = mega_vix_futures_equiv_df.sort_index(ascending=False)
export_df.index.name = 'Date'
export_df['Total VIX ETP Volume (VIX Equivalent)'] = \
    export_df[['XIV US Equity', 'SVXY US Equity',
               '1709583D US Equity', 'VXX US Equity', 'VIXY US Equity',
               'UVXY US Equity', 'TVIXF US Equity',
               '00677U TT Equity', '1552 JP Equity',
               'PHDG US Equity', 'VQT US Equity', 'ZIVZF US Equity']].dropna(how='all').abs().sum(axis=1)
export_df['Total VIX Futures Volume'] = \
    export_df[['UX1 Index', 'UX2 Index', 'UX3 Index']].dropna(how='all').abs().sum(axis=1)
export_df = export_df[['Total VIX ETP Volume (VIX Equivalent)', 'Total VIX Futures Volume']
                      + list(mega_vix_futures_equiv_df.columns)]  # Reorder columns
export_df.to_csv(DOWNLOADS_DIR+f"vix_etps_equiv_volume_{END_DATE.strftime('%Y-%m-%d')}.csv")
