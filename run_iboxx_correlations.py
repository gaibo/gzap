import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET, generate_expiries
from futures_reader import create_bloomberg_connection, stitch_bloomberg_futures
import matplotlib.pyplot as plt
from collections.abc import Iterable
from mpl_tools import save_fig
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use('cboe-fivethirtyeight')

DATA_DIR = 'C:/Users/gzhang/Downloads/iBoxx Bloomberg Pulls/'   # Local data save
DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
START_DATE = pd.Timestamp('2000-01-01')
END_DATE = pd.Timestamp('now').normalize() - BUSDAY_OFFSET  # Default yesterday
ROLL_N_BEFORE_EXPIRY = 3    # Default 3
con = create_bloomberg_connection()


###############################################################################
# Pull from Bloomberg

# High Yield: IBY1 futures, IBXXIBHY, HYG, IBOXHY, CWY1 futures
# NOTE: futures daily percent return requires stitching together near and next terms
# NOTE: HYG Total Return (field 'TOT_RETURN_INDEX_GROSS_DVDS') is needed for correlations
# NOTE: for some reason, Bloomberg provides aggregate futures info through CWY1, but not literal price
# NOTE: for CDX index Total Return, need my fancy scaling stitching code
iby1 = con.bdh('IBY1 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
iby1.to_csv(DATA_DIR + f"IBY1_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
iby2 = con.bdh('IBY2 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
iby2.to_csv(DATA_DIR + f"IBY2_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
ibxxibhy = con.bdh('IBXXIBHY Index', ['PX_LAST'],
                   f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ibxxibhy.to_csv(DATA_DIR + f"IBXXIBHY_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
hyg = con.bdh('HYG US Equity', ['PX_LAST', 'PX_VOLUME', 'TOT_RETURN_INDEX_GROSS_DVDS'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
hyg.to_csv(DATA_DIR + f"HYG_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
iboxhy = con.bdh('IBOXHY Index', ['PX_LAST'],
                 f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
iboxhy.to_csv(DATA_DIR + f"IBOXHY_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
cwy1 = con.bdh('CWY1 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
cwy1.to_csv(DATA_DIR + f"CWY1_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
scaled_cds = pd.read_csv('P:/PrdDevSharedDB/BBG Pull Scripts/scaled_cds_indexes.csv',
                         index_col='date', parse_dates=True)
scaled_cdx_na_hy = scaled_cds['CDX NA HY'].dropna()

# Investment Grade: IHB1 futures, IBXXIBIG, LQD, IBOXIG, CWI1 futures
# NOTE: CWI futures from Bloomberg are unusable - prices are broken
ihb1 = con.bdh('IHB1 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ihb1.to_csv(DATA_DIR + f"IHB1_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
ihb2 = con.bdh('IHB2 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ihb2.to_csv(DATA_DIR + f"IHB2_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
ibxxibig = con.bdh('IBXXIBIG Index', ['PX_LAST'],
                   f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ibxxibig.to_csv(DATA_DIR + f"IBXXIBIG_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
lqd = con.bdh('LQD US Equity', ['PX_LAST', 'PX_VOLUME', 'TOT_RETURN_INDEX_GROSS_DVDS'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
lqd.to_csv(DATA_DIR + f"LQD_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
iboxig = con.bdh('IBOXIG Index', ['PX_LAST'],
                 f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
iboxig.to_csv(DATA_DIR + f"IBOXIG_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
cwi1 = con.bdh('CWI1 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
cwi1.to_csv(DATA_DIR + f"CWI1_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
scaled_cdx_na_ig = scaled_cds['CDX NA IG'].dropna()

# Market Standards: SPX, VIX, Treasuries, EUR/USD rate
spx = con.bdh('SPX Index', ['PX_LAST', 'TOT_RETURN_INDEX_GROSS_DVDS'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
spx.to_csv(DATA_DIR + f"SPX_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
vix = con.bdh('VIX Index', ['PX_LAST'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
vix.to_csv(DATA_DIR + f"VIX_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
tu1 = con.bdh('TU1 Comdty', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
tu1.to_csv(DATA_DIR + f"TU1_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
tu2 = con.bdh('TU2 Comdty', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
tu2.to_csv(DATA_DIR + f"TU2_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
fv1 = con.bdh('FV1 Comdty', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
fv1.to_csv(DATA_DIR + f"FV1_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
fv2 = con.bdh('FV2 Comdty', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
fv2.to_csv(DATA_DIR + f"FV2_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
ty1 = con.bdh('TY1 Comdty', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ty1.to_csv(DATA_DIR + f"TY1_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
ty2 = con.bdh('TY2 Comdty', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ty2.to_csv(DATA_DIR + f"TY2_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
us1 = con.bdh('US1 Comdty', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
us1.to_csv(DATA_DIR + f"US1_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
us2 = con.bdh('US2 Comdty', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
us2.to_csv(DATA_DIR + f"US2_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")
eurusd = con.bdh('EURUSD BGN Curncy', ['PX_LAST'],
                 f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
eurusd.to_csv(DATA_DIR + f"EURUSD_bbg_{END_DATE.strftime('%Y-%m-%d')}.csv")


###############################################################################

# This is literally insane, but have to fix maturity date settlement values from Bloomberg for iBoxx futures
# NOTE: only affects maturity date (first of month for iBoxx) on generic 1st term futures
iboxx_maturities = generate_expiries(START_DATE, END_DATE, specific_product='iBoxx')
iby1.loc[iby1.index & iboxx_maturities, 'PX_LAST'] = \
    ibxxibhy.loc[iby1.index & iboxx_maturities, 'PX_LAST'].round(2).values
ihb1.loc[ihb1.index & iboxx_maturities, 'PX_LAST'] = \
    ibxxibig.loc[ihb1.index & iboxx_maturities, 'PX_LAST'].round(2).values


###############################################################################
# Calculate bespoke rolled futures

iby_roll_df = stitch_bloomberg_futures(iby1['PX_LAST'], iby2['PX_LAST'], roll_n_before_expiry=3,
                                       start_datelike=START_DATE, end_datelike=END_DATE,
                                       specific_product='iBoxx')
ihb_roll_df = stitch_bloomberg_futures(ihb1['PX_LAST'], ihb2['PX_LAST'], roll_n_before_expiry=3,
                                       start_datelike=START_DATE, end_datelike=END_DATE,
                                       specific_product='iBoxx')

tu_roll_df = stitch_bloomberg_futures(tu1['PX_LAST'], tu2['PX_LAST'], roll_n_before_expiry=3,
                                      start_datelike=START_DATE, end_datelike=END_DATE,
                                      specific_product='Treasury Futures 2')
fv_roll_df = stitch_bloomberg_futures(fv1['PX_LAST'], fv2['PX_LAST'], roll_n_before_expiry=3,
                                      start_datelike=START_DATE, end_datelike=END_DATE,
                                      specific_product='Treasury Futures 5')
ty_roll_df = stitch_bloomberg_futures(ty1['PX_LAST'], ty2['PX_LAST'], roll_n_before_expiry=3,
                                      start_datelike=START_DATE, end_datelike=END_DATE,
                                      specific_product='Treasury Futures 10')
us_roll_df = stitch_bloomberg_futures(us1['PX_LAST'], us2['PX_LAST'], roll_n_before_expiry=3,
                                      start_datelike=START_DATE, end_datelike=END_DATE,
                                      specific_product='Treasury Futures 30')


###############################################################################
# Calculate correlations

def create_rolling_corr_df(timeseries_1, timeseries_2, rolling_months=(1, 2, 3, 6), drop_or_ffill='drop'):
    """ Generate DataFrame of rolling correlations
        NOTE: generally for correlations, the 2 input time series are returns (% change); however, there may
              be cases where instead you care about level change (subtraction) or even raw level numbers
    :param timeseries_1: time series dataset 1
    :param timeseries_2: time series dataset 2
    :param rolling_months: number(s) of months for the rolling window; dimension will be number of DF columns
    :param drop_or_ffill: set 'drop' to drop dates that are not in common; set 'ffill' to forward-fill NaNs
    :return: pd.DataFrame with 'Rolling {n} Month' columns containing rolling correlation time series
    """
    ts_df = pd.DataFrame({'TS1': timeseries_1, 'TS2': timeseries_2})    # Contains NaN on uncommon dates
    if drop_or_ffill == 'drop':
        ts_df = ts_df.dropna(how='any')
    elif drop_or_ffill == 'ffill':
        ts_df = ts_df.fillna(method='ffill')
    else:
        raise ValueError(f"drop_or_ffill must be either 'drop' or 'ffill', not '{drop_or_ffill}'")
    corr_dict = {}
    if not isinstance(rolling_months, Iterable):
        rolling_months = [rolling_months]
    for n_month_window in rolling_months:
        corr_dict[n_month_window] = \
            ts_df.iloc[:, 0].rolling(n_month_window * 21, center=False).corr(ts_df.iloc[:, 1]).dropna()
    corr_df = pd.DataFrame({f'Rolling {n} Month': corr_dict[n] for n in rolling_months})
    corr_df.index.name = 'Trade Date'
    return corr_df


def calc_overall_corr(timeseries_1, timeseries_2, start_datelike=None, end_datelike=None, drop_or_ffill='drop'):
    """ Calculate overall correlation between two time series
    :param timeseries_1: time series dataset 1
    :param timeseries_2: time series dataset 2
    :param start_datelike: date-like representation of start date; set None to use entirety of time series
    :param end_datelike: date-like representation of end date; set None to use entirety of time series
    :param drop_or_ffill: set 'drop' to drop dates that are not in common; set 'ffill' to forward-fill NaNs
    :return: number between -1 and 1
    """
    ts_df = pd.DataFrame({'TS1': timeseries_1, 'TS2': timeseries_2})  # Contains NaN on uncommon dates
    if drop_or_ffill == 'drop':
        ts_df = ts_df.dropna(how='any')
    elif drop_or_ffill == 'ffill':
        ts_df = ts_df.fillna(method='ffill')
    else:
        raise ValueError(f"drop_or_ffill must be either 'drop' or 'ffill', not '{drop_or_ffill}'")
    ts_df_cropped = ts_df.loc[start_datelike:end_datelike]
    return ts_df_cropped.corr().iloc[1, 0]  # Get element from correlation matrix


CORR_START = None   # '2019-07-01'
CORR_END = None
ASSET_CHANGE_DICT = {'IBHY': iby_roll_df['Stitched Change'],
                     'IBXXIBHY': ibxxibhy['PX_LAST'].pct_change(),
                     'HYG': hyg['TOT_RETURN_INDEX_GROSS_DVDS'].pct_change(),
                     'IBOXHY': iboxhy['PX_LAST'].pct_change(),
                     'CDX NA HY': scaled_cdx_na_hy.pct_change(),
                     'SPX': spx['TOT_RETURN_INDEX_GROSS_DVDS'].pct_change(),
                     'VIX': vix['PX_LAST'].pct_change(),
                     '2-Year Treasury Futures': tu_roll_df['Stitched Change'],
                     '5-Year Treasury Futures': fv_roll_df['Stitched Change'],
                     '10-Year Treasury Futures': ty_roll_df['Stitched Change'],
                     '30-Year Treasury Futures': us_roll_df['Stitched Change'],
                     'EUR-USD': eurusd['PX_LAST'].pct_change(),
                     'IBIG': ihb_roll_df['Stitched Change'],
                     'IBXXIBIG': ibxxibig['PX_LAST'].pct_change(),
                     'LQD': lqd['TOT_RETURN_INDEX_GROSS_DVDS'].pct_change(),
                     'IBOXIG': iboxig['PX_LAST'].pct_change(),
                     'CDX NA IG': scaled_cdx_na_ig.pct_change()}

# IBHY
ibhy1_change = ASSET_CHANGE_DICT['IBHY']
ibhy_corr_targets = ['IBXXIBHY', 'HYG', 'IBOXHY', 'CDX NA HY', 'SPX', 'VIX', '2-Year Treasury Futures',
                     '5-Year Treasury Futures', '10-Year Treasury Futures', '30-Year Treasury Futures', 'EUR-USD']
ibhy_corr_df_dict = {}
ibhy_overall_corr_dict = {}
for corr_target in ibhy_corr_targets:
    ibhy_corr_df_dict[corr_target] = \
        create_rolling_corr_df(ibhy1_change, ASSET_CHANGE_DICT[corr_target]).loc[CORR_START:CORR_END]
    ibhy_overall_corr_dict[corr_target] = \
        calc_overall_corr(ibhy1_change, ASSET_CHANGE_DICT[corr_target], CORR_START, CORR_END)
    # Plot: various window rolls
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax.set_title(f'Correlation: IBHY Futures 1st Month vs. {corr_target}')
    for n in [1, 3, 6]:
        ax.plot(ibhy_corr_df_dict[corr_target][f'Rolling {n} Month'], label=f'{n}-Month Rolling Correlation')
    ax.axhline(ibhy_overall_corr_dict[corr_target], color='k', linestyle='--',
               label=f'Overall Correlation ({ibhy_overall_corr_dict[corr_target]*100:.1f}%)')
    ax.legend()
    save_fig(fig, f'IBHY1_{corr_target}_corr_rolling.png', DOWNLOADS_DIR)

# IBIG
ibig1_change = ASSET_CHANGE_DICT['IBIG']
ibig_corr_targets = ['IBXXIBIG', 'LQD', 'IBOXIG', 'CDX NA IG', 'SPX', 'VIX', '2-Year Treasury Futures',
                     '5-Year Treasury Futures', '10-Year Treasury Futures', '30-Year Treasury Futures', 'EUR-USD']
ibig_corr_df_dict = {}
ibig_overall_corr_dict = {}
for corr_target in ibig_corr_targets:
    ibig_corr_df_dict[corr_target] = \
        create_rolling_corr_df(ibig1_change, ASSET_CHANGE_DICT[corr_target]).loc[CORR_START:CORR_END]
    ibig_overall_corr_dict[corr_target] = \
        calc_overall_corr(ibig1_change, ASSET_CHANGE_DICT[corr_target], CORR_START, CORR_END)
    # Plot: various window rolls
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax.set_title(f'Correlation: IBIG Futures 1st Month vs. {corr_target}')
    for n in [1, 3, 6]:
        ax.plot(ibig_corr_df_dict[corr_target][f'Rolling {n} Month'], label=f'{n}-Month Rolling Correlation')
    ax.axhline(ibig_overall_corr_dict[corr_target], color='k', linestyle='--',
               label=f'Overall Correlation ({ibig_overall_corr_dict[corr_target]*100:.1f}%)')
    ax.legend()
    save_fig(fig, f'IBIG1_{corr_target}_corr_rolling.png', DOWNLOADS_DIR)


###############################################################################

# Export roll_df
iby_roll_df.to_csv(DOWNLOADS_DIR+'iby_roll_df.csv')
ihb_roll_df.to_csv(DOWNLOADS_DIR+'ihb_roll_df.csv')
for corr_target in ibhy_corr_targets:
    ibhy_corr_df_dict[corr_target].to_csv(DOWNLOADS_DIR+f'IBHY1_{corr_target}_corr_rolling.csv')
for corr_target in ibig_corr_targets:
    ibig_corr_df_dict[corr_target].to_csv(DOWNLOADS_DIR+f'IBIG1_{corr_target}_corr_rolling.csv')
