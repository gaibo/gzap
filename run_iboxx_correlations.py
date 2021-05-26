import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET, generate_expiries
from futures_reader import create_bloomberg_connection, stitch_bloomberg_futures
import matplotlib.pyplot as plt
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

# Generate iBoxx maturities and surrounding "useful" dates
iboxx_maturities = generate_expiries(START_DATE, END_DATE, specific_product='iBoxx')
iboxx_maturities_df = pd.DataFrame({'Maturity': iboxx_maturities,
                                    'Selected Roll Date': iboxx_maturities - ROLL_N_BEFORE_EXPIRY*BUSDAY_OFFSET,
                                    'Post-Roll Return Date': iboxx_maturities - (ROLL_N_BEFORE_EXPIRY-1)*BUSDAY_OFFSET,
                                    'Bloomberg Stitch Date': iboxx_maturities + BUSDAY_OFFSET}).set_index('Maturity')
iboxx_maturities_df = iboxx_maturities_df[iby1.index[0]:iby1.index[-1]].copy()  # Crop

# This is literally insane, but have to fix maturity date settlement values from Bloomberg
iby1.loc[iboxx_maturities_df.index, 'PX_LAST'] = \
    ibxxibhy.loc[iboxx_maturities_df.index, 'PX_LAST'].round(2).values
ihb1.loc[ihb1.index & iboxx_maturities, 'PX_LAST'] = \
    ibxxibig.loc[ihb1.index & iboxx_maturities, 'PX_LAST'].round(2).values

###############################################################################

# [EXPERIMENTAL] Testing out functionalized version of this script
iby_roll_df_v2 = stitch_bloomberg_futures(iby1['PX_LAST'], iby2['PX_LAST'], iboxx_maturities_df)
iby_roll_df_v2_alt = stitch_bloomberg_futures(iby1['PX_LAST'], iby2['PX_LAST'], roll_n_before_expiry=3,
                                              start_datelike=START_DATE, end_datelike=END_DATE,
                                              specific_product='iBoxx')
# ihb_roll_df = stitch_bloomberg_futures(ihb1['PX_LAST'], ihb2['PX_LAST'], iboxx_maturities_df)
ihb_roll_df = stitch_bloomberg_futures(ihb1['PX_LAST'], ihb2['PX_LAST'], roll_n_before_expiry=3,
                                       start_datelike=START_DATE, end_datelike=END_DATE,
                                       specific_product='iBoxx')
