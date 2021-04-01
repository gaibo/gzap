import pandas as pd
import matplotlib.pyplot as plt
from futures_reader import create_bloomberg_connection
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
DATA_DIR = 'C:/Users/gzhang/Downloads/iBoxx Bloomberg Pulls/'
START_DATE = pd.Timestamp('2000-01-01')
END_DATE = pd.Timestamp('2021-03-31')
con = create_bloomberg_connection()

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

# Correlation
n = 6   # Months for rolling correlation

iby1_ibxxibhy = pd.DataFrame({'IBY1': iby1['PX_LAST'], 'IBXXIBHY': ibxxibhy['PX_LAST']}).sort_index()
iby1_ibxxibhy_change = iby1_ibxxibhy.pct_change()
iby1_ibxxibhy_corr = \
    iby1_ibxxibhy_change.iloc[:, 0].rolling(n*21, center=False).corr(iby1_ibxxibhy_change.iloc[:, 1]).dropna()

iby1_hyg = pd.DataFrame({'IBY1': iby1['PX_LAST'], 'HYG': hyg['TOT_RETURN_INDEX_GROSS_DVDS']}).sort_index()
iby1_hyg_change = iby1_hyg.pct_change()
iby1_hyg_corr = \
    iby1_hyg_change.iloc[:, 0].rolling(n*21, center=False).corr(iby1_hyg_change.iloc[:, 1]).dropna()

iby1_iboxhy = pd.DataFrame({'IBY1': iby1['PX_LAST'], 'IBOXHY': iboxhy['PX_LAST']}).sort_index()
iby1_iboxhy_change = iby1_iboxhy.pct_change()
iby1_iboxhy_corr = \
    iby1_iboxhy_change.iloc[:, 0].rolling(n*21, center=False).corr(iby1_iboxhy_change.iloc[:, 1]).dropna()

iby1_cdx_na_hy = pd.DataFrame({'IBY1': iby1['PX_LAST'], 'CDX NA HY': scaled_cdx_na_hy}).sort_index()
iby1_cdx_na_hy_change = iby1_cdx_na_hy.pct_change()
iby1_cdx_na_hy_corr = \
    iby1_cdx_na_hy_change.iloc[:, 0].rolling(n*21, center=False).corr(iby1_cdx_na_hy_change.iloc[:, 1]).dropna()

iby1_spx = pd.DataFrame({'IBY1': iby1['PX_LAST'], 'SPX': spx['TOT_RETURN_INDEX_GROSS_DVDS']}).sort_index()
iby1_spx_change = iby1_spx.pct_change()
iby1_spx_corr = \
    iby1_spx_change.iloc[:, 0].rolling(n*21, center=False).corr(iby1_spx_change.iloc[:, 1]).dropna()

iby1_vix = pd.DataFrame({'IBY1': iby1['PX_LAST'], 'VIX': vix['PX_LAST']}).sort_index()
iby1_vix_change = iby1_vix.pct_change()
iby1_vix_corr = \
    iby1_vix_change.iloc[:, 0].rolling(n*21, center=False).corr(iby1_vix_change.iloc[:, 1]).dropna()

iby1_fv1 = pd.DataFrame({'IBY1': iby1['PX_LAST'], 'FV1': fv1['PX_LAST']}).sort_index()
iby1_fv1_change = iby1_fv1.pct_change()
iby1_fv1_corr = \
    iby1_fv1_change.iloc[:, 0].rolling(n*21, center=False).corr(iby1_fv1_change.iloc[:, 1]).dropna()

iby1_eurusd = pd.DataFrame({'IBY1': iby1['PX_LAST'], 'EUR/USD': eurusd['PX_LAST']}).sort_index()
iby1_eurusd_change = iby1_eurusd.pct_change()
iby1_eurusd_corr = \
    iby1_eurusd_change.iloc[:, 0].rolling(n*21, center=False).corr(iby1_eurusd_change.iloc[:, 1]).dropna()

# Plot 1 - Various Window Rolls
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Correlation: Front Month IBHY Futures vs. HYG Total Return')
for n in [1, 3, 6]:
    ax.plot(iby1_hyg_change.iloc[:, 0].rolling(n*21, center=False).corr(iby1_hyg_change.iloc[:, 1]).dropna(),
            label=f'{n}-Month Rolling Correlation')
overall_corr = iby1_hyg_change.corr().iloc[1, 0]
ax.axhline(overall_corr, label=f'Overall Correlation ({overall_corr*100:.1f}%)',
           color='k', linestyle='--')
ax.legend()

# Plot 2 - Various Assets
_, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('6-Month Rolling Correlation: Front Month IBHY Futures vs. Various Assets')
for asset_name, asset_corr in [('IBXXIBHY', iby1_ibxxibhy_corr), ('HYG', iby1_hyg_corr), ('IBOXHY', iby1_iboxhy_corr),
                               ('CDX NA HY', iby1_cdx_na_hy_corr), ('SPX', iby1_spx_corr), ('VIX', iby1_vix_corr),
                               ('5y Tsy Futures', iby1_fv1_corr), ('EUR/USD', iby1_eurusd_corr)]:
    ax.plot(asset_corr, label=f'{asset_name}')
ax.legend()
