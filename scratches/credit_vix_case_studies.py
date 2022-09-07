import pandas as pd
import matplotlib.pyplot as plt

# Testing out usage of custom FiveThirtyEight
plt.style.use('cboe-fivethirtyeight')

####

DATA_DIR_STABILIS = 'Y:/Research/Research1/Gaibo/Stabilis Starter Pack/Data/'
DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

# Load VIX
vix = pd.read_csv(DATA_DIR_STABILIS+'VIX.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
vix_ret = vix.pct_change().dropna() * 100
# Load TYVIX
tyvix = pd.read_csv(DATA_DIR_STABILIS+'TYVIX.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
tyvix_ret = tyvix.pct_change().dropna() * 100
# Load Credit VIX
# NA IG %
vixigp = pd.read_csv(DATA_DIR_STABILIS+'VIXIGP.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
vixigp_ret = vixigp.pct_change().dropna() * 100
# NA HY %
vixhyp = pd.read_csv(DATA_DIR_STABILIS+'VIXHYP.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
vixhyp_ret = vixhyp.pct_change().dropna() * 100
# EU IG %
vixiep = pd.read_csv(DATA_DIR_STABILIS+'VIXIEP.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
vixiep_ret = vixiep.pct_change().dropna() * 100
# EU HY %
vixxop = pd.read_csv(DATA_DIR_STABILIS+'VIXXOP.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
vixxop_ret = vixxop.pct_change().dropna() * 100

# Load NA
# Government bonds
govt = pd.read_csv(DATA_DIR_STABILIS+'GOVT.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
govt_ret = govt.pct_change().dropna() * 100
# IG bonds
lqd = pd.read_csv(DATA_DIR_STABILIS+'LQD.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
lqd_ret = lqd.pct_change().dropna() * 100
# HY bonds
hyg = pd.read_csv(DATA_DIR_STABILIS+'HYG.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
hyg_ret = hyg.pct_change().dropna() * 100
# Equities
spxt = pd.read_csv(DATA_DIR_STABILIS+'SPXT.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
spxt_ret = spxt.pct_change().dropna() * 100

# Load EU
# Government bonds
sega = pd.read_csv(DATA_DIR_STABILIS+'SEGA.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
sega_ret = sega.pct_change().dropna() * 100
# IG bonds
ieac = pd.read_csv(DATA_DIR_STABILIS+'IEAC.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
ieac_ret = ieac.pct_change().dropna() * 100
# HY bonds
ihyg = pd.read_csv(DATA_DIR_STABILIS+'IHYG.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
ihyg_ret = ihyg.pct_change().dropna() * 100
# Equities
sxxp = pd.read_csv(DATA_DIR_STABILIS+'SXXP.csv', index_col='Date', parse_dates=True, squeeze=True).sort_index()
sxxp_ret = sxxp.pct_change().dropna() * 100

####

# Illustrate effects on all VIXs and US assets
# VIX, TYVIX, Credit VIX
# on top of SPXT, GOVT
fig, axs = plt.subplots(2, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].plot(vix.loc['2018-06':'2020-01'], color='C0', alpha=1, label='VIX')
axs[0].plot(vixigp.loc['2018-06':'2020-01'], color='C2', alpha=1, label='VIXIGP')
axs[0].plot(vixhyp.loc['2018-06':'2020-01'], color='C3', alpha=1, label='VIXHYP')
axs[0].axvline(pd.Timestamp('2019-07-20'), linestyle='--', color='k')
axs[0].axvline(pd.Timestamp('2019-11'), linestyle='--', color='k')
axs_0r = axs[0].twinx()
axs_0r.plot(tyvix.loc['2018-06':'2020-01'], color='C1', alpha=0.5, label='TYVIX')
axs_0r.grid(False, which='major', axis='both')
axs[0].legend(loc=2, fontsize='x-large')
axs_0r.legend(loc=1, fontsize='x-large')
axs[1].plot(spxt.loc['2018-06':'2020-01'], color='C4', alpha=1, label='SPX Total Return')
axs_1r = axs[1].twinx()
axs_1r.plot(govt.loc['2018-06':'2020-01'], color='C5', alpha=1, label='GOVT - All Treasuries ETF')
axs_1r.grid(False, which='major', axis='both')
axs[1].axvline(pd.Timestamp('2019-07-20'), linestyle='--', color='k')
axs[1].axvline(pd.Timestamp('2019-11'), linestyle='--', color='k')
axs[1].legend(loc=3, fontsize='x-large')
axs_1r.legend(loc=4, fontsize='x-large')
axs[1].set_xlabel('Date', fontsize='x-large')
axs[0].set_title('2019 10-2 Inversion: VIX Indexes vs. Market Assets', fontsize='x-large')
fig.set_tight_layout(True)

# Illustrate US vs. Europe Credit VIX difference and supplement with US vs. Europe assets
# Normalize all the assets
spxt_norm = spxt.loc['2018-01':'2018-10'].copy()
spxt_norm = spxt_norm/spxt_norm[0] * 100
sxxp_norm = sxxp.loc['2018-01':'2018-10'].copy()
sxxp_norm = sxxp_norm/sxxp_norm[0] * 100
govt_norm = govt.loc['2018-01':'2018-10'].copy()
govt_norm = govt_norm/govt_norm[0] * 100
sega_norm = sega.loc['2018-01':'2018-10'].copy()
sega_norm = sega_norm/sega_norm[0] * 100
lqd_norm = lqd.loc['2018-01':'2018-10'].copy()
lqd_norm = lqd_norm/lqd_norm[0] * 100
hyg_norm = hyg.loc['2018-01':'2018-10'].copy()
hyg_norm = hyg_norm/hyg_norm[0] * 100
ieac_norm = ieac.loc['2018-01':'2018-10'].copy()
ieac_norm = ieac_norm/ieac_norm[0] * 100
ihyg_norm = ihyg.loc['2018-01':'2018-10'].copy()
ihyg_norm = ihyg_norm/ihyg_norm[0] * 100
# 4 Credit VIXs, maybe VIX
# on top of SPXT, GOVT
fig, axs = plt.subplots(2, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].plot(vixigp.loc['2018-01':'2018-10'], color='C2', alpha=1, label='VIXIGP')
axs[0].plot(vixhyp.loc['2018-01':'2018-10'], color='C3', alpha=1, label='VIXHYP')
axs[0].plot(vixiep.loc['2018-01':'2018-10'], color='C4', alpha=1, label='VIXIEP')
axs[0].plot(vixxop.loc['2018-01':'2018-10'], color='C5', alpha=1, label='VIXXOP')
axs[0].plot(vix.loc['2018-01':'2018-10'], color='C0', alpha=0.4, label='VIX')
axs[0].axvline(pd.Timestamp('2018-05'), linestyle='--', color='k')
axs[0].axvline(pd.Timestamp('2018-07'), linestyle='--', color='k')
# axs_0r = axs[0].twinx()
# axs_0r.plot(tyvix.loc['2018-01':'2018-10'], color='C1', alpha=0.4, label='TYVIX')
# axs_0r.grid(False, which='major', axis='both')
axs[0].legend(loc=2, fontsize='x-large')
# axs_0r.legend(loc=1, fontsize='x-large')
# axs[1].plot(spxt_norm, color='C1', alpha=1, label='SPX Total Return')
# axs[1].plot(sxxp_norm, color='C2', alpha=1, label='Euro Stoxx 600')
# axs[1].plot(govt_norm, color='C3', alpha=1, label='GOVT - US (IG) Treasuries ETF')
# axs[1].plot(sega_norm, color='C4', alpha=1, label='SEGA - EU (IG) Treasuries ETF')
axs[1].plot(lqd_norm, color='C0', alpha=1, label='LQD - NA IG Corp Bond ETF')
axs[1].plot(hyg_norm, color='C1', alpha=1, label='HYG - NA HY Corp Bond ETF')
axs[1].plot(ieac_norm, color='grey', alpha=1, label='IEAC - EU IG Corp Bond ETF')
axs[1].plot(ihyg_norm, color='brown', alpha=1, label='IHYG - EU HY Corp Bond ETF')
# axs_1r = axs[1].twinx()
# axs_1r.grid(False, which='major', axis='both')
axs[1].axvline(pd.Timestamp('2018-05'), linestyle='--', color='k')
axs[1].axvline(pd.Timestamp('2018-07'), linestyle='--', color='k')
axs[1].legend(loc=3, fontsize='x-large')
# axs_1r.legend(loc=4, fontsize='x-large')
axs[1].set_xlabel('Date', fontsize='x-large')
axs[0].set_title('Mid-2018 Europe-Focused Credit Vol', fontsize='x-large')
fig.set_tight_layout(True)
