import pandas as pd
import matplotlib.pyplot as plt
from futures_reader import create_bloomberg_connection
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
START_DATE = pd.Timestamp('2000-01-01')
END_DATE = pd.Timestamp('2021-03-26')
con = create_bloomberg_connection()
VIX_FUTURES = ['UX1 Index', 'UX2 Index', 'UX3 Index']
TEMP_TICKERS = ['SPVXSTR Index', 'SPVXSP Index']
TEMP_FIELDS = ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'EQY_SH_OUT', 'FUND_NET_ASSET_VAL', 'FUND_TOTAL_ASSETS']
data_raw = con.bdh(VIX_ETPS + VIX_FUTURES + TEMP_TICKERS,
                   TEMP_FIELDS,
                   f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}",
                   elms=[('currency','USD')])
data_raw = data_raw[VIX_ETPS + VIX_FUTURES + TEMP_TICKERS].copy()    # Enforce column order

# IBY1 futures, IBXXIBHY, HYG, IBOXHY, CWY1 futures
iby1 = con.bdh('IBY1 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}")
iby1 = pd.read_csv(DOWNLOADS_DIR + 'IBY1_bbg_2021-01-26.csv',
                   parse_dates=['Date'], index_col='Date')
hyg = pd.read_csv(DOWNLOADS_DIR + 'HYG_bbg_2021-01-26.csv',
                  parse_dates=['Date'], index_col='Date')
ibxxibhy = pd.read_csv(DOWNLOADS_DIR + 'IBXXIBHY_bbg_2021-01-26.csv',
                       parse_dates=['Date'], index_col='Date')

# Correlation
# hyg_ibxxibhy = (hyg.join(ibxxibhy, how='inner', rsuffix='_IBXXIBHY')
#                 .rename({'PX_LAST': 'HYG', 'PX_LAST_IBXXIBHY': 'IBXXIBHY'}, axis=1)
#                 .drop('PX_VOLUME', axis=1)
#                 .sort_index())
# hyg_ibxxibhy_change = hyg_ibxxibhy.pct_change()
# hyg_ibxxibhy_change = hyg_ibxxibhy_change.loc['2018-09-10':]
hyg_iby1 = (hyg.join(iby1, how='inner', rsuffix='_IBY1')
            .rename({'PX_LAST': 'HYG', 'PX_LAST_IBY1': 'IBY1'}, axis=1)
            .drop(['PX_VOLUME', 'PX_VOLUME_IBY1'], axis=1)
            .sort_index())
hyg_iby1_change = hyg_iby1.pct_change()

# Plot
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Correlation: HYG vs. Front Month IBHY Futures')
for n in range(1, 4):
    ax.plot(hyg_iby1_change['HYG'].rolling(n*21, center=False).corr(hyg_iby1_change['IBY1']),
            label=f'{n}-Month Rolling Correlation')
overall_corr = hyg_iby1_change.corr().iloc[1,0]
ax.axhline(overall_corr, label=f'Overall Correlation ({overall_corr*100:.1f}%)',
           color='k', linestyle='--')
ax.legend()

####

# LQD, IHB1 (1st term futures prices)
lqd = pd.read_csv(DOWNLOADS_DIR + 'LQD_bbg_2021-01-26.csv',
                  parse_dates=['Date'], index_col='Date')
ihb1 = pd.read_csv(DOWNLOADS_DIR + 'IHB1_bbg_2021-01-26.csv',
                   parse_dates=['Date'], index_col='Date')

# Correlation
# hyg_ibxxibhy = (hyg.join(ibxxibhy, how='inner', rsuffix='_IBXXIBHY')
#                 .rename({'PX_LAST': 'HYG', 'PX_LAST_IBXXIBHY': 'IBXXIBHY'}, axis=1)
#                 .drop('PX_VOLUME', axis=1)
#                 .sort_index())
# hyg_ibxxibhy_change = hyg_ibxxibhy.pct_change()
# hyg_ibxxibhy_change = hyg_ibxxibhy_change.loc['2018-09-10':]
lqd_ihb1 = (lqd.join(ihb1, how='inner', rsuffix='_IHB1')
            .rename({'PX_LAST': 'LQD', 'PX_LAST_IHB1': 'IHB1'}, axis=1)
            .drop(['PX_VOLUME', 'PX_VOLUME_IHB1'], axis=1)
            .sort_index())
lqd_ihb1_change = lqd_ihb1.pct_change()

# Plot
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Correlation: LQD vs. Front Month IBIG Futures')
for n in range(1, 4):
    ax.plot(lqd_ihb1_change['LQD'].rolling(n*21, center=False).corr(lqd_ihb1_change['IHB1']),
            label=f'{n}-Month Rolling Correlation')
overall_corr = lqd_ihb1_change.corr().iloc[1,0]
ax.axhline(overall_corr, label=f'Overall Correlation ({overall_corr*100:.1f}%)',
           color='k', linestyle='--')
ax.legend()
