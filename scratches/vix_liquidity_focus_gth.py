import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

#### Liquidity 3-Stack ########################################################

# Read data
liquidity_file_name = 'Near_Term_Futures__data (1).csv'
liquidity_data = pd.read_csv(DOWNLOADS_DIR+liquidity_file_name, index_col='Trade Date', parse_dates=True)

# Clean
vx = liquidity_data[liquidity_data['Futures Root'] == 'VX']
vxt = liquidity_data[liquidity_data['Futures Root'] == 'VXT']
vxm = liquidity_data[liquidity_data['Futures Root'] == 'VXM']
vx_rth = vx[vx['trading session'] == 'RTH']
vx_gth = vx[vx['trading session'] == 'GTH']
vxt_rth = vxt[vxt['trading session'] == 'RTH']
vxt_gth = vxt[vxt['trading session'] == 'GTH']
vxm_gth = vxm[vxm['trading session'] == 'GTH']
vxm_rth = vxm[vxm['trading session'] == 'RTH']
select_cols = ['Avg. Time Wgtd Bid Size', 'Avg. Time Wgtd Ask Size', 'Avg. Time Wgtd Spread']
vx_rth = vx_rth[select_cols]
vx_gth = vx_gth[select_cols]
vxt_rth = vxt_rth[select_cols]
vxt_gth = vxt_gth[select_cols]
vxm_rth = vxm_rth[select_cols]
vxm_gth = vxm_gth[select_cols]

# VX RTH
huge_spread_shortlist_rth = vx_rth[vx_rth['Avg. Time Wgtd Spread'] > 0.06]     # A lot of COVID days
slightly_suspect_days_rth = huge_spread_shortlist_rth.loc[:'2020-02'].index     # Leave out COVID
# Plot 3-stack bid ask spread
fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].set_title('Avg. Time Wgtd Bid Size')
axs[0].plot(vx_rth['Avg. Time Wgtd Bid Size'].drop(slightly_suspect_days_rth),
            color='C0', label='RTH')
axs[0].plot(vx_gth['Avg. Time Wgtd Bid Size'], color='C1', label='GTH')
axs[0].legend()
axs[1].set_title('Avg. Time Wgtd Ask Size')
axs[1].plot(vx_rth['Avg. Time Wgtd Ask Size'].drop(slightly_suspect_days_rth),
            color='C0', label='RTH')
axs[1].plot(vx_gth['Avg. Time Wgtd Ask Size'], color='C1', label='GTH')
axs[1].legend()
axs[2].set_title('Avg. Time Wgtd Spread')
axs[2].plot(vx_rth['Avg. Time Wgtd Spread'].drop(slightly_suspect_days_rth),
            color='C0', label='RTH')
axs[2].plot(vx_gth['Avg. Time Wgtd Spread'], color='C1', label='GTH')
axs[2].legend()
axs[2].set_xlabel('Date')
fig.set_tight_layout(True)

# VX GTH
# huge_spread_shortlist_gth = vx_gth[vx_gth['Avg. Time Wgtd Spread'] > 0.06]  # NOTE: doesn't work as well as with RTH: outliers not so clear
# slightly_suspect_days_gth = huge_spread_shortlist_gth.loc[:'2020-02'].index
slightly_suspect_days_gth = []
# Plot 3-stack bid ask spread
fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].set_title('Avg. Time Wgtd Bid Size')
axs[0].plot(vx_gth['Avg. Time Wgtd Bid Size'].drop(slightly_suspect_days_gth),
            color='C1', label='GTH')
axs[0].legend()
axs[1].set_title('Avg. Time Wgtd Ask Size')
axs[1].plot(vx_gth['Avg. Time Wgtd Ask Size'].drop(slightly_suspect_days_gth),
            color='C1', label='GTH')
axs[1].legend()
axs[2].set_title('Avg. Time Wgtd Spread')
axs[2].plot(vx_gth['Avg. Time Wgtd Spread'].drop(slightly_suspect_days_gth),
            color='C1', label='GTH')
axs[2].legend()
axs[2].set_xlabel('Date')
fig.set_tight_layout(True)


#### GTH Share ################################################################

# Read data - from Mini-VIX Futures Dashboard VXM Volume & OI Sheet, but need to manually edit to get days not months
vx_vxm_session_file_name = 'Volume_by_Trading_Session_data (3).csv'
vx_vxm_session = pd.read_csv(DOWNLOADS_DIR+vx_vxm_session_file_name, parse_dates=['Month, Day, Year of Trading Dt'])
vx_vxm_session = vx_vxm_session.rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1)

# Clean
vx = vx_vxm_session[vx_vxm_session['Futures Root'] == 'VX'].drop('Futures Root', axis=1)
vxm = vx_vxm_session[vx_vxm_session['Futures Root'] == 'VXM'].drop('Futures Root', axis=1)
vx_volumes = vx.groupby('Trade Date')['Volume'].sum()
vx_session_split = vx.groupby(['Trade Date', 'Trading Session'])['Volume'].sum() / vx_volumes
vx_gth_perc = (1 - vx_session_split.xs('US', level='Trading Session')) * 100

# Plot
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Global Trading Hours')
ax.bar(vx_volumes.loc['2019':].index, vx_volumes.loc['2019':], color='C0', label='VIX Futures Volume')
ax.set_yticklabels([f'{int(x):,}' for x in ax.get_yticks().tolist()])
ax.legend(loc=2)
axr = ax.twinx()
axr.plot(vx_gth_perc.rolling(21, center=True).mean().loc['2019':], color='C1', label='GTH Share, Rolling 1-Month Mean')
axr.set_yticklabels([f'{int(x)}%' for x in axr.get_yticks().tolist()])
axr.grid(False, which='major', axis='both')
axr.legend(loc=1)
ax.set_xlim(pd.Timestamp('2019-01-01'), vx_volumes.index[-1])
fig.set_tight_layout(True)

# Plot - all of available Tableau history
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Global Trading Hours')
ax.bar(vx_volumes.index, vx_volumes, color='C0', label='VIX Futures Volume')
ax.set_yticklabels([f'{int(x):,}' for x in ax.get_yticks().tolist()])
ax.legend(loc=2)
axr = ax.twinx()
axr.plot(vx_gth_perc.rolling(21, center=True).mean(), color='C1', label='GTH Share, Rolling 1-Month Mean')
axr.set_yticklabels([f'{int(x)}%' for x in axr.get_yticks().tolist()])
axr.grid(False, which='major', axis='both')
axr.legend(loc=1)
# ax.set_xlim(pd.Timestamp('2019-01-01'), vx_volumes.index[-1])
fig.set_tight_layout(True)
