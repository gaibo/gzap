import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

#### Liquidity 3-Stack

liquidity_file_name = 'vix_near_term_futures_liquidity.csv'

# Read data
liquidity_data = pd.read_csv(DOWNLOADS_DIR+liquidity_file_name, index_col='Trade Date', parse_dates=True)
rth = liquidity_data[liquidity_data['Trading Session'] == 'RTH']
gth = liquidity_data[liquidity_data['Trading Session'] == 'GTH']

# Evaluate huge spread days
huge_spread_shortlist = rth[rth['Avg Time-Weighted Spread'] > 0.06]     # A lot of COVID days
huge_spread_days = rth[rth['Avg Time-Weighted Spread'] > 0.11].index
slightly_suspect_days = huge_spread_shortlist.loc[:'2020-02'].index

# Plot 3-stack bid ask spread, each with RTH and GTH
# Normal version
fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].set_title('Avg Time-Weighted Bid Size')
axs[0].plot(rth['Avg Time-Weighted Bid Size'].drop(slightly_suspect_days), color='C0', label='RTH')
axs[0].plot(gth['Avg Time-Weighted Bid Size'], color='C1', label='GTH')
axs[0].legend()
axs[1].set_title('Avg Time-Weighted Ask Size')
axs[1].plot(rth['Avg Time-Weighted Ask Size'].drop(slightly_suspect_days), color='C0', label='RTH')
axs[1].plot(gth['Avg Time-Weighted Ask Size'], color='C1', label='GTH')
axs[1].legend()
axs[2].set_title('Avg Time-Weighted Spread')
axs[2].plot(rth['Avg Time-Weighted Spread'].drop(slightly_suspect_days), color='C0', label='RTH')
# axs[2].plot(gth['Avg Time-Weighted Spread'], color='C1', label='GTH')
axs[2].legend()
axs[2].set_xlabel('Date')
fig.set_tight_layout(True)
# Rolling mean, minimalist version
fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].set_title('Avg Time-Weighted Bid Size (Rolling 5-Day Mean)')
axs[0].plot(rth['Avg Time-Weighted Bid Size'].drop(slightly_suspect_days).rolling(5).mean(), color='C0', label='RTH')
axs[0].plot(gth['Avg Time-Weighted Bid Size'].rolling(5).mean(), color='C1', label='GTH')
axs[0].legend()
axs[1].set_title('Avg Time-Weighted Ask Size (Rolling 5-Day Mean)')
axs[1].plot(rth['Avg Time-Weighted Ask Size'].drop(slightly_suspect_days).rolling(5).mean(), color='C0', label='RTH')
axs[1].plot(gth['Avg Time-Weighted Ask Size'].rolling(5).mean(), color='C1', label='GTH')
axs[1].legend()
axs[2].set_title('Avg Time-Weighted Spread')
axs[2].plot(rth['Avg Time-Weighted Spread'].drop(slightly_suspect_days), color='C0', label='RTH')
# axs[2].plot(gth['Avg Time-Weighted Spread'], color='C1', label='GTH')
axs[2].legend()
axs[2].set_xlabel('Date')
fig.set_tight_layout(True)

#### Monthly ADV Heatmap

adv_change_file_name = 'CFE_Firm_Monthly_ADV_data.csv'

# Load data
adv_change_data = pd.read_csv(DOWNLOADS_DIR+adv_change_file_name)
# Reformat months
adv_change_data['Month of Trading Dt'] = pd.to_datetime(adv_change_data['Month of Trading Dt']).dt.strftime('%Y-%m')
# Combine Virtu name switch
# NOTE: okay I did this in literally the stupidest way but it works
adv_change_data_indexed = adv_change_data.set_index(['Name', 'Month of Trading Dt']).sort_index()
old_virtu = adv_change_data_indexed.loc['Virtu Financial BD LLC']
new_virtu = adv_change_data_indexed.loc['Virtu Americas LLC']
virtu_adv = old_virtu['ADV'].replace(np.NaN, 0) + new_virtu['ADV'].replace(np.NaN, 0)
virtu_diff = virtu_adv.diff()
virtu_df = pd.DataFrame({'Name': 'Virtu Financial BD + Americas',
                         'Month of Trading Dt': virtu_adv.index,
                         'ADV': virtu_adv.values,
                         'Difference in ADV': virtu_diff.values})
adv_change_data = adv_change_data.append(virtu_df, sort=False)
# Combine BofA name switch
old_bofa = adv_change_data_indexed.loc['Merrill Lynch, Pierce, Fenner & Smith Incorporated']
new_bofa = adv_change_data_indexed.loc['BofA Securities, Inc.']
bofa_adv = old_bofa['ADV'].replace(np.NaN, 0) + new_bofa['ADV'].replace(np.NaN, 0)
bofa_diff = bofa_adv.diff()
bofa_df = pd.DataFrame({'Name': 'BofA Securities + Merrill Lynch',
                        'Month of Trading Dt': bofa_adv.index,
                        'ADV': bofa_adv.values,
                        'Difference in ADV': bofa_diff.values})
adv_change_data = adv_change_data.append(bofa_df, sort=False)
# Drop combined names
adv_change_data_indexed = adv_change_data.set_index(['Name', 'Month of Trading Dt']).sort_index()
adv_change_data_indexed = adv_change_data_indexed.drop(['Virtu Financial BD LLC', 'Virtu Americas LLC',
                                                        'Merrill Lynch, Pierce, Fenner & Smith Incorporated', 'BofA Securities, Inc.'])

# Get list of top-trading firms over time period
top_firms = adv_change_data_indexed.groupby('Name')['ADV'].mean().sort_values(ascending=False).iloc[:10].index

# Separate adv and diff
indexed_adv = adv_change_data_indexed['ADV']
indexed_diff = adv_change_data_indexed['Difference in ADV']

# Fill in formatted DFs for export
adv_months = adv_change_data_indexed.index.get_level_values('Month of Trading Dt').unique()
adv_progression = pd.DataFrame(index=top_firms, columns=adv_months)
adv_change_progression = pd.DataFrame(index=top_firms, columns=adv_months)
for firm in top_firms:
    for month in adv_months:
        adv_progression.loc[firm, month] = indexed_adv.loc[(firm, month)]
        adv_change_progression.loc[firm, month] = indexed_diff.loc[(firm, month)]

#### Historical Percentiles (Daily Accuracy)

#### iBoxx Volumes Breakdown

iboxx_file_name = 'iBoxx_Daily_Volume_data.csv'
iboxx_oi_file_name = 'Current_Open_Interest_and_Volume_by_Futures_Root_F_data.csv'

# Load data
ibhy_data = pd.read_csv(DOWNLOADS_DIR+iboxx_file_name, parse_dates=['Month, Day, Year of Trading Dt'])
ibhy_size = ibhy_data.set_index(['Month, Day, Year of Trading Dt', 'CTI', 'Name']).sort_index()
ibhy_day_cti_volume = ibhy_size.groupby(['Month, Day, Year of Trading Dt', 'CTI'])['Size'].sum()/2
ibhy_days = ibhy_day_cti_volume.index.get_level_values(0).unique()
ibhy_volume_breakdown = pd.DataFrame(index=ibhy_days,
                                     columns=['CTI 1', 'CTI 2', 'CTI 4', 'Total'])
ibhy_volume_breakdown['CTI 1'] = ibhy_day_cti_volume.xs(1, level='CTI')
ibhy_volume_breakdown['CTI 2'] = ibhy_day_cti_volume.xs(2, level='CTI')
ibhy_volume_breakdown['CTI 4'] = ibhy_day_cti_volume.xs(4, level='CTI')
ibhy_volume_breakdown['Total'] = ibhy_volume_breakdown[['CTI 1', 'CTI 2', 'CTI 4']].sum(axis=1)
ibhy_volume_breakdown.index.name = 'Trading Date'
# Open interest separately
ibhy_oi_data = pd.read_csv(DOWNLOADS_DIR+iboxx_oi_file_name, index_col='Trading Date', parse_dates=True).sort_index()
ibhy_oi = ibhy_oi_data.groupby('Trading Date')['Current Open Interest'].sum()
ibhy_volume_breakdown['Open Interest'] = ibhy_oi

# Aggregate by month (must do manually because # days varies)
unique_yearmonths = ibhy_volume_breakdown.index.strftime('%Y-%m').unique()
# Volumes
volume_sums = pd.DataFrame(columns=['CTI 1', 'CTI 2', 'CTI 4', 'Total'])
for yearmonth in unique_yearmonths:
    volume_sums.loc[yearmonth] = ibhy_volume_breakdown.loc[yearmonth, ['CTI 1', 'CTI 2', 'CTI 4', 'Total']].sum()

# Break down by firms
ibhy_firm = ibhy_size.reset_index()
ibhy_firm = ibhy_firm.rename({'Month, Day, Year of Trading Dt': 'Trading Date'}, axis=1)
ibhy_unique_firms = ibhy_firm.groupby(['Trading Date', 'CTI'])['Name'].nunique()
unique_customers_daily = ibhy_unique_firms.xs(4, level='CTI')
ibhy_monthly = pd.DataFrame(columns=['MM Volume', 'Customer Volume', 'Total Volume', 'Unique Customers'])
ibhy_firm_indexed = ibhy_firm.set_index('Trading Date')
for yearmonth in unique_yearmonths:
    cti_volume = ibhy_firm_indexed.loc[yearmonth].groupby('CTI')['Size'].sum()
    mm_volume = cti_volume[cti_volume.index != 4].sum()
    customer_volume = cti_volume[cti_volume.index == 4].sum()
    cti_unique_names = ibhy_firm_indexed.loc[yearmonth].groupby('CTI')['Account '].nunique()
    cti_4 = cti_unique_names[cti_unique_names.index == 4]
    if cti_4.empty:
        unique_customers = 0
    else:
        unique_customers = cti_4.squeeze()
    ibhy_monthly.loc[yearmonth] = (mm_volume, customer_volume, mm_volume+customer_volume, unique_customers)

#### Mini VIX

vx_vxm_session_file_name = 'Volume_by_Trading_Session_data.csv'

vx_vxm_session = pd.read_csv(DOWNLOADS_DIR+vx_vxm_session_file_name, parse_dates=['Trade Date'])
vx = vx_vxm_session[vx_vxm_session['Futures Root'] == 'VX'].drop('Futures Root', axis=1)
vxm = vx_vxm_session[vx_vxm_session['Futures Root'] == 'VXM'].drop('Futures Root', axis=1)

vx_volumes = vx.groupby('Trade Date')['Volume'].sum()
vx_session_split = vx.groupby(['Trade Date', 'Trading Session'])['Volume'].sum() / vx_volumes
vx_gth_perc = (1 - vx_session_split.xs('US', level='Trading Session')) * 100

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

vxm_cti = vxm.groupby(['Trade Date', 'CTI'])['Volume'].sum()
vxm_customer = vxm_cti.xs(4, level='CTI')
vxm_mm = vxm_cti.drop(4, level='CTI').groupby('Trade Date').sum()
vxm_total = vxm_cti.groupby('Trade Date').sum()
vxm_customer_perc = (vxm_customer / vxm_total) * 100

fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Mini VIX')
ax.bar(vxm_total.index, vxm_total, color='C0', label='Mini VIX Futures Volume')
ax.set_yticklabels([f'{int(x):,}' for x in ax.get_yticks().tolist()])
ax.legend(loc=2)
axr = ax.twinx()
axr.plot(vxm_customer_perc.loc['2020-08-11':], color='C1', label='Customer Share')
# axr.plot(vxm_customer_perc.rolling(21, center=True).mean(), color='C1', label='Customer Share, Rolling 1-Month Mean')
axr.set_yticklabels([f'{int(x)}%' for x in axr.get_yticks().tolist()])
axr.grid(False, which='major', axis='both')
axr.legend(loc=1)
# ax.set_xlim(pd.Timestamp('2019-01-01'), vx_volumes.index[-1])
fig.set_tight_layout(True)
