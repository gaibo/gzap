import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
TABLEAU_DATA_DIR = 'C:/Users/gzhang/Downloads/Tableau Pulled 2021-01-20/'


#### Liquidity 3-Stack: bid size, ask size, spread ############################

# Read data
liquidity_file_name = 'Near_Term_Futures__data (2).csv'     # [CONFIGURE]
liquidity_data = pd.read_csv(TABLEAU_DATA_DIR+liquidity_file_name, index_col='Trade Date', parse_dates=True)
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

# VX RTH vs. GTH
huge_spread_shortlist_rth = vx_rth[vx_rth['Avg. Time Wgtd Spread'] > 0.06]     # A lot of COVID days
# slightly_suspect_days_rth = huge_spread_shortlist_rth.loc[:'2020-02'].index     # Leave out COVID
covid_days = huge_spread_shortlist_rth.loc['2020-03':'2020-04'].index
slightly_suspect_days_rth = huge_spread_shortlist_rth.drop(covid_days).index  # Leave COVID spike
# Plot 3-stack bid ask spread
fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
fig.suptitle('VIX Liquidity')
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
# axs[2].plot(vx_gth['Avg. Time Wgtd Spread'], color='C1', label='GTH')
axs[2].legend()
axs[2].set_xlabel('Date')
# fig.set_tight_layout(True)

# VX GTH only
# # huge_spread_shortlist_gth = vx_gth[vx_gth['Avg. Time Wgtd Spread'] > 0.06]  # NOTE: doesn't work as well as with RTH: outliers not so clear
# # slightly_suspect_days_gth = huge_spread_shortlist_gth.loc[:'2020-02'].index
# slightly_suspect_days_gth = []
# # Plot 3-stack bid ask spread
# fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
# axs[0].set_title('Avg. Time Wgtd Bid Size')
# axs[0].plot(vx_gth['Avg. Time Wgtd Bid Size'].drop(slightly_suspect_days_gth),
#             color='C1', label='GTH')
# axs[0].legend()
# axs[1].set_title('Avg. Time Wgtd Ask Size')
# axs[1].plot(vx_gth['Avg. Time Wgtd Ask Size'].drop(slightly_suspect_days_gth),
#             color='C1', label='GTH')
# axs[1].legend()
# axs[2].set_title('Avg. Time Wgtd Spread')
# axs[2].plot(vx_gth['Avg. Time Wgtd Spread'].drop(slightly_suspect_days_gth),
#             color='C1', label='GTH')
# axs[2].legend()
# axs[2].set_xlabel('Date')
# fig.set_tight_layout(True)

# Rolling mean, minimalist version
# fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
# axs[0].set_title('Avg Time-Weighted Bid Size (Rolling 5-Day Mean)')
# axs[0].plot(rth['Avg Time-Weighted Bid Size'].drop(slightly_suspect_days).rolling(5).mean(), color='C0', label='RTH')
# axs[0].plot(gth['Avg Time-Weighted Bid Size'].rolling(5).mean(), color='C1', label='GTH')
# axs[0].legend()
# axs[1].set_title('Avg Time-Weighted Ask Size (Rolling 5-Day Mean)')
# axs[1].plot(rth['Avg Time-Weighted Ask Size'].drop(slightly_suspect_days).rolling(5).mean(), color='C0', label='RTH')
# axs[1].plot(gth['Avg Time-Weighted Ask Size'].rolling(5).mean(), color='C1', label='GTH')
# axs[1].legend()
# axs[2].set_title('Avg Time-Weighted Spread')
# axs[2].plot(rth['Avg Time-Weighted Spread'].drop(slightly_suspect_days), color='C0', label='RTH')
# # axs[2].plot(gth['Avg Time-Weighted Spread'], color='C1', label='GTH')
# axs[2].legend()
# axs[2].set_xlabel('Date')
# fig.set_tight_layout(True)

# VIX TAS RTH vs. GTH
# huge_spread_shortlist_rth = vx_rth[vx_rth['Avg. Time Wgtd Spread'] > 0.06]     # A lot of COVID days
# slightly_suspect_days_rth = huge_spread_shortlist_rth.loc[:'2020-02'].index     # Leave out COVID
slightly_suspect_days_rth_tas = []
# Plot 3-stack bid ask spread
fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
fig.suptitle('VIX TAS Liquidity')
axs[0].set_title('Avg. Time Wgtd Bid Size')
axs[0].plot(vxt_rth['Avg. Time Wgtd Bid Size'].drop(slightly_suspect_days_rth_tas),
            color='C0', label='RTH')
axs[0].plot(vxt_gth['Avg. Time Wgtd Bid Size'], color='C1', label='GTH')
axs[0].legend()
axs[1].set_title('Avg. Time Wgtd Ask Size')
axs[1].plot(vxt_rth['Avg. Time Wgtd Ask Size'].drop(slightly_suspect_days_rth_tas),
            color='C0', label='RTH')
axs[1].plot(vxt_gth['Avg. Time Wgtd Ask Size'], color='C1', label='GTH')
axs[1].legend()
axs[2].set_title('Avg. Time Wgtd Spread')
axs[2].plot(vxt_rth['Avg. Time Wgtd Spread'].drop(slightly_suspect_days_rth_tas),
            color='C0', label='RTH')
axs[2].plot(vxt_gth['Avg. Time Wgtd Spread'], color='C1', label='GTH')
axs[2].legend()
axs[2].set_xlabel('Date')
# fig.set_tight_layout(True)

# VXM RTH vs. GTH
# huge_spread_shortlist_rth = vx_rth[vx_rth['Avg. Time Wgtd Spread'] > 0.06]     # A lot of COVID days
# slightly_suspect_days_rth = huge_spread_shortlist_rth.loc[:'2020-02'].index     # Leave out COVID
slightly_suspect_days_rth_vxm = []
# Plot 3-stack bid ask spread
fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
fig.suptitle('Mini VIX Liquidity')
axs[0].set_title('Avg. Time Wgtd Bid Size')
axs[0].plot(vxm_rth['Avg. Time Wgtd Bid Size'].drop(slightly_suspect_days_rth_vxm),
            color='C0', label='RTH')
axs[0].plot(vxm_gth['Avg. Time Wgtd Bid Size'], color='C1', label='GTH')
axs[0].legend()
axs[1].set_title('Avg. Time Wgtd Ask Size')
axs[1].plot(vxm_rth['Avg. Time Wgtd Ask Size'].drop(slightly_suspect_days_rth_vxm),
            color='C0', label='RTH')
axs[1].plot(vxm_gth['Avg. Time Wgtd Ask Size'], color='C1', label='GTH')
axs[1].legend()
axs[2].set_title('Avg. Time Wgtd Spread')
axs[2].plot(vxm_rth['Avg. Time Wgtd Spread'].drop(slightly_suspect_days_rth_vxm),
            color='C0', label='RTH')
axs[2].plot(vxm_gth['Avg. Time Wgtd Spread'], color='C1', label='GTH')
axs[2].legend()
axs[2].set_xlabel('Date')
# fig.set_tight_layout(True)


#### Monthly ADV Heatmap ######################################################

adv_change_file_name = 'CFE_Firm_Monthly_ADV_data (1).csv'  # [CONFIGURE]

# Load data
adv_change_data = pd.read_csv(TABLEAU_DATA_DIR+adv_change_file_name)
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
# top_firms = adv_change_data_indexed.groupby('Name')['ADV'].mean().sort_values(ascending=False).iloc[:10].index
top_firms = adv_change_data_indexed.groupby('Name')['ADV'].mean().sort_values(ascending=False).index

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


#### Historical Percentiles (Daily Accuracy) ##################################

# Load CFEVOLOI - good general purpose volume and OI dataset
cfevoloi_file_name = 'cfevoloi (3).csv'     # [CONFIGURE]
cfevoloi_usecols = ['VOLATILITY INDEX VOLUME', 'VOLATILITY INDEX OI',
                    'Corporate Bond High Yield Volume', 'Corporate Bond High Yield OI',
                    'Corporate Bond Liquid Investment Grade Volume', 'Corporate Bond Liquid Investment Grade OI',
                    'CBOE Mini-VIX VOLUME', 'CBOE Mini-VIX OI']
cfevoloi = pd.read_csv(TABLEAU_DATA_DIR+cfevoloi_file_name, skiprows=1, usecols=['Date']+cfevoloi_usecols,
                       index_col='Date', parse_dates=True).sort_index()
cfevoloi_rename_dict = dict(zip(cfevoloi_usecols,
                                ['VX Volume', 'VX OI', 'IBHY Volume', 'IBHY OI',
                                 'IBIG Volume', 'IBIG OI', 'VXM Volume', 'VXM OI']))
cfevoloi = cfevoloi.rename(cfevoloi_rename_dict, axis=1)

# Load horrible makeshift iBoxx OI because inception-2018-11-07 is not in CFEVOLOI lmao
iboxx_oi_file_name = 'Current_Open_Interest_and_Volume_by_Futures_Root_F_data (1).csv'  # [CONFIGURE]
iboxx_oi_data = pd.read_csv(TABLEAU_DATA_DIR+iboxx_oi_file_name, index_col='Trading Date', parse_dates=True).sort_index()
ibhy_oi_data = iboxx_oi_data[iboxx_oi_data['Product'] == 'IBHY']
ibhy_oi = ibhy_oi_data.groupby('Trading Date')['Current Open Interest'].sum()

#### iBoxx Volumes Breakdown

iboxx_file_name = 'iBoxx_Daily_Volume_data (IBHY).csv'     # [CONFIGURE]

# Load data
ibhy_data = pd.read_csv(TABLEAU_DATA_DIR+iboxx_file_name, parse_dates=['Month, Day, Year of Trading Dt'])
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
ibhy_volume_breakdown['Open Interest'] = ibhy_oi    # Can't use cfevoloi['IBHY OI'] because missing data
# ibhy_volume_breakdown = ibhy_volume_breakdown.drop(pd.Timestamp('2021-01-21'))

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

#### VIX (GTH Share)

# Read data - from Mini-VIX Futures Dashboard VXM Volume & OI Sheet, but need to manually edit to get days not months
vx_vxm_session_file_name = 'Volume_by_Trading_Session_data (3).csv'
vx_vxm_session = pd.read_csv(TABLEAU_DATA_DIR+vx_vxm_session_file_name, parse_dates=['Month, Day, Year of Trading Dt'])
vx_vxm_session = vx_vxm_session.rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1)

# Clean
vx = vx_vxm_session[vx_vxm_session['Futures Root'] == 'VX'].drop('Futures Root', axis=1)
vxm = vx_vxm_session[vx_vxm_session['Futures Root'] == 'VXM'].drop('Futures Root', axis=1)
vx_volumes = vx.groupby('Trade Date')['Volume'].sum()
vx_session_split = vx.groupby(['Trade Date', 'Trading Session'])['Volume'].sum() / vx_volumes
vx_gth_perc = (1 - vx_session_split.xs('US', level='Trading Session')) * 100

# # Plot - 2019-present
# fig, ax = plt.subplots(figsize=(19.2, 10.8))
# ax.set_title('Global Trading Hours')
# ax.bar(vx_volumes.loc['2019':].index, vx_volumes.loc['2019':], color='C0', label='VIX Futures Volume')
# ax.set_yticklabels([f'{int(x):,}' for x in ax.get_yticks().tolist()])
# ax.legend(loc=2)
# axr = ax.twinx()
# axr.plot(vx_gth_perc.rolling(21, center=True).mean().loc['2019':], color='C1', label='GTH Share, Rolling 1-Month Mean')
# axr.set_yticklabels([f'{int(x)}%' for x in axr.get_yticks().tolist()])
# axr.grid(False, which='major', axis='both')
# axr.legend(loc=1)
# ax.set_xlim(pd.Timestamp('2019-01-01'), vx_volumes.index[-1])
# fig.set_tight_layout(True)

# Plot - all of available Tableau history
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Global Trading Hours - VIX')
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

# Same thing with Mini VIX
vxm_volumes = vxm.groupby('Trade Date')['Volume'].sum()
vxm_session_split = vxm.groupby(['Trade Date', 'Trading Session'])['Volume'].sum() / vxm_volumes
vxm_gth_perc = (1 - vxm_session_split.xs('US', level='Trading Session')) * 100
# Plot - all of available Tableau history
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Global Trading Hours - Mini VIX')
ax.bar(vxm_volumes.index, vxm_volumes, color='C0', label='Mini VIX Futures Volume')
ax.set_yticklabels([f'{int(x):,}' for x in ax.get_yticks().tolist()])
ax.legend(loc=2)
axr = ax.twinx()
axr.plot(vxm_gth_perc, color='C1', label='GTH Share')
# axr.plot(vxm_gth_perc.rolling(21, center=True).mean(), color='C1', label='GTH Share, Rolling 1-Month Mean')
axr.set_yticklabels([f'{int(x)}%' for x in axr.get_yticks().tolist()])
axr.grid(False, which='major', axis='both')
axr.legend(loc=1)
# ax.set_xlim(pd.Timestamp('2019-01-01'), vx_volumes.index[-1])
fig.set_tight_layout(True)

#### Mini VIX Customer Share of Volume (using same data)

vxm_cti = vxm.groupby(['Trade Date', 'CTI'])['Volume'].sum()
vxm_customer = vxm_cti.xs(4, level='CTI')
vxm_mm = vxm_cti.drop(4, level='CTI').groupby('Trade Date').sum()
vxm_total = vxm_cti.groupby('Trade Date').sum()
vxm_customer_perc = (vxm_customer / vxm_total) * 100

# Plot - full history since 2020-08
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Customer Share - Mini VIX')
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

# Same thing with VIX
vx_cti = vx.groupby(['Trade Date', 'CTI'])['Volume'].sum()
vx_customer = vx_cti.xs(4, level='CTI')
vx_mm = vx_cti.drop(4, level='CTI').groupby('Trade Date').sum()
vx_total = vx_cti.groupby('Trade Date').sum()
vx_customer_perc = (vx_customer / vx_total) * 100
# Plot
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Customer Share - VIX')
ax.bar(vx_total.index, vx_total, color='C0', label='VIX Futures Volume')
ax.set_yticklabels([f'{int(x):,}' for x in ax.get_yticks().tolist()])
ax.legend(loc=2)
axr = ax.twinx()
# axr.plot(vx_customer_perc.loc[:], color='C1', label='Customer Share')
axr.plot(vx_customer_perc.rolling(21, center=True).mean(), color='C1', label='Customer Share, Rolling 1-Month Mean')
axr.set_yticklabels([f'{int(x)}%' for x in axr.get_yticks().tolist()])
axr.grid(False, which='major', axis='both')
axr.legend(loc=1)
# ax.set_xlim(pd.Timestamp('2019-01-01'), vx_volumes.index[-1])
fig.set_tight_layout(True)


#### New Mega Pull ############################################################

# VIX
vx_mega = pd.read_csv(TABLEAU_DATA_DIR + 'iBoxx_Daily_Volume_data (VX).csv',
                      parse_dates=['Month, Day, Year of Trading Dt'])
vx_mega = vx_mega.rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1)
# iBoxx style analysis
vx_day_cti_volume = vx_mega.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vx_days = vx_day_cti_volume.index.get_level_values(0).unique()
vx_volume_breakdown = pd.DataFrame(index=vx_days, columns=['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total'])
vx_volume_breakdown['CTI 1'] = vx_day_cti_volume.xs(1, level='CTI')
vx_volume_breakdown['CTI 2'] = vx_day_cti_volume.xs(2, level='CTI')
vx_volume_breakdown['CTI 3'] = vx_day_cti_volume.xs(3, level='CTI')
vx_volume_breakdown['CTI 4'] = vx_day_cti_volume.xs(4, level='CTI')
vx_volume_breakdown['Total'] = vx_volume_breakdown[['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4']].sum(axis=1)
# vx_volume_breakdown.index.name = 'Trade Date'
# Open interest separately
vx_volume_breakdown['Open Interest'] = cfevoloi['VX OI']    # Can't use cfevoloi['IBHY OI'] because missing data
# ibhy_volume_breakdown = ibhy_volume_breakdown.drop(pd.Timestamp('2021-01-21'))

# Aggregate by month (must do manually because # days varies)
unique_yearmonths = vx_volume_breakdown.index.strftime('%Y-%m').unique()
# Volumes
vx_volume_sums = pd.DataFrame(columns=['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total'])
for yearmonth in unique_yearmonths:
    vx_volume_sums.loc[yearmonth] = vx_volume_breakdown.loc[yearmonth, ['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total']].sum()

# Break down by firms
vx_unique_accounts = vx_mega.groupby(['Trade Date', 'CTI'])['Account '].nunique()
vx_unique_customer_accounts_daily = vx_unique_accounts.xs(4, level='CTI')
vx_monthly = pd.DataFrame(columns=['MM Volume', 'Customer Volume', 'Total Volume', 'Unique Customer Accounts'])
vx_mega_indexed = vx_mega.set_index('Trade Date')
for yearmonth in unique_yearmonths:
    cti_volume = vx_mega_indexed.loc[yearmonth].groupby('CTI')['Size'].sum()/2  # Litch's thing has 2xVolume
    mm_volume = cti_volume[cti_volume.index != 4].sum()
    customer_volume = cti_volume[cti_volume.index == 4].sum()
    cti_unique_names = vx_mega_indexed.loc[yearmonth].groupby('CTI')['Account '].nunique()
    cti_4 = cti_unique_names[cti_unique_names.index == 4]
    if cti_4.empty:
        unique_customers = 0
    else:
        unique_customers = cti_4.squeeze()
    vx_monthly.loc[yearmonth] = (mm_volume, customer_volume, mm_volume+customer_volume, unique_customers)

# Mini VIX - I'm too tired to change the name right now
vxm_mega = pd.read_csv(TABLEAU_DATA_DIR + 'iBoxx_Daily_Volume_data (VXM).csv',
                      parse_dates=['Month, Day, Year of Trading Dt'])
vxm_mega = vxm_mega.rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1)
# iBoxx style analysis
vxm_day_cti_volume = vxm_mega.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vxm_days = vxm_day_cti_volume.index.get_level_values(0).unique()
vxm_volume_breakdown = pd.DataFrame(index=vxm_days, columns=['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total'])
vxm_volume_breakdown['CTI 1'] = vxm_day_cti_volume.xs(1, level='CTI')
vxm_volume_breakdown['CTI 2'] = vxm_day_cti_volume.xs(2, level='CTI')
vxm_volume_breakdown['CTI 3'] = vxm_day_cti_volume.xs(3, level='CTI')
vxm_volume_breakdown['CTI 4'] = vxm_day_cti_volume.xs(4, level='CTI')
vxm_volume_breakdown['Total'] = vxm_volume_breakdown[['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4']].sum(axis=1)
# vxm_volume_breakdown.index.name = 'Trade Date'
# Open interest separately
vxm_volume_breakdown['Open Interest'] = cfevoloi['VXM OI']    # Can't use cfevoloi['IBHY OI'] because missing data
# ibhy_volume_breakdown = ibhy_volume_breakdown.drop(pd.Timestamp('2021-01-21'))

# Aggregate by month (must do manually because # days varies)
unique_yearmonths = vxm_volume_breakdown.index.strftime('%Y-%m').unique()
# Volumes
vxm_volume_sums = pd.DataFrame(columns=['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total'])
for yearmonth in unique_yearmonths:
    vxm_volume_sums.loc[yearmonth] = vxm_volume_breakdown.loc[yearmonth, ['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total']].sum()

# Break down by firms
vxm_unique_accounts = vxm_mega.groupby(['Trade Date', 'CTI'])['Account '].nunique()
vxm_unique_customer_accounts_daily = vxm_unique_accounts.xs(4, level='CTI')
vxm_monthly = pd.DataFrame(columns=['MM Volume', 'Customer Volume', 'Total Volume', 'Unique Customer Accounts'])
vxm_mega_indexed = vxm_mega.set_index('Trade Date')
for yearmonth in unique_yearmonths:
    cti_volume = vxm_mega_indexed.loc[yearmonth].groupby('CTI')['Size'].sum()/2  # Litch's thing has 2xVolume
    mm_volume = cti_volume[cti_volume.index != 4].sum()
    customer_volume = cti_volume[cti_volume.index == 4].sum()
    cti_unique_names = vxm_mega_indexed.loc[yearmonth].groupby('CTI')['Account '].nunique()
    cti_4 = cti_unique_names[cti_unique_names.index == 4]
    if cti_4.empty:
        unique_customers = 0
    else:
        unique_customers = cti_4.squeeze()
    vxm_monthly.loc[yearmonth] = (mm_volume, customer_volume, mm_volume+customer_volume, unique_customers)

#### GTH breakdown by CTI and number of accounts

# VIX
# Isolate session, volume by CTI
vx_asian = vx_mega[vx_mega['Trading Session'] == 'Asian']
vx_european = vx_mega[vx_mega['Trading Session'] == 'European']
vx_us = vx_mega[vx_mega['Trading Session'] == 'US']
vx_asian_day_cti_volume = vx_asian.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vx_asian_day_cti_volume_unstacked = vx_asian_day_cti_volume.unstack()
vx_european_day_cti_volume = vx_european.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vx_european_day_cti_volume_unstacked = vx_european_day_cti_volume.unstack()
vx_us_day_cti_volume = vx_us.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vx_us_day_cti_volume_unstacked = vx_us_day_cti_volume.unstack()

# Isolate CTI 4, volume by session
vx_cti4 = vx_mega[vx_mega['CTI'] == 4]
vx_cti4_day_session_volume = vx_cti4.groupby(['Trade Date', 'Trading Session'])['Size'].sum()/2
vx_cti4_day_session_volume_unstacked = vx_cti4_day_session_volume.unstack()

# Isolate CTI 4, unique accounts by session
vx_cti4_day_session_accounts = vx_cti4.groupby(['Trade Date', 'Trading Session'])['Account '].nunique()
vx_cti4_day_session_accounts_unstacked = vx_cti4_day_session_accounts.unstack()

# Mini VIX
# Isolate session, volume by CTI
vxm_asian = vxm_mega[vxm_mega['Trading Session'] == 'Asian']
vxm_european = vxm_mega[vxm_mega['Trading Session'] == 'European']
vxm_us = vxm_mega[vxm_mega['Trading Session'] == 'US']
vxm_asian_day_cti_volume = vxm_asian.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vxm_asian_day_cti_volume_unstacked = vxm_asian_day_cti_volume.unstack()
vxm_european_day_cti_volume = vxm_european.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vxm_european_day_cti_volume_unstacked = vxm_european_day_cti_volume.unstack()
vxm_us_day_cti_volume = vxm_us.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vxm_us_day_cti_volume_unstacked = vxm_us_day_cti_volume.unstack()

# Isolate CTI 4, volume by session
vxm_cti4 = vxm_mega[vxm_mega['CTI'] == 4]
vxm_cti4_day_session_volume = vxm_cti4.groupby(['Trade Date', 'Trading Session'])['Size'].sum()/2
vxm_cti4_day_session_volume_unstacked = vxm_cti4_day_session_volume.unstack()

# Isolate CTI 4, unique accounts by session
vxm_cti4_day_session_accounts = vxm_cti4.groupby(['Trade Date', 'Trading Session'])['Account '].nunique()
vxm_cti4_day_session_accounts_unstacked = vxm_cti4_day_session_accounts.unstack()
