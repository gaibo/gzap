from options_futures_expirations_v3 import BUSDAY_OFFSET
test = pd.date_range('2018-01-01', '2022-07-31', freq=BUSDAY_OFFSET)
ser_test = pd.Series(test, index=test)
ser_processed = pd.Series()
for year_str in ['2018', '2019', '2020', '2021', '2022']:
    for month_str in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
        year_month = year_str + '-' + month_str
        try:
            ser_processed.loc[year_month] = ser_test.loc[year_month].shape[0]
        except KeyError:
            break

from treasury_rates_reader import *
from scipy.interpolate import CubicSpline
# Get CMT yields
treasury_yields = load_treasury_rates()
# Create cubic spline for test date using scipy's class
x = np.array(list(MATURITY_NAME_TO_DAYS_DICT.values()))
y = treasury_yields.loc['2021-02-22'].values/100
f = CubicSpline(x, y, bc_type='natural')    # Use bc_type='natural'
# Generate spline value (yield) for each day
input_l = list(range(1, 10951))
rate_l = f(input_l)
# Create Yiwei's comparison DF
tmp = pd.DataFrame({'x':input_l, 'y':rate_l})
tmp['z'] = np.log( (1+tmp['y'].values/2)**2 )
comp_df = tmp[(tmp['x']>0)&(tmp['x']<36)].set_index('x')
print(comp_df*100)

# Create my comparison DF
my_bey = map((lambda x: get_rate('2021-02-22', x, return_rate_type='bey')), range(1, 36))
my_final = map((lambda x: get_rate('2021-02-22', x)), range(1, 36))
my_df = pd.DataFrame({'my_bey': my_bey, 'my_final': my_final}, index=range(1, 36))
print(my_df)

from futures_reader import create_bloomberg_connection


import matplotlib.pyplot as plt
from universal_tools import share_dateindex

effr_raw = pd.read_excel(AMERIBOR_DIR / 'EFFR.xlsx', parse_dates=['Effective Date'])
effr = effr_raw.set_index('Effective Date')['Rate (%)'].sort_index()

[rate_30, effr_30] = share_dateindex([ambor30t, effr], ffill=True)
spread_30 = rate_30 - effr_30
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.plot(rate_30, label='AMBOR30T')
ax.plot(effr_30, label='EFFR')
ax.plot(spread_30, label='Spread')
ax.axhline(y=0, color='k', linestyle='--')
ax.legend()
ax.set_title('AMBOR30T EFFR Spread')

[rate_90, effr_90] = share_dateindex([ambor90t, effr], ffill=True)
spread_90 = rate_90 - effr_90
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.plot(rate_90, label='AMBOR90T')
ax.plot(effr_90, label='EFFR')
ax.plot(spread_90, label='Spread')
ax.axhline(y=0, color='k', linestyle='--')
ax.legend()
ax.set_title('AMBOR90T EFFR Spread')

fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.plot(rate_30, label='AMBOR30T')
ax.plot(effr_30, label='EFFR')
ax.plot(effr_30 + spread_30.rolling(3*252).mean().shift(1),
        label='EFFR + 3-Year Average Spread')
ax.plot(effr_30 + spread_30.rolling(252).mean().shift(1),
        label='EFFR + 1-Year Average Spread')
ax.plot(effr_30 + spread_30.rolling(126).mean().shift(1),
        label='EFFR + 6-Month Average Spread')
ax.plot(effr_30 + spread_30.rolling(63).mean().shift(1),
        label='EFFR + 3-Month Average Spread')
ax.plot(effr_30 + spread_30.rolling(30).mean().shift(1),
        label='EFFR + 1-Month Average Spread')
ax.legend()
ax.set_title('AMBOR30T vs. EFFR Fallbacks')

fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.plot(rate_90, label='AMBOR90T')
ax.plot(effr_90, label='EFFR')
ax.plot(effr_90 + spread_90.rolling(3*252).mean().shift(1),
        label='EFFR + 3-Year Average Spread')
ax.plot(effr_90 + spread_90.rolling(252).mean().shift(1),
        label='EFFR + 1-Year Average Spread')
ax.plot(effr_90 + spread_90.rolling(126).mean().shift(1),
        label='EFFR + 6-Month Average Spread')
ax.plot(effr_90 + spread_90.rolling(63).mean().shift(1),
        label='EFFR + 3-Month Average Spread')
ax.plot(effr_90 + spread_90.rolling(30).mean().shift(1),
        label='EFFR + 1-Month Average Spread')
ax.legend()
ax.set_title('AMBOR90T vs. EFFR Fallbacks')

fig, ax = plt.subplots(figsize=(19.2, 10.8))
test[test['Issuer Name'] == 'HANNOVER FDG CO. LLC  ']['Interest Rate'].plot(label='Interest Rate')
ambor30t_250.plot(label='AMBOR30T with 250bp Trimming')
ambor30t_100.plot(label='AMBOR30T with 100bp Trimming')
ax.set_title('HANNOVER FDG CO. LLC: High Impact Commercial Paper Issuances')
ax.legend(loc=2)
# axr = ax.twinx()
# hannover_dbpv = test[test['Issuer Name'] == 'HANNOVER FDG CO. LLC  '].loc['2016-06-01':].groupby('Date')['DBPV'].max()
# total_dbpv = test.loc['2016-06-01':].groupby('Date')['DBPV'].sum()
# hannover_pct = (hannover_dbpv/total_dbpv*100).dropna()
# axr.bar(hannover_pct.index, hannover_pct, color='C2', alpha=0.6, label='Relative % Contribution')
# axr.legend(loc=1)

----

# Mike Margolis exports

import matplotlib.pyplot as plt
from universal_tools import share_dateindex

effr_raw = pd.read_excel(AMERIBOR_DIR / 'EFFR.xlsx', parse_dates=['Effective Date'])
effr = effr_raw.set_index('Effective Date')['Rate (%)'].sort_index()
ambor30t = test_rates.loc['2016-06-01':].copy()
[rate_30, effr_30] = share_dateindex([ambor30t, effr], ffill=True)
spread_30 = rate_30 - effr_30

import matplotlib as mpl
plt.style.use('cboe-fivethirtyeight')
mpl.rcParams['lines.linewidth'] = 2.5

# EFFR + Spreads
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.plot(rate_30, label='AMBOR30T')
ax.plot(effr_30, label='EFFR')
ax.plot(effr_30 + spread_30.rolling(3*252).mean().shift(1),
        label='EFFR + 3-Year Average Spread')
# ax.plot(effr_30 + spread_30.rolling(252).mean().shift(1),
#         label='EFFR + 1-Year Average Spread')
ax.plot(effr_30 + spread_30.rolling(126).mean().shift(1),
        label='EFFR + 6-Month Average Spread')
# ax.plot(effr_30 + spread_30.rolling(63).mean().shift(1),
#         label='EFFR + 3-Month Average Spread')
ax.plot(effr_30 + spread_30.rolling(21).mean().shift(1),
        label='EFFR + 1-Month Average Spread')
ax.legend(fontsize='x-large')
ax.set_title('AMBOR30T vs. EFFR Fallbacks', fontsize='x-large')
ax.axhline(y=0, color='k', linestyle='--')
ax.set_xlabel(None, fontsize='x-large')
ax.set_ylabel('Rate (%)', fontsize='x-large')
fig.set_tight_layout(True)

# How things would spike
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.plot(effr_30 - rate_30, label='EFFR-AMBOR30T Spread', color='C1')
effr_3y_spread_30 = effr_30 + spread_30.rolling(3*252).mean().shift(1)
effr_6m_spread_30 = effr_30 + spread_30.rolling(126).mean().shift(1)
effr_1m_spread_30 = effr_30 + spread_30.rolling(21).mean().shift(1)
ax.plot(effr_3y_spread_30 - rate_30,
        label='(EFFR + 3-Year Avg Spread)-AMBOR30T Spread', color='C2')
ax.plot(effr_6m_spread_30 - rate_30,
        label='(EFFR + 6-Month Avg Spread)-AMBOR30T Spread', color='C3')
ax.plot(effr_1m_spread_30 - rate_30,
        label='(EFFR + 1-Month Avg Spread)-AMBOR30T Spread', color='C4')
ax.plot(rate_30.shift(1) - rate_30,
        label='(1-Day Carry Over AMBOR30T)-AMBOR30T Spread', color='C5')
ax.legend(fontsize='x-large')
ax.set_title('Difference Between AMBOR30T and EFFR Fallbacks', fontsize='x-large')
ax.axhline(y=0, color='C0', linestyle='--')
ax.set_xlabel(None, fontsize='x-large')
ax.set_ylabel('Rate Difference (%)', fontsize='x-large')
fig.set_tight_layout(True)

# Day-to-next-day change
fig, ax = plt.subplots(figsize=(19.2, 10.8))
effr_3y_spread_30 = effr_30 + spread_30.rolling(3*252).mean().shift(1)
effr_6m_spread_30 = effr_30 + spread_30.rolling(126).mean().shift(1)
effr_1m_spread_30 = effr_30 + spread_30.rolling(21).mean().shift(1)
ax.plot(rate_30 - rate_30.shift(1),
        label='AMBOR30T (No Switch)', color='C0')
ax.plot(effr_3y_spread_30 - rate_30.shift(1),
        label='EFFR + 3-Year Avg Spread', color='C2')
ax.plot(effr_6m_spread_30 - rate_30.shift(1),
        label='EFFR + 6-Month Avg Spread', color='C3')
ax.plot(effr_1m_spread_30 - rate_30.shift(1),
        label='EFFR + 1-Month Avg Spread', color='C4')
ax.axhline(y=0, color='C5', linestyle='--', label='Carry Over Previous AMBOR30T Rate')
ax.legend(fontsize='x-large')
ax.set_title('Daily Rate Level Change Switching from AMBOR30T to Fallbacks', fontsize='x-large')
ax.set_xlabel(None, fontsize='x-large')
ax.set_ylabel('Rate Difference (%)', fontsize='x-large')
fig.set_tight_layout(True)

----

# How things would spike (bars)
fig, ax = plt.subplots(figsize=(19.2, 10.8))
effr_ambor30t_spread = effr_30 - rate_30
ax.bar(effr_ambor30t_spread.index, effr_ambor30t_spread, label='EFFR-AMBOR30T Spread', color='C1')
effr_3y_spread_30 = effr_30 + spread_30.rolling(3*252).mean().shift(1)
effr_1m_spread_30 = effr_30 + spread_30.rolling(21).mean().shift(1)
effr_3y_spread_30_ambor30t_spread = effr_3y_spread_30 - rate_30
effr_1m_spread_30_ambor30t_spread = effr_1m_spread_30 - rate_30
shifted_ambor30t_ambor30t_spread = rate_30.shift(1) - rate_30
ax.bar(effr_3y_spread_30_ambor30t_spread.index, effr_3y_spread_30_ambor30t_spread,
       label='(EFFR + 3-Year Avg Spread)-AMBOR30T Spread', color='C2')
ax.bar(effr_1m_spread_30_ambor30t_spread.index, effr_1m_spread_30_ambor30t_spread,
       label='(EFFR + 1-Month Avg Spread)-AMBOR30T Spread', color='C4')
ax.bar(shifted_ambor30t_ambor30t_spread.index, shifted_ambor30t_ambor30t_spread,
       label='(1-Day Carry Over AMBOR30T)-AMBOR30T Spread', color='C5')
ax.axhline(y=0, color='C0', linestyle='--', label='AMBOR30T Baseline')
ax.legend(fontsize='x-large')
ax.set_title('Difference Between AMBOR30T and EFFR Fallbacks', fontsize='x-large')
ax.set_xlabel(None, fontsize='x-large')
ax.set_ylabel('Rate Difference (%)', fontsize='x-large')
fig.set_tight_layout(True)

----

ax.set_title(f'Correlation: IBIG Futures 1st Month vs. {corr_target}')
for n in [1, 3, 6]:
    ax.plot(ibig_corr_df_dict[corr_target][f'Rolling {n} Month'], label=f'{n}-Month Rolling Correlation')
ax.axhline(ibig_overall_corr_dict[corr_target], color='k', linestyle='--',
           label=f'Overall Correlation ({ibig_overall_corr_dict[corr_target]*100:.1f}%)')
ax.legend()

new_test = [next_expiry('2000-01-01', quarterly_only(last_of_month), n, expiry_time=TREASURY_FUTURES_MATURITY_TIME) for n in range(1, 101)]
old_test = [next_treasury_futures_maturity('2000-01-01', n, tenor=2) for n in range(1, 101)]
assert new_test == old_test
new_test = [next_expiry('2000-01-01', quarterly_only(last_of_month), n, expiry_time=TREASURY_FUTURES_MATURITY_TIME) for n in range(1, 101)]
old_test = [next_treasury_futures_maturity('2000-01-01', n, tenor=5) for n in range(1, 101)]
assert new_test == old_test
new_test = [next_expiry('2000-01-01', quarterly_only(seventh_before_last_of_month), n, expiry_time=TREASURY_FUTURES_MATURITY_TIME) for n in range(1, 101)]
old_test = [next_treasury_futures_maturity('2000-01-01', n, tenor=10) for n in range(1, 101)]
assert new_test == old_test
new_test = [next_expiry('2000-01-01', quarterly_only(seventh_before_last_of_month), n, expiry_time=TREASURY_FUTURES_MATURITY_TIME) for n in range(1, 101)]
old_test = [next_treasury_futures_maturity('2000-01-01', n, tenor=30) for n in range(1, 101)]
assert new_test == old_test

new_test = [prev_expiry('2025-01-01', quarterly_only(last_of_month), n, expiry_time=TREASURY_FUTURES_MATURITY_TIME) for n in range(1, 101)]
old_test = [prev_treasury_futures_maturity('2025-01-01', n, tenor=2) for n in range(1, 101)]
assert new_test == old_test
new_test = [prev_expiry('2000-01-01', quarterly_only(last_of_month), n, expiry_time=TREASURY_FUTURES_MATURITY_TIME) for n in range(1, 101)]
old_test = [prev_treasury_futures_maturity('2000-01-01', n, tenor=5) for n in range(1, 101)]
assert new_test == old_test
new_test = [prev_expiry('2000-01-01', quarterly_only(seventh_before_last_of_month), n, expiry_time=TREASURY_FUTURES_MATURITY_TIME) for n in range(1, 101)]
old_test = [prev_treasury_futures_maturity('2000-01-01', n, tenor=10) for n in range(1, 101)]
assert new_test == old_test
new_test = [prev_expiry('2000-01-01', quarterly_only(seventh_before_last_of_month), n, expiry_time=TREASURY_FUTURES_MATURITY_TIME) for n in range(1, 101)]
old_test = [prev_treasury_futures_maturity('2000-01-01', n, tenor=30) for n in range(1, 101)]
assert new_test == old_test

cfevoloi_extension = pd.DataFrame({'Corporate Bond High Yield Volume': ibhy_vol,
                                   'Corporate Bond High Yield OI': ibhy_oi,
                                   'Corporate Bond Liquid Investment Grade Volume': ibig_vol,
                                   'Corporate Bond Liquid Investment Grade OI': ibig_oi})
cfevoloi_extension.index.name = 'Date'

def ytm_approx(c, fv, pv, t):
    return (c + (fv-pv)/t) / ((fv+pv)/2)


vx_firm_unique_accounts = vx_mega.groupby(['Trade Date', 'Name', 'CTI'])['Account '].nunique()
vx_firm_unique_customer_accounts_daily = vx_firm_unique_accounts.xs(4, level='CTI')
vx_firm_monthly = pd.DataFrame(columns=['MM Volume', 'Customer Volume', 'Total Volume', 'Unique Customer Accounts'])
vx_mega_indexed = vx_mega.set_index('Trade Date')   # This is the best way to get data by month
firm_dict = {}
for yearmonth in vx_mega_indexed.index.strftime('%Y-%m').unique():
    print(yearmonth, '...')
    firm_dict[yearmonth] = {}
    firm_cti_volume = vx_mega_indexed.loc[yearmonth].groupby(['Name', 'CTI'])['Size'].sum()/2  # Litch's thing has 2xVolume
    firm_cti_volume_unstack = firm_cti_volume.unstack()
    firm_dict[yearmonth]['Volume'] = firm_cti_volume_unstack
    # mm_volume = cti_volume[cti_volume.index != 4].sum()
    # customer_volume = cti_volume[cti_volume.index == 4].sum()
    firm_cti_accounts = vx_mega_indexed.loc[yearmonth].groupby(['Name', 'CTI'])['Account '].nunique()
    firm_cti_accounts_unstack = firm_cti_accounts.unstack()
    firm_dict[yearmonth]['Accounts'] = firm_cti_accounts_unstack
    # cti_4 = cti_unique_names[cti_unique_names.index == 4]
    # if cti_4.empty:
    #     unique_customers = 0
    # else:
    #     unique_customers = cti_4.squeeze()
    # vx_firm_monthly.loc[yearmonth] = (mm_volume, customer_volume, mm_volume+customer_volume, unique_customers)


legacy_tas_rename = \
    (legacy_tas
     .rename({'Entry Time': 'Trade Time', 'Date': 'Trade Date', 'Expiry Date': 'Expire Date',
              'Class': 'Class', 'Trade Size': 'Size', 'Trade Price': 'Price'}, axis=1))
legacy_tas_rename.index.name = 'Trade Time'     # From 'Entry Time'
missing_tas_rename = \
    (missing_tas
     .rename({'transact_time_no_tz': 'Trade Time', 'transact_date': 'Trade Date', 'expire_date': 'Expire Date',
              'futures_root': 'Class', 'size': 'Size', 'price': 'Price'}, axis=1))
missing_tas_rename.index.name = 'Trade Time'     # From 'transact_time_no_tz'
missing_tas_rename = missing_tas_rename.drop(['transact_time', 'tas'], axis=1)
modern_tas_rename = modern_tas.drop('Product Type', axis=1)

# Make legacy trade sizes consistent with the others (one-sided vs. two)
legacy_tas_rename['Size'] *= 2

# Combine
mega_tas_df = pd.concat([legacy_tas_rename.between_time('14:00', '15:15'),
                         missing_tas_rename.between_time('14:00', '15:15').loc['2018-02-24':'2018-03-19 16:00:01'],
                         modern_tas_rename.between_time('14:00', '15:15')], sort=False)
mega_tas_df = mega_tas_df[['Trade Date', 'Expire Date', 'Size', 'Price', 'Bid Price', 'Ask Price']]

mega_tas_200_215 = mega_tas_df.between_time('14:00', '14:15', include_end=False)    # Doing right-exclusive, though doesn't matter for TAS
mega_tas_215_230 = mega_tas_df.between_time('14:15', '14:30', include_end=False)
mega_tas_230_245 = mega_tas_df.between_time('14:30', '14:45', include_end=False)
mega_tas_245_300 = mega_tas_df.between_time('14:45', '15:00', include_end=False)
mega_tas_300_315 = mega_tas_df.between_time('15:00', '15:15', include_end=True)

result_tas_200 = mega_tas_200_215.groupby('Trade Date')['Size'].sum()
result_tas_215 = mega_tas_215_230.groupby('Trade Date')['Size'].sum()
result_tas_230 = mega_tas_230_245.groupby('Trade Date')['Size'].sum()
result_tas_245 = mega_tas_245_300.groupby('Trade Date')['Size'].sum()
result_tas_300 = mega_tas_300_315.groupby('Trade Date')['Size'].sum()
result_tas_df = pd.DataFrame({'Volume 2pm-2:15pm': result_tas_200, 'Volume 2:15pm-2:30pm': result_tas_215, 'Volume 2:30pm-2:45pm': result_tas_230,
                              'Volume 2:45pm-3pm': result_tas_245, 'Volume 3pm-3:15pm': result_tas_300})    # Automatically aligns index
result_tas_df.to_csv(DOWNLOADS_DIR + 'tas_volume_near_settlement.csv')


select_cols = ['Avg. Time Wgtd Bid Size', 'Avg. Time Wgtd Ask Size', 'Avg. Time Wgtd Spread']
vx_rth = vx_rth[select_cols]
vx_gth = vx_gth[select_cols]
vxm_rth = vxm_rth[select_cols]
vxm_gth = vxm_gth[select_cols]
vxt_rth = vxt_rth[select_cols]
vxt_gth = vxt_gth[select_cols]

# Evaluate huge spread days
huge_spread_shortlist_rth = vx_rth[vx_rth['Avg. Time Wgtd Spread'] > 0.06]     # A lot of COVID days
# huge_spread_days = rth[rth['Avg Time-Weighted Spread'] > 0.11].index
slightly_suspect_days_rth = huge_spread_shortlist_rth.loc[:'2020-02'].index
huge_spread_shortlist_gth = vx_gth[vx_gth['Avg. Time Wgtd Spread'] > 0.06]     # A lot of COVID days
# huge_spread_days = rth[rth['Avg Time-Weighted Spread'] > 0.11].index
slightly_suspect_days_gth = huge_spread_shortlist_gth.loc[:'2020-02'].index

# Plot 3-stack bid ask spread, each with RTH and GTH
# Normal version
fig, axs = plt.subplots(3, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].set_title('Avg. Time Wgtd Bid Size')
axs[0].plot(vx_rth['Avg. Time Wgtd Bid Size'].drop(slightly_suspect_days_rth),
            color='C0', label='RTH')
# axs[0].plot(vx_gth['Avg. Time Wgtd Bid Size'], color='C1', label='GTH')
axs[0].legend()
axs[1].set_title('Avg. Time Wgtd Ask Size')
axs[1].plot(vx_rth['Avg. Time Wgtd Ask Size'].drop(slightly_suspect_days_rth),
            color='C0', label='RTH')
# axs[1].plot(vx_gth['Avg. Time Wgtd Ask Size'], color='C1', label='GTH')
axs[1].legend()
axs[2].set_title('Avg. Time Wgtd Spread')
axs[2].plot(vx_rth['Avg. Time Wgtd Spread'].drop(slightly_suspect_days_rth),
            color='C0', label='RTH')
# axs[2].plot(vx_gth['Avg. Time Wgtd Spread'], color='C1', label='GTH')
axs[2].legend()
axs[2].set_xlabel('Date')
fig.set_tight_layout(True)

# GTH
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

def test(fdsa, fdsss, gjjdk):
    """
    :param fdsa:
    :param fdsss:
    :param gjjdk:
    :return:
    """

# Prune deleted remote branches that still appear in branch -a
git remote prune gaibo
# More explicit delete of those prune-able branches
git branch -d -r gaibo/upload_file

# Delete remote branch
git push -d gaibo upload_file
# Delete local branch (may need -D if branch unmerged)
git branch -d upload_file

# Fetch most recent branch data from remote
git fetch cboe

# Get a remote branch (first look with git branch -a)
git checkout --track origin/daves_branch

git push --set-upstream cboe develop

# This is how you change tracking
git branch master --set-upstream-to gaibo/master

use git push -u <remote>
to change remotes between cboe and gaibo, etc.

git push -u <remote> <branch>
and
git push --set-upstream <remote> <branch>
are the same thing

downloads_dir = 'C:/Users/gzhang/Downloads/'
from futures_reader import pull_fut_prices
fut_prices_1 = pull_fut_prices('SER', '2018-05-04', None, end_year_current=True, n_maturities_past_end=7,
                               contract_cycle='monthly', file_dir=downloads_dir, file_name='sofr_1_month_futures_prices.csv')
fut_prices_3 = pull_fut_prices('SFR', '2018-05-04', None, end_year_current=True, n_maturities_past_end=3,
                               contract_cycle='quarterly', file_dir=downloads_dir, file_name='sofr_3_month_futures_prices.csv')


def pull_fut_prices(start_datelike, end_datelike=None, bloomberg_con=None,
                    file_dir=BLOOMBERG_PULLS_FILEDIR, file_name=TREASURY_FUT_CSV_FILENAME):
    """ Pull Treasury futures prices from Bloomberg Terminal and write them to disk
    :param start_datelike: date-like representation of start date
    :param end_datelike: date-like representation of end date
    :param bloomberg_con: active pdblp Bloomberg connection; if None, runs create_bloomberg_connection()
    :param file_dir: directory to write data file (overrides default directory)
    :param file_name: exact file name to write to file_dir (overrides default file name)
    :return: pd.DataFrame with all Treasury futures prices between start and end dates
    """
    start_date = datelike_to_timestamp(start_datelike)
    if end_datelike is None:
        end_date = pd.Timestamp('now').normalize()
    else:
        end_date = datelike_to_timestamp(end_datelike)
    # Create list of all Treasury futures Bloomberg tickers in use between start and end dates
    ticker_list = []
    for tenor_code in TENOR_CODE_DICT.values():
        for year in range(start_date.year, end_date.year):
            # For all years up to but not including current year of end_date
            for quarter_code in QUARTER_CODE_LIST:
                ticker = tenor_code + quarter_code + f'{year%100:02d}' + ' Comdty'
                ticker_list.append(ticker)
        # For current year of end_date
        for quarter_code in QUARTER_CODE_LIST:
            ticker = tenor_code + quarter_code + f'{end_date.year%10}' + ' Comdty'
            ticker_list.append(ticker)
    # Get last price time-series of each ticker
    bbg_start_dt = start_date.strftime('%Y%m%d')
    bbg_end_dt = end_date.strftime('%Y%m%d')
    if bloomberg_con is None:
        bloomberg_con = create_bloomberg_connection()
        must_close_con = True
    else:
        must_close_con = False
    fut_price_df = bloomberg_con.bdh(ticker_list, 'PX_LAST', start_date=bbg_start_dt, end_date=bbg_end_dt)
    if must_close_con:
        bloomberg_con.stop()    # Close connection iff it was specifically made for this
    # Export
    fut_price_df.to_csv(file_dir + file_name)
    return fut_price_df


if coupon is None:
    # No fixed coupon given - clear sign of pricing generic cash flows
    if remaining_coupon_periods is None or remaining_payments is None:
        raise ValueError("coupon is None so function is pricing generic cash flows;\n"
                         "both remaining_coupon_periods and remaining_payments are expected.")
    discount_factors = 1 / (1 + ytm_semiannual) ** remaining_coupon_periods
    discounted_cash_flows = remaining_payments * discount_factors
    calc_price = discounted_cash_flows.sum()
    if verbose:
        for i, discounted_payment in enumerate(discounted_cash_flows, 1):
            print(f"Discounted Payment {i}: {discounted_payment}")
        print(f"Calculated Generic Price: {calc_price}")
    return calc_price

import matplotlib.pyplot as plt
from treasury_rates_reader import get_rate, load_treasury_rates, MATURITY_NAME_TO_DAYS_DICT
rates = load_treasury_rates()
days_to_maturity = range(1, 10951)
downloads_dir = 'C:/Users/gzhang/Downloads/'

test_treasury_spline = \
    pd.Series([get_rate('2015-01-22', i, loaded_rates=rates,
                        return_rate_type='treasury', drop_2_mo=False, use_spline_bounds=False)
               for i in days_to_maturity], index=days_to_maturity)
mat_yield = rates.loc['2015-01-22'].copy()
days_index = [MATURITY_NAME_TO_DAYS_DICT[mat] for mat in mat_yield.index]
mat_yield.index = days_index

# Plot
fig, ax = plt.subplots(figsize=(19.2, 10.8))
# ax.set_title('Problem rectified by $\geq$ instead of $>$ in $CMT_x$ determination', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(mat_yield, linestyle='None', marker='o', color='C0', label='CMT Yields (All 0.01)')
ax.plot(days_to_maturity, test_treasury_spline, marker=None, color='C0', label='Spline', linewidth=3)
# ax.plot(days_to_maturity, test_treasury_spline, marker=None, color='C1', label='Spline Bounded', linewidth=3)
# ax.axvline(30, linestyle='--', color='gray', label='Extrapolation/Interpolation Boundary')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
# ax.grid(which='both', axis='y')
# ax.set_xlim(left=0, right=750)
# ax.set_ylim(bottom=-0.5, top=1.75)
# ax.set_xlim(left=0, right=60)
# ax.set_ylim(bottom=-0.25, top=0.75)

#################################

cboe_calendar = mcal.get_calendar('NYSE')
trading_days = cboe_calendar.schedule(start_date=START_DATE, end_date=END_DATE).index
trading_days = pd.date_range(start=START_DATE, end=END_DATE, freq=BUSDAY_OFFSET)

#############3

import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET
from cme_eod_file_reader import read_cme_file
from xtp_eod_file_reader import read_xtp_file
from hanweck_eod_file_reader import read_hanweck_file, read_cme_or_hanweck_file

# Verify that CME, Hanweck, and XTP all match
trading_dates = pd.date_range(start='2020-01-20', end='2020-01-31', freq=BUSDAY_OFFSET)
for date in trading_dates:
    # Load
    cme = read_cme_file(10, date, verbose=False)    # Default to 'e' file, which contains complete prices
    xtp = read_xtp_file(10, date, verbose=False)
    hanweck = read_hanweck_file(10, date, verbose=False)
    # Isolate prices (for each series) for comparison
    cme_price = cme.set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])['Settlement']
    xtp_price = xtp.set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])['Settlement']
    hanweck_price = hanweck.set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])['Settlement']
    # Check equality
    date_str = date.strftime('%Y-%m-%d')
    if cme_price.equals(hanweck_price) and (hanweck_price.equals(xtp_price) or date_str == '2020-01-24'):
        print(f"{date_str}: PASS")  # 2020-01-24 is 1 of 2 known problematic XTP dates
    else:
        print(f"\n****{date_str}: FAIL****\n")

# Bonus: read_cme_or_hanweck_file() seamlessly transitions between CME and Hanweck on 2020-02-03
cmehanweck_cme = read_cme_or_hanweck_file(30, '2020-01-31')
cmehanweck_hanweck = read_cme_or_hanweck_file(30, '2020-02-03')
cmehanweck_hanweck_forced = read_cme_or_hanweck_file(30, '2020-01-31', force_use='hanweck')
assert cmehanweck_cme['Settlement'].equals(cmehanweck_hanweck_forced['Settlement'])

##################

import pandas as pd
from pandas.errors import EmptyDataError
from options_futures_expirations_v3 import BUSDAY_OFFSET
from cme_eod_file_reader import read_cme_file
from xtp_eod_file_reader import read_xtp_file
from hanweck_eod_file_reader import read_hanweck_file, read_cme_or_hanweck_file

READABLE_DATA_DATES = []    # Trading days on which there is usable data
HANWECK_EMPTY_DATES = []    # Trading days on which data files are empty (but at least there are columns)
EMPTY_DATA_DATES = []       # Trading days on which data files are empty
MISSING_DATA_DATES = []     # Trading days on which there are no data files
TRULY_DIFFERENT_DATES = []  # READABLE_DATA_DATES on which XTP and CME final prices do not match exactly
PRICE_CHANGE_DATES = []     # READABLE_DATA_DATES on which settlement prices change over course of snapshots

trading_days = pd.date_range(start='2019-10-28', end='2020-05-04', freq=BUSDAY_OFFSET)
for date_str in trading_days.strftime('%Y-%m-%d'):
    try:
        xtp = read_xtp_file(10, date_str, return_full=False)
        xtp_full = read_xtp_file(10, date_str, return_full=True)
        READABLE_DATA_DATES.append(date_str)
    except EmptyDataError:
        EMPTY_DATA_DATES.append(date_str)
        continue
    except FileNotFoundError:
        MISSING_DATA_DATES.append(date_str)
        continue
    # Perform operations on READABLE dates
    # 1) Load comparable XTP and CME prices and check/record differences
    xtp_prices = xtp.set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])['Settlement']
    try:
        cme_or_hanweck = read_cme_or_hanweck_file(10, date_str)
    except ValueError:
        print(f"\n****** {date_str} HANWECK BAD NO DATA ******\n")
        HANWECK_EMPTY_DATES.append(date_str)
        continue
    cme_or_hanweck_prices = cme_or_hanweck.set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])['Settlement']
    diff_bool = xtp_prices.ne(cme_or_hanweck_prices)
    diff_idx = diff_bool[diff_bool]     # Remove False rows for aesthetic
    n_diffs = diff_idx.sum()
    if n_diffs != 0:
        print(f"\n****** {date_str} BAD - NO MATCH {n_diffs} DIFFS ******")
        print(f"CME Official:\n{cme_or_hanweck_prices[diff_idx.index]}")
        print(f"XTP Capture:\n{xtp_prices[diff_idx.index]}")
        print(f"***********************************************\n")
        TRULY_DIFFERENT_DATES.append(date_str)
    # 2) Check/record if settlement prices changed over snapshots
    changesum = xtp_full.groupby(['Last Trade Date', 'Put/Call', 'Strike Price'])['Settlement'].diff().abs().sum()
    if changesum != 0:
        print(f"WARNING: {date_str} settlement price(s) "
              f"changed between snapshots; sum of changes is {changesum}.")
        PRICE_CHANGE_DATES.append(date_str)

print(f"\nTotal trading dates: {len(trading_days)}")
print(f"Readable data dates: {len(READABLE_DATA_DATES)}")
print(f"Readable data dates with differences to purchased CME data: {len(TRULY_DIFFERENT_DATES)}")
print(f"Readable data dates with settlement price changes: {len(PRICE_CHANGE_DATES)}")
print(f"File not found dates ({len(MISSING_DATA_DATES)}): {MISSING_DATA_DATES}")
print(f"File found but data empty dates ({len(EMPTY_DATA_DATES)}): {EMPTY_DATA_DATES}")
print(f"Hanweck inexplicably empty! ({len(HANWECK_EMPTY_DATES)}): {HANWECK_EMPTY_DATES}\n")

#######################

from cme_eod_file_reader import read_cme_file
from xtp_eod_file_reader import read_xtp_file
from hanweck_eod_file_reader import read_hanweck_file

date = '2020-01-24'
cme = read_cme_file(10, date, 'e').set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])['Settlement']
xtp = read_xtp_file(10, date).set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])['Settlement']
hanweck = read_hanweck_file(10, date).set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])['Settlement']

###############################################################################

from treasury_rates_reader import *
# import matplotlib.pyplot as plt
loaded_rates = pull_treasury_rates()
# loaded_rates.loc[pd.Timestamp('2020-04-25')] = [0.19, 0.26, 0.29, 0.37, 0.4, 0.55, 0.71, 0.92, 1.22, 1.42, 2.21, 2.59]
# loaded_rates.loc[pd.Timestamp('2020-04-26')] = [1.0, 0.85, 0.75, 0.8, 1.0, 1.5, 2, 1.75, 2.75, 3.5, 4, 4]
days_to_maturity = range(1, 10951)
downloads_dir = 'C:/Users/gzhang/Downloads/'

# CERT Tests
prod_1 = \
    pd.Series([get_rate('2020-04-27', i, loaded_rates=loaded_rates,
                        drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
prod_1.to_csv(downloads_dir+'2020-04-27_new_everything.csv', header=False)
cert_7 = \
    pd.Series([get_rate('2020-04-24', i, loaded_rates=loaded_rates,
                        drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
cert_7.to_csv(downloads_dir+'2020-04-24_new_everything.csv', header=False)
cert_6 = \
    pd.Series([get_rate('2020-04-23', i, loaded_rates=loaded_rates,
                        drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
cert_6.to_csv(downloads_dir+'2020-04-23_new_everything.csv', header=False)
cert_5 = \
    pd.Series([get_rate('2020-04-22', i, loaded_rates=loaded_rates,
                        drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
cert_5.to_csv(downloads_dir+'2020-04-22_new_everything.csv', header=False)
cert_4 = \
    pd.Series([get_rate('2020-04-21', i, loaded_rates=loaded_rates,
                        drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
cert_4.to_csv(downloads_dir+'2020-04-21_new_everything.csv', header=False)
cert_3 = \
    pd.Series([get_rate('2020-04-20', i, loaded_rates=loaded_rates,
                        drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
cert_3.to_csv(downloads_dir+'2020-04-20_new_everything.csv', header=False)
cert_2 = \
    pd.Series([get_rate('2020-04-17', i, loaded_rates=loaded_rates,
                        drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
cert_2.to_csv(downloads_dir+'2020-04-17_new_everything.csv', header=False)
cert_1 = \
    pd.Series([get_rate('2020-04-16', i, loaded_rates=loaded_rates,
                        drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
cert_1.to_csv(downloads_dir+'2020-04-16_new_everything.csv', header=False)

# # Initial Tests: 2020-03-30
# add_2_mo = \
#     pd.Series([get_rate('2020-03-30', i, loaded_rates=loaded_rates,
#                         return_rate_type='VIX', drop_2_mo=False, use_spline_bounds=False)
#                for i in days_to_maturity], index=days_to_maturity)
# add_2_mo.to_csv(downloads_dir+'2020-03-30_add_2_mo.csv', header=False)
# new_everything = \
#     pd.Series([get_rate('2020-03-30', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# new_everything.to_csv(downloads_dir+'2020-03-30_new_everything.csv', header=False)
#
# # "yesterday" Test: sum data mistaken as rates
# yesterday = \
#     pd.Series([get_rate('2020-04-25', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# yesterday.to_csv(downloads_dir+'yesterday_new_everything.csv', header=False)
#
# # Comprehensive Tests: 5 dates + "crazy"
# comp_1 = \
#     pd.Series([get_rate('2020-03-04', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# comp_1.to_csv(downloads_dir+'2020-03-04_new_everything.csv', header=False)
# comp_2 = \
#     pd.Series([get_rate('2020-03-24', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# comp_2.to_csv(downloads_dir+'2020-03-24_new_everything.csv', header=False)
# comp_3 = \
#     pd.Series([get_rate('2020-03-25', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# comp_3.to_csv(downloads_dir+'2020-03-25_new_everything.csv', header=False)
# comp_4 = \
#     pd.Series([get_rate('2020-03-26', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# comp_4.to_csv(downloads_dir+'2020-03-26_new_everything.csv', header=False)
# comp_5 = \
#     pd.Series([get_rate('2020-03-27', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# comp_5.to_csv(downloads_dir+'2020-03-27_new_everything.csv', header=False)
# comp_crazy = \
#     pd.Series([get_rate('2020-04-26', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# comp_crazy.to_csv(downloads_dir+'crazy_new_everything.csv', header=False)

###############################################################################

if ffill_by_1:
    day_yields = loaded_rates.loc[:date].fillna(method='ffill', limit=1).iloc[-1]
else:
    day_yields = loaded_rates.loc[:date].iloc[-1]

from treasury_rates_reader import *
import matplotlib.pyplot as plt
loaded_rates = pull_treasury_rates()
loaded_rates.loc[pd.Timestamp('2020-04-15')] = [0.19, 0.26, 0.29, 0.37, 0.4, 0.55, 0.71, 0.92, 1.22, 1.42, 2.21, 2.59]
loaded_rates.loc[pd.Timestamp('2020-04-16')] = [1.0, 0.85, 0.75, 0.8, 1.0, 1.5, 2, 1.75, 2.75, 3.5, 4, 4]
days_to_maturity = range(1, 10951)
downloads_dir = 'C:/Users/gzhang/Downloads/'

# Problem rectified by >= instead of > CMT_x determination
test = \
    pd.Series([get_rate('2020-03-24', i, loaded_rates=loaded_rates,
                        return_rate_type='treasury', drop_2_mo=False, use_spline_bounds=False)
               for i in days_to_maturity], index=days_to_maturity)
test_bounded = \
pd.Series([get_rate('2020-03-24', i, loaded_rates=loaded_rates,
                        return_rate_type='treasury', drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
mat_yield = loaded_rates.loc['2020-03-24'].copy()
days_index = [MATURITY_NAME_TO_DAYS_DICT[mat] for mat in mat_yield.index]
mat_yield.index = days_index
# Plot
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Problem rectified by $\geq$ instead of $>$ in $CMT_x$ determination', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(mat_yield[:3], linestyle='None', marker='o', color='C0', label='CMT Yields (All 0.01)')
ax.plot(days_to_maturity[:120], test[:120], marker=None, color='C0', label='Spline', linewidth=3)
ax.plot(days_to_maturity[:120], test_bounded[:120], marker=None, color='C1', label='Spline Bounded', linewidth=3)
ax.axvline(30, linestyle='--', color='gray', label='Extrapolation/Interpolation Boundary')
# ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
# ax.grid(which='both', axis='y')
# ax.set_xlim(left=0, right=750)
# ax.set_ylim(bottom=-0.5, top=1.75)
# ax.set_xlim(left=0, right=60)
# ax.set_ylim(bottom=-0.25, top=0.75)

spline_test_1 = \
    pd.Series([get_rate('2020-03-04', i, loaded_rates=loaded_rates,
                        return_rate_type='treasury', drop_2_mo=False, use_spline_bounds=False)
               for i in days_to_maturity], index=days_to_maturity)
spline_bounded_test_1 = \
    pd.Series([get_rate('2020-03-04', i, loaded_rates=loaded_rates,
                        return_rate_type='treasury', drop_2_mo=False, use_spline_bounds=True)
               for i in days_to_maturity], index=days_to_maturity)
spline_test_1.to_csv(downloads_dir+'spline_test_1.csv', header=False)
spline_bounded_test_1.to_csv(downloads_dir+'spline_bounded_test_1.csv', header=False)

# test_neelesh = \
#     pd.Series([get_rate('2020-04-15', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# test_neelesh.to_csv(downloads_dir+'neelesh_yesterday.csv')
#
# test_1 = \
#     pd.Series([get_rate('2020-03-04', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# test_1.to_csv(downloads_dir+'test_2020-03-04.csv')
#
# test_1_no_bounds = \
#     pd.Series([get_rate('2020-03-04', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=False)
#                for i in days_to_maturity], index=days_to_maturity)
# test_1_no_bounds.to_csv(downloads_dir+'test_2020-03-04_no_bounds.csv')
#
# test_2 = \
#     pd.Series([get_rate('2020-03-24', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# test_2.to_csv(downloads_dir+'test_2020-03-24.csv')
#
# test_3 = \
#     pd.Series([get_rate('2020-03-25', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# test_3.to_csv(downloads_dir+'test_2020-03-25.csv')
#
# test_4 = \
#     pd.Series([get_rate('2020-03-26', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# test_4.to_csv(downloads_dir+'test_2020-03-26.csv')
#
# test_5 = \
#     pd.Series([get_rate('2020-03-27', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# test_5.to_csv(downloads_dir+'test_2020-03-27.csv')
#
# test_6 = \
#     pd.Series([get_rate('2020-04-16', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# test_6.to_csv(downloads_dir+'test_crazy.csv')

# add_2_mo = \
#     pd.Series([get_rate('2020-03-30', i, loaded_rates=loaded_rates,
#                         return_rate_type='VIX', drop_2_mo=False, use_spline_bounds=False)
#                for i in days_to_maturity], index=days_to_maturity)
# new_everything = \
#     pd.Series([get_rate('2020-03-30', i, loaded_rates=loaded_rates,
#                         drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# add_2_mo.to_csv(downloads_dir+'add_2_mo.csv')
# new_everything.to_csv(downloads_dir+'new_everything.csv')


# cubic_spline_output = \
#     pd.Series([get_rate('2020-03-30', i, loaded_rates=loaded_rates,
#                         return_rate_type='treasury', drop_2_mo=False, use_spline_bounds=False)
#                for i in days_to_maturity], index=days_to_maturity)
# cubic_spline_bounded = \
#     pd.Series([get_rate('2020-03-30', i, loaded_rates=loaded_rates,
#                         return_rate_type='treasury', drop_2_mo=False, use_spline_bounds=True)
#                for i in days_to_maturity], index=days_to_maturity)
# cubic_spline_output.to_csv(downloads_dir+'spline.csv')
# cubic_spline_bounded.to_csv(downloads_dir+'spline_bounded.csv')


# idx_above = np.argmax(next_rates_rates >= cmt_1)     # 0 if no such rate (i.e. completely inverted)
# idx_below = np.argmax(next_rates_rates <= cmt_1)     # 0 if no such rate (i.e. no inversion)
# # Calculate lower and upper bounds
# if idx_above == 0:
#     m_lower = 0
#
# if idx_above == 0:
#     bound_lower = np.full_like(time_to_maturity, cmt_1, dtype=float)
# else:
#     t_above, cmt_above = rates_time_to_maturity[idx_above], rates_rates[idx_above]
#     m_lower = (cmt_above - cmt_1) / (t_above - t_1)     # >= 0 slope
#     b_lower = cmt_1 - m_lower*t_1
#     bound_lower = m_lower*time_to_maturity + b_lower
# if idx_below == 0:
#     bound_upper = np.full_like(time_to_maturity, cmt_1, dtype=float)
# else:
#     t_below, cmt_below = rates_time_to_maturity[idx_below], rates_rates[idx_below]
#     m_upper = (cmt_below - cmt_1) / (t_below - t_1)     # <= 0 slope
#     b_upper = cmt_1 - m_upper*t_1
#     bound_upper = m_upper*time_to_maturity + m_upper


import matplotlib.pyplot as plt
import numpy as np
from treasury_rates_reader import get_rate, load_treasury_rates, MATURITY_NAME_TO_DAYS_DICT
rates = load_treasury_rates()
rates.loc[pd.Timestamp('2020-03-23')] = [0.0, 0.0, 0.2, 1, 1.5, 2.5, 2, 1.75, 2.75, 3.5, 4, 4]
rates.loc[pd.Timestamp('2020-03-24')] = [1.0, 0.85, 0.75, 0.8, 1.0, 1.5, 2, 1.75, 2.75, 3.5, 4, 4]
rates.loc[pd.Timestamp('2020-03-25')] = [0.1, 0.5, 1.0, 1.2, 1.5, 1.5, 1.6, 1.75, 1.75, 1.8, 1.8, 1.8]
spline_current = pd.Series([get_rate('2020-03-23', i, rates, return_rate_type='treasury', conditional_neg_filter=False) for i in range(1, 11000)],
                           index=range(1, 11000))
spline_inverted = pd.Series([get_rate('2020-03-24', i, rates, return_rate_type='treasury', conditional_neg_filter=False) for i in range(1, 11000)],
                            index=range(1, 11000))
spline_okayneg = pd.Series([get_rate('2020-03-25', i, rates, return_rate_type='treasury', conditional_neg_filter=False) for i in range(1, 11000)],
                           index=range(1, 11000))
markers_x = [MATURITY_NAME_TO_DAYS_DICT[mat_string] for mat_string in rates.columns]
markers_y_current = rates.loc['2020-03-23'].values
markers_y_inverted = rates.loc['2020-03-24'].values
markers_y_okayneg = rates.loc['2020-03-25'].values

# Upper and Lower Bounds Illustration t < 30 Okay Negative
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Treasury CMT New Spline Bounds', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(markers_x[:6], markers_y_okayneg[:6], linestyle='None', marker='o', color='C0', label='Non-Inverted CMT Yields')
ax.plot(spline_okayneg, marker=None, color='C0', label='Spline', linewidth=4)
ax.axvline(30, linestyle='--', color='gray', label='Period Right Bound')
m_lower = (markers_y_okayneg[1] - markers_y_okayneg[0])/(markers_x[1] - markers_x[0])
b_lower = markers_y_okayneg[0] - m_lower*markers_x[0]
x = np.arange(0, 751)
ax.plot(x, m_lower*x + b_lower, marker=None, color='C1', label='Lower Bound', linestyle='--')
# m_upper = (markers_y_okayneg[1] - markers_y_okayneg[0])/(markers_x[1] - markers_x[0])
# b_upper = markers_y_okayneg[0] - m_upper*markers_x[0]
# ax.plot(x, m_upper*x + b_upper, marker=None, color='C2', label='Upper Bound', linestyle='--')
ax.axhline(markers_y_okayneg[0], linestyle='--', color='C2', label='Upper Bound')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
ax.grid(which='both', axis='y')
ax.set_xlim(left=0, right=750)
ax.set_ylim(bottom=-0.5, top=1.75)
# ax.set_xlim(left=0, right=60)
# ax.set_ylim(bottom=-0.25, top=0.75)

# Upper and Lower Bounds Illustration t < 30
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Treasury CMT New Spline Bounds', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(markers_x[:6], markers_y_inverted[:6], linestyle='None', marker='o', color='C0', label='Inverted CMT Yields')
ax.plot(spline_inverted, marker=None, color='C0', label='Spline', linewidth=4)
ax.axvline(30, linestyle='--', color='gray', label='Period Right Bound')
m_lower = (markers_y_inverted[5] - markers_y_inverted[0])/(markers_x[5] - markers_x[0])
b_lower = markers_y_inverted[0] - m_lower*markers_x[0]
x = np.arange(0, 751)
ax.plot(x, m_lower*x + b_lower, marker=None, color='C1', label='Lower Bound', linestyle='--')
m_upper = (markers_y_inverted[1] - markers_y_inverted[0])/(markers_x[1] - markers_x[0])
b_upper = markers_y_inverted[0] - m_upper*markers_x[0]
ax.plot(x, m_upper*x + b_upper, marker=None, color='C2', label='Upper Bound', linestyle='--')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
ax.grid(which='both', axis='y')
ax.set_xlim(left=0, right=750)
ax.set_ylim(bottom=0.5, top=1.75)
# ax.set_xlim(left=0, right=60)
# ax.set_ylim(bottom=0.9, top=1.3)

# Upper and Lower Bounds Illustration t > 30
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Treasury CMT New Spline Bounds', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(markers_x, markers_y_inverted, linestyle='None', marker='o', color='C0', label='Inverted CMT Yields')
ax.plot(spline_inverted, marker=None, color='C0', label='Spline')
ax.axvline(1095, linestyle='--', color='gray', label='Period Left Bound')
ax.axvline(1825, linestyle='--', color='gray', label='Period Right Bound')
lower_bound = min(markers_y_inverted[6], markers_y_inverted[7])
ax.axhline(lower_bound, linestyle='--', color='C1', label='Period Lower Bound')
upper_bound = max(markers_y_inverted[6], markers_y_inverted[7])
ax.axhline(upper_bound, linestyle='--', color='C2', label='Period Upper Bound')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
ax.grid(which='both', axis='y')
# ax.set_xlim(left=0, right=11000)
ax.set_xlim(left=1000, right=1900)
ax.set_ylim(bottom=1.5, top=2.25)

import matplotlib.pyplot as plt
from treasury_rates_reader import get_rate, load_treasury_rates, MATURITY_NAME_TO_DAYS_DICT
rates = load_treasury_rates()
rates.loc[pd.Timestamp('2020-03-17')] = [-0.25, -0.25, -0.24, -0.29, -0.29, -0.36, -0.43, -0.49, -0.67, -0.73, -1.10, -1.34]
rates.loc[pd.Timestamp('2020-03-18')] = rates.loc['2020-03-16'] - 0.5
rates.loc[pd.Timestamp('2020-03-19')] = [0.01, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rates.loc[pd.Timestamp('2020-03-20')] = [-0.01, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rates.loc[pd.Timestamp('2020-03-21')] = [-0.01, 5, 1, 2, 3, 4, 5, 4, 4, -4, 4, 4]
rates.loc[pd.Timestamp('2020-03-22')] = [-0.01, 0.5, 0.2, 1, 1.5, 2.5, 2, 1.75, 2.75, 3.5, 4, 4]
term_structure = pd.Series([get_rate('2020-03-16', i, rates) for i in range(1, 11000)])
neg_term_structure = pd.Series([get_rate('2020-03-17', i, rates) for i in range(1, 11000)])
sub_term_structure = pd.Series([get_rate('2020-03-18', i, rates) for i in range(1, 11000)])
insane_term_structure = pd.Series([get_rate('2020-03-19', i, rates) for i in range(1, 365)])
insane_neg_term_structure = pd.Series([get_rate('2020-03-20', i, rates) for i in range(1, 365)])
insane_t_gt_30_term_structure = pd.Series([get_rate('2020-03-21', i, rates) for i in range(1, 11000)])
reasonable_t_gt_30_term_structure = pd.Series([get_rate('2020-03-22', i, rates) for i in range(1, 11000)])
markers_x = [MATURITY_NAME_TO_DAYS_DICT[mat_string] for mat_string in rates.columns]
markers_y = rates.loc['2020-03-16'].values
markers_y_neg = rates.loc['2020-03-17'].values
markers_y_sub = rates.loc['2020-03-18'].values
markers_x_insane = markers_x[:5]
markers_y_insane = rates.loc['2020-03-19'].values[:5]
markers_y_insane_neg = rates.loc['2020-03-20'].values[:5]
markers_y_t_gt_30 = rates.loc['2020-03-21'].values
markers_y_t_gt_30_reasonable = rates.loc['2020-03-22'].values

# General Interpolation
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Treasury CMT Negative Rates Interpolation', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(markers_x, markers_y, linestyle='None', marker='o', color='C0', label='2020-03-16 CMT Yields')
ax.plot(term_structure, marker=None, color='C0', label='2020-03-16 CC Rates')
ax.plot(markers_x, markers_y_neg, linestyle='None', marker='o', color='C1', label='Negated CMT Yields')
ax.plot(neg_term_structure, marker=None, color='C1', label='Negated CC Rates')
ax.plot(markers_x, markers_y_sub, linestyle='None', marker='o', color='C2', label='Subtracted CMT Yields')
ax.plot(sub_term_structure, marker=None, color='C2', label='Subtracted CC Rates')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
ax.grid(which='both', axis='y')
ax.set_xlim(left=0, right=11000)
# ax.set_ylim(top=250)
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=50, ha='right')
# save_fig(fig, 'spx_vs_treasury_vega_volume')

# All Positives Insane
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Treasury CMT Negative Rates Interpolation', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(markers_x_insane, markers_y_insane, linestyle='None', marker='o', color='C3', label='Insane CMT Yields All Positive')
ax.plot(insane_term_structure, marker=None, color='C3', label='Insane CC Rates All Positive')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
ax.grid(which='both', axis='y')
ax.set_xlim(left=0, right=365)
ax.set_ylim(bottom=-7, top=5.5)

# With Negatives Insane
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Treasury CMT Negative Rates Interpolation', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(markers_x_insane, markers_y_insane_neg, linestyle='None', marker='o', color='C4', label='Insane CMT Yields with Negative')
ax.plot(insane_neg_term_structure, marker=None, color='C4', label='Insane CC Rates with Negative')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
ax.grid(which='both', axis='y')
ax.set_xlim(left=0, right=365)
ax.set_ylim(bottom=-7, top=5.5)

# t > 30 Extreme Case
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Treasury CMT Negative Rates Interpolation', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(markers_x, markers_y_t_gt_30, linestyle='None', marker='o', color='C5', label='Insane CMT Yields')
ax.plot(insane_t_gt_30_term_structure, marker=None, color='C5', label='Insane CC Rates')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
ax.grid(which='both', axis='y')
ax.set_xlim(left=0, right=11000)
# ax.set_ylim(bottom=-7, top=5.5)

# t > 30 Reasonable Case
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Treasury CMT Negative Rates Interpolation', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(markers_x, markers_y_t_gt_30_reasonable, linestyle='None', marker='o', color='C5', label='Reasonable CMT Yields')
ax.plot(reasonable_t_gt_30_term_structure, marker=None, color='C5', label='Reasonable CC Rates')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
ax.grid(which='both', axis='y')
ax.set_xlim(left=0, right=11000)
# ax.set_ylim(bottom=-7, top=5.5)

# 2005
front = pd.Timestamp('2005-01-15')
second = pd.Timestamp('2005-02-19')
trade_date = pd.Timestamp('2005-01-03')
term_structure = pd.Series([get_rate('2005-01-03', i, rates) for i in range(1, 11000)])
term_structure_vix = pd.Series([get_rate('2005-01-03', i, rates, return_rate_type='VIX') for i in range(1, 11000)])
markers_y = rates.loc['2005-01-03'].values
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('2005 Test', fontsize=16)
ax.set_xlabel('Maturity (Days)', fontsize=16)
ax.set_ylabel('Interpolated Rate (%)', fontsize=16)
ax.plot(markers_x, markers_y, linestyle='None', marker='o', color='C0', label='CMT Yields')
ax.plot(term_structure, marker=None, color='C0', label='CC Rates')
ax.plot(term_structure_vix, marker=None, color='C1', label='VIX Bad CC Rates')
ax.axhline(0, linestyle='--', color='k')
ax.legend(fontsize=16)
ax.grid(which='both', axis='y')
ax.set_xlim(left=0, right=11000)
# ax.set_ylim(bottom=-7, top=5.5)

def lookup_val_in_col(data, lookup_val, lookup_col, exact_only=False, groupby_cols=None):
    """ Return row (of first occurrence, if multiple) of nearest value in column
    :param data: input DataFrame
    :param lookup_val: value to look for in column
    :param lookup_col: column to look in
    :param exact_only: set True if only exact value match is desired
    :param groupby_cols: use instead of df.groupby(groupby_cols).apply(lambda data: lookup_val_in_col(...))
    :return: row (via index) containing column value that matches lookup value;
             if multiple matches, only first occurrence; if exact_only and no exact match, empty
    """
    if exact_only:
        exact_matches = data[data[lookup_col] == lookup_val]
        if groupby_cols is not None:
            return exact_matches.groupby(groupby_cols).first()
        else:
            return exact_matches
    col_val_abs_diff = (data[lookup_col] - lookup_val).abs()
    if groupby_cols is not None:
        # Aggregate by groupby_cols
        data_noindex = data.reset_index()
        data_noindex['col_val_abs_diff'] = col_val_abs_diff
        nearest_val_idxs = data_noindex.groupby(groupby_cols)['col_val_abs_diff'].idxmin()
        return data_noindex.loc[nearest_val_idxs].set_index(groupby_cols)
    else:
        # No need to aggregate
        nearest_val_idx = col_val_abs_diff.idxmin()
        if exact_only and col_val_abs_diff.loc[nearest_val_idx] != 0:
            return pd.DataFrame(columns=data.columns)   # New empty DataFrame
        else:
            return data.loc[nearest_val_idx].copy()


import pandas as pd
import matplotlib.pyplot as plt
from options_analytics import implied_vol_b76, delta_b76
from options_data_tools import remove_duplicate_series, add_t_to_exp, add_rate, add_forward, \
                               lookup_val_in_col, change_weekday
from timer_tools import Timer
from metaballon.DataSettings import OptionDataSources
from metaballon import OptionDataCenter, OptionDataSlip

TIMER = Timer()

# LOAD LIVEVOL DATA ###########################################################
TIMER.start("Pulling data from LiveVol...")
SOURCE = OptionDataSources.LIVEVOL
ROOT_LIST = ['SPX']
spx_options_dict = {}
for year in range(2006, 2020):
    start_date = pd.Timestamp(year, 1, 1)
    end_date = pd.Timestamp(year, 12, 31)
    slip = OptionDataSlip.make_option_slip_by_root(start_time=start_date, end_time=end_date,
                                                   root=ROOT_LIST, add_columns=['volume'],
                                                   data_src=SOURCE)
    try:
        spx_options_dict[year] = OptionDataCenter.run_option_slip(slip)
    except Exception as e:
        print("{} Exception: {}".format(year, e))
        continue
raw_data = pd.concat(spx_options_dict.values()).reset_index(drop=True)
TIMER.stop("{:,} rows".format(raw_data.shape[0]))
# Make copy of pulled data to work on
data_spx = (raw_data[['quote_date', 'expiry', 'pc', 'strike',
                      'bid', 'ask', 'mid', 'volume']]
            .sort_values(['quote_date', 'expiry', 'pc', 'strike'])
            .reset_index(drop=True))
###########################################################################

# LIVEVOL ONLY: filter out meaningless series with mid of 0, since it affects finding forward
TIMER.start("Removing series with mid-price of 0...")
data_spx = data_spx[data_spx['mid'] > 0]
TIMER.stop('{:,} rows'.format(data_spx.shape[0]))

TIMER.start("Removing duplicate series...")
data_spx = remove_duplicate_series(data_spx, 'quote_date', 'expiry', 'strike',
                                   'pc', 'volume')
TIMER.stop('{:,} rows'.format(data_spx.shape[0]))

TIMER.start("Correcting Saturday expirations to Friday for Greeks...")
data_spx = change_weekday(data_spx, 'expiry', 'Saturday', 'Friday', do_ensure_bus_day=True)
TIMER.stop('{:,} rows'.format(data_spx.shape[0]))

TIMER.start("Calculating time to expiration...")
data_spx = add_t_to_exp(data_spx, 'quote_date', 'expiry', 't_to_exp')
TIMER.stop('{:,} rows'.format(data_spx.shape[0]))

TIMER.start("Removing expired options...")
data_spx = data_spx[data_spx['t_to_exp'] > 0]
TIMER.stop("{:,} rows".format(data_spx.shape[0]))

# Filter out options past 70 days
TIMER.start("Removing options with expiries outside of 70 days...")
data_spx = data_spx[data_spx['t_to_exp'] <= 70/365].copy()
TIMER.stop("{:,} rows".format(data_spx.shape[0]))

TIMER.start("Getting rates...")
data_spx = add_rate(data_spx, 'quote_date', 't_to_exp', 'rate')
TIMER.stop("{:,} rows".format(data_spx.shape[0]))

TIMER.start("Calculating forward prices...")
data_spx = add_forward(data_spx, 'quote_date', 'expiry', 'strike',
                       'pc', 'mid', 't_to_exp', 'rate', 'forward')
TIMER.stop('{:,} rows'.format(data_spx.shape[0]))

# # LIVEVOL ONLY: data source has series with no trading volume
# TIMER.start("Filtering out no-volume series...")
# data_spx = data_spx[data_spx['volume'] > 0]
# TIMER.stop("{:,} rows".format(data_spx.shape[0]))

TIMER.start("Calculating implied vol (Black-76)...")
data_spx['implied_vol'] = implied_vol_b76(data_spx['pc'],
                                          data_spx['t_to_exp'],
                                          data_spx['strike'],
                                          data_spx['forward'],
                                          data_spx['rate'],
                                          data_spx['mid'])
TIMER.stop('{:,} rows'.format(data_spx.shape[0]))

# Filter out negative implied volatilities
TIMER.start("Filtering out negative implied volatilities...")
data_spx = data_spx[data_spx['implied_vol'] >= 0]
TIMER.stop("{:,} rows".format(data_spx.shape[0]))

TIMER.start("Calculating delta (Black-76)...")
data_spx['delta'] = delta_b76(data_spx['pc'],
                              data_spx['t_to_exp'],
                              data_spx['strike'],
                              data_spx['forward'],
                              data_spx['rate'],
                              data_spx['implied_vol'])
TIMER.stop('{:,} rows'.format(data_spx.shape[0]))

# Find 30 delta series only
data_spx_calls = data_spx[data_spx['pc'] == True].drop('pc', axis=1)
data_spx_puts = data_spx[data_spx['pc'] == False].drop('pc', axis=1)
TIMER.start()
delta_calls = lookup_val_in_col(data_spx_calls, 0.3, 'delta', groupby_cols=['quote_date', 'expiry'])
delta_puts = lookup_val_in_col(data_spx_puts, -0.3, 'delta', groupby_cols=['quote_date', 'expiry'])
TIMER.stop()

# Export
delta_calls.loc['2010-05-17':].to_csv('spx_options_calls_30_delta.csv')
delta_puts.loc['2010-05-17':].to_csv('spx_options_puts_30_delta.csv')

########################################################################

ymd_tuples = no_semiannual.apply(lambda row:
                                 get_whole_year_month_day_difference(delivery_month, row['maturityDate']), axis=1)
ymd_df = pd.DataFrame(ymd_tuples.tolist(), index=ymd_tuples.index,
                      columns=['remainingMaturityYears', 'remainingMaturityMonths', 'remainingMaturityDays'])
if tenor in [10, 30]:
    ymd_df['remainingMaturityMonthsModified'] = ymd_df['remainingMaturityMonths']//3*3
else:
    ymd_df['remainingMaturityMonthsModified'] = ymd_df['remainingMaturityMonths']
ymd_df['remainingMaturity'] = ymd_df['remainingMaturityYears'] + ymd_df['remainingMaturityMonthsModified']/12
no_semiannual = pd.concat([no_semiannual, ymd_df], axis=1)

import feedparser
treasury_rates_xml_url = 'https://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData'
feed = feedparser.parse(treasury_rates_xml_url)
latest_entry = feed.entries[-1]['m_properties'].split()
latest_date = pd.Timestamp(latest_entry[1])

legacy_url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Datasets/yield.xml'


# Basis Point Index vs. Realized and Difference
bp_list = [tyvix_bp, jgbvix_bp, srvix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]
for obj, color in zip(bp_list, color_list):
    if obj.name == 'SRVIX':
        _, axs = plt.subplots(2, 1, sharex='all')
        [joined_index, joined_undl_rv] = \
            share_dateindex([obj.price(), obj.undl_realized_vol(do_shift=True, window=252, bps=True)])
        make_lineplot([joined_index, joined_undl_rv], [obj.name, 'Realized Volatility'],
                      ylabel='Volatility (bps)',
                      title='{} with Realized ({} Days Shifted)'.format(obj.name, 252), ax=axs[0])
        difference = joined_index - joined_undl_rv
        # make_fillbetween(difference.index, joined_index, joined_undl_rv,
        #                  label='Difference', color='mediumseagreen', ax=axs[0])
        make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
        make_lineplot([difference], color_list=['g'], ax=axs[1])
    else:
        if obj.name in [vixig_bp.name, vixhy_bp.name, vixie_bp.name, vixxo_bp.name, vixfs_bp.name]:
            in_bps = True
        else:
            in_bps = False
        _, axs = plt.subplots(2, 1, sharex='all')
        [joined_index, joined_undl_rv] = \
            share_dateindex([obj.price(), obj.undl_realized_vol(do_shift=True, bps=True, price_in_bps=in_bps)])
        make_lineplot([joined_index, joined_undl_rv], [obj.name, 'Realized Volatility'],
                      ylabel='Volatility (bps)',
                      title='{} with Realized ({} Days Shifted)'.format(obj.name, 21), ax=axs[0])
        difference = joined_index - joined_undl_rv
        # make_fillbetween(difference.index, joined_index, joined_undl_rv,
        #                  label='Difference', color='mediumseagreen', ax=axs[0])
        make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
        make_lineplot([difference], color_list=['g'], ax=axs[1])

####

bp_list = [tyvix_bp, jgbvix_bp, srvix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]
color_list = ['C2', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

# Basis Point Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([bp_list[0].price(), bp_list[0].undl_realized_vol(do_shift=True, bps=True)])
bp_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(bp_diff, hist=False,
               label=bp_list[0].name, xlabel='Implied Vol Premium (bps)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line=color_list[0])
for o, color in zip(bp_list[1:], color_list[1:]):
    if o.name == 'SRVIX':
        [joined_index, joined_undl_rv] = \
            share_dateindex([o.price(), o.undl_realized_vol(do_shift=True, window=252, bps=True)])
        bp_diff = joined_index - joined_undl_rv
        make_histogram(bp_diff, hist=False,
                       label=o.name, xlabel='Implied Vol Premium (bps)', ylabel='Probability',
                       title='Risk Premium Distribution',
                       color_line=color, ax=ax_prem)
    else:
        if o.name in [vixig_bp.name, vixhy_bp.name, vixie_bp.name, vixxo_bp.name, vixfs_bp.name]:
            in_bps = True
        else:
            in_bps = False
        [joined_index, joined_undl_rv] = \
            share_dateindex([o.price(), o.undl_realized_vol(do_shift=True, bps=True, price_in_bps=in_bps)])
        bp_diff = joined_index - joined_undl_rv
        make_histogram(bp_diff, hist=False,
                       label=o.name, xlabel='Implied Vol Premium (bps)', ylabel='Probability',
                       title='Risk Premium Distribution',
                       color_line=color, ax=ax_prem)

cvix_list = [vixig, vixhy, vixie, vixxo, vixfs]
cvix_color_list = ['C4', 'C5', 'C6', 'C7', 'C8']

# Re-colored Credit VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
pc_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(pc_diff, hist=False,
                            label='VIX', xlabel='Implied Vol Premium (%)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
for o, color in zip(cvix_list[1:], cvix_color_list[1:]):
    [joined_index, joined_undl_rv] = \
        share_dateindex([o.price(), 100*o.undl_realized_vol(do_shift=True)])
    pc_diff = joined_index - joined_undl_rv
    make_histogram(pc_diff, hist=False,
                   label=o.name, xlabel='Implied Vol Premium (%)', ylabel='Probability',
                   title='Risk Premium Distribution',
                   color_line=color, ax=ax_prem)

irvix_list = [vix, tyvix, jgbvix, srvix]
irvix_color_list = ['C0', 'C2', 'C1', 'C3']

# Re-colored Interest Rate VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
pc_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(pc_diff, hist=False,
                            label='VIX', xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
for o, color in zip(irvix_list[1:], irvix_color_list[1:]):
    if o.name == 'SRVIX':
        [joined_index, joined_undl_rv] = \
            share_dateindex([o.price(), o.undl_realized_vol(do_shift=True, window=252, bps=True)])
        pc_diff = joined_index - joined_undl_rv
        make_histogram(pc_diff, hist=False,
                       label=o.name, xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
                       title='Risk Premium Distribution',
                       color_line=color, ax=ax_prem)
    else:
        [joined_index, joined_undl_rv] = \
            share_dateindex([o.price(), 100*o.undl_realized_vol(do_shift=True)])
        pc_diff = joined_index - joined_undl_rv
        make_histogram(pc_diff, hist=False,
                       label=o.name, xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
                       title='Risk Premium Distribution',
                       color_line=color, ax=ax_prem)

####

# Medians (with dates aligned)
bp_list = [tyvix_bp, jgbvix_bp, srvix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]
bp_dict = {name: median for name, median in zip(map(lambda o: o.name, bp_list), map(lambda o: o.price().loc['2012-06-18':].median(), bp_list))}
bp_median_table = pd.DataFrame(bp_dict, index=['Median Level (Since 2012-06-18)']).T

pc_list = [vix, tyvix, jgbvix, vixig, vixhy, vixie, vixxo, vixfs]
pc_dict = {name: median for name, median in zip(map(lambda o: o.name, pc_list), map(lambda o: o.price().loc['2012-06-18':].median(), pc_list))}
pc_median_table = pd.DataFrame(pc_dict, index=['Median Level (Since 2012-06-18)']).T

####

bp_list = [tyvix_bp, jgbvix_bp, srvix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]
bp_dict = {name: median for name, median in zip(map(lambda o: o.name, bp_list), map(lambda o: o.price().median(), bp_list))}
bp_median_table = pd.DataFrame(bp_dict, index=['Median Level']).T

fig, ax = plt.subplots()
ax.plot(tyvix_bp.price(), label='BP TYVIX', color='C2')
ax.plot(jgbvix_bp.price(), label='BP JGB VIX', color='C1')
ax.plot(srvix.price(), label='SRVIX', color='C3')
ax.plot(vixig_bp.price().loc[:'2018-04-25'], label='BP VIXIG', color='C4')
ax.plot(vixhy_bp.price().loc[:'2018-04-25'], label='BP VIXHY', color='C5')
ax.plot(vixie_bp.price().loc[:'2018-04-25'], label='BP VIXIE', color='C6')
ax.plot(vixxo_bp.price().loc[:'2018-04-25'], label='BP VIXXO', color='C7')
ax.plot(vixfs_bp.price().loc[:'2018-04-25'], label='BP VIXFS', color='C8')
ax.legend(fontsize=13)
ax.plot(vixig_bp.price().loc['2018-08-23':], label='BP VIXIG', color='C4')
ax.plot(vixhy_bp.price().loc['2018-08-23':], label='BP VIXHY', color='C5')
ax.plot(vixie_bp.price().loc['2018-08-23':], label='BP VIXIE', color='C6')
ax.plot(vixxo_bp.price().loc['2018-08-23':], label='BP VIXXO', color='C7')
ax.plot(vixfs_bp.price().loc['2018-08-23':], label='BP VIXFS', color='C8')
ax.set_ylabel('Volatility Index (bps)', fontsize=16)
ax.set_title('All Basis Point Volatility Indexes', fontsize=16)

fig, ax = plt.subplots()
ax.plot(tyvix_bp.price(), label='BP TYVIX', color='C2')
ax.plot(jgbvix_bp.price(), label='BP JGB VIX', color='C1')
ax.plot(srvix.price(), label='SRVIX', color='C3')
ax.legend()
ax.set_ylabel('Volatility Index (bps)')
ax.set_title('Interest Rate Volatility Indexes (Basis Point Versions)')

fig, ax = plt.subplots()
ax.plot(vixig_bp.price().loc[:'2018-04-25'], label='BP VIXIG', color='C0')
ax.plot(vixhy_bp.price().loc[:'2018-04-25'], label='BP VIXHY', color='C1')
ax.plot(vixie_bp.price().loc[:'2018-04-25'], label='BP VIXIE', color='C2')
ax.plot(vixxo_bp.price().loc[:'2018-04-25'], label='BP VIXXO', color='C3')
ax.plot(vixfs_bp.price().loc[:'2018-04-25'], label='BP VIXFS', color='C4')
ax.legend(fontsize=13)
ax.plot(vixig_bp.price().loc['2018-08-23':], label='BP VIXIG', color='C0')
ax.plot(vixhy_bp.price().loc['2018-08-23':], label='BP VIXHY', color='C1')
ax.plot(vixie_bp.price().loc['2018-08-23':], label='BP VIXIE', color='C2')
ax.plot(vixxo_bp.price().loc['2018-08-23':], label='BP VIXXO', color='C3')
ax.plot(vixfs_bp.price().loc['2018-08-23':], label='BP VIXFS', color='C4')
ax.set_ylabel('Volatility Index (bps)', fontsize=16)
ax.set_title('Credit Volatility Indexes (Basis Point Versions)', fontsize=16)

# [Figure 4] VIXs in Rates Group with VIX Index
start_date = tyvix.price().index[0]
[truncd_tyvix, truncd_jgbvix, truncd_srvix, truncd_vixig, truncd_vixhy, truncd_vixie, truncd_vixxo, truncd_vixfs] = \
    map(lambda p: p.truncate(start_date), [tyvix_bp.price(), jgbvix_bp.price(), srvix.price(),
                                           vixig_bp.price(), vixhy_bp.price(), vixie_bp.price(), vixxo_bp.price(), vixfs_bp.price()])
_, axleft = plt.subplots()
axleft.plot(truncd_vix, label='S&P500 VIX')
axleft.plot(truncd_tyvix, label='TYVIX')
axleft.plot(truncd_jgbvix, label='JGB VIX')
axleft.legend(loc=2)
axleft.set_ylabel('Volatility Index (% Price)')
axleft.set_title('VIXs in Rates Group with VIX Index')
axright = axleft.twinx()
axright.plot(truncd_srvix, label='SRVIX', color='C3')
axright.legend(loc=1)
axright.set_ylabel('Volatility Index (SRVIX) (bps)')
axright.set_ylim(20, 115)

# [Figure 16, 17, 18, 19] Interest Rate VIX Difference Charts
_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['S&P500 VIX', 'Realized Volatility'],
              ylabel='Volatility (% Price)',
              title='S&P500 VIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])
axs[0].set_ylabel('Volatility (% Price)', fontsize=16)
axs[0].set_title('S&P500 VIX with Realized (21 Days Shifted)', fontsize=16)
axs[0].legend(fontsize=14)
axs[1].legend(fontsize=14)

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([tyvix.price(), 100*tyvix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['TYVIX', 'Realized Volatility'],
              ylabel='Volatility (% Price)',
              title='TYVIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])
axs[0].set_ylabel('Volatility (% Price)', fontsize=16)
axs[0].set_title('TYVIX with Realized (21 Days Shifted)', fontsize=16)
axs[0].legend(fontsize=14)
axs[1].legend(fontsize=14)

# [Figure 15] Credit VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
vix_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(vix_diff, hist=False,
                            label='VIX', xlabel='Implied Vol Premium (%)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixig.price(), 100*vixig.undl_realized_vol(do_shift=True)])
vixig_diff = joined_index - joined_undl_rv
make_histogram(vixig_diff, hist=False,
               label='VIXIG', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C1', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixhy.price(), 100*vixhy.undl_realized_vol(do_shift=True)])
vixhy_diff = joined_index - joined_undl_rv
make_histogram(vixhy_diff, hist=False,
               label='VIXHY', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C2', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixie.price(), 100*vixie.undl_realized_vol(do_shift=True)])
vixie_diff = joined_index - joined_undl_rv
make_histogram(vixie_diff, hist=False,
               label='VIXIE', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C3', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixxo.price(), 100*vixxo.undl_realized_vol(do_shift=True)])
vixxo_diff = joined_index - joined_undl_rv
make_histogram(vixxo_diff, hist=False,
               label='VIXXO', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C4', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixfs.price(), 100*vixfs.undl_realized_vol(do_shift=True)])
vixfs_diff = joined_index - joined_undl_rv
make_histogram(vixfs_diff, hist=False,
               label='VIXFS', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C5', ax=ax_prem)

####

treasury_vix_data = pd.read_csv('data/cme_eod_treasury_vix.csv', index_col='Date', parse_dates=True)

####

def get_regime_data_list(intervals, data):
    return [data.loc[start:end] for start, end in intervals]

def combine_data_list(data_list):
    acc = pd.Series()
    for data in data_list:
        acc = acc.append(data.dropna())
    return acc.sort_index()

tyvix_hl, tyvix_lows, tyvix_highs = tyvix.vol_regime()
vix_hl, vix_lows, vix_highs = vix.vol_regime()

vix_tyvix_hl = pd.DataFrame({'vix': vix_hl, 'tyvix': tyvix_hl}).dropna()
high_high_days = vix_tyvix_hl[(vix_tyvix_hl['vix']=='high') & (vix_tyvix_hl['tyvix']=='high')].index
low_low_days = vix_tyvix_hl[(vix_tyvix_hl['vix']=='low') & (vix_tyvix_hl['tyvix']=='low')].index
diffs_high = test['10YR - 2YR Yield'].diff().loc[high_high_days]
diffs_low = test['10YR - 2YR Yield'].diff().loc[low_low_days]

ten_two_list_low = get_regime_data_list(tyvix_lows, test['10YR - 2YR Yield'])
combined_ten_two_low = combine_data_list(map(lambda df: df.diff(), ten_two_list_low))
ten_two_list_high = get_regime_data_list(tyvix_highs, test['10YR - 2YR Yield'])
combined_ten_two_high = combine_data_list(map(lambda df: df.diff(), ten_two_list_high))
ttest_ind(combined_ten_two_low.loc['2014':], combined_ten_two_high.loc['2014':])

ten_two_list_low = get_regime_data_list(tyvix_lows, tenyr)
combined_ten_two_low = combine_data_list(map(lambda df: df.diff(), ten_two_list_low))
ten_two_list_high = get_regime_data_list(tyvix_highs, tenyr)
combined_ten_two_high = combine_data_list(map(lambda df: df.diff(), ten_two_list_high))
ttest_ind(combined_ten_two_low.loc['2014':], combined_ten_two_high.loc['2014':])

####

test = pd.read_csv('Y:/Research/Research1/Gaibo/S&P Webinar Figures/3 - predictive signaling raw data.csv', index_col='Date', parse_dates=True)
_, axleft = plt.subplots()
make_lineplot(tyvix.price(), 'TYVIX', ax=axleft)
make_regime(tyvix.vol_regime()[2], 'High Vol Regime', 'grey', 'Date', 'Index Level', 'TYVIX Vol Regimes', ax=axleft)
make_regime(tyvix.vol_regime()[1], 'Low Vol Regime', 'white', 'Date', 'Index Level', 'TYVIX Vol Regimes', ax=axleft)
# axleft.autoscale(enable=True, axis='x', tight=True)
axleft.legend(loc=2, fontsize=13)
axleft.set_ylabel('Volatility Index (%)', fontsize=16)
axright = axleft.twinx()
axright.plot(test['10YR - 2YR Yield'], label='10yr - 2yr Yield', color='C1')
axright.legend(loc=1, fontsize=13)
axright.set_ylabel('% (Annualized)', fontsize=16)
axright.set_xlim('2016-01-01', '2019-09-03')
# axleft.autoscale(enable=True, axis='x', tight=True)
# axright.set_ylim(20, 115)
axleft.set_title('TYVIX Vol Regimes', fontsize=16)
axleft.set_xlabel('Date', fontsize=16)

####

cdx_ig_old = bbg_data['IBOXUMAE CBBT Curncy', 'PX_LAST'].dropna()
itraxx_ie_old = bbg_data['ITRXEBE CBBT Curncy', 'PX_LAST'].dropna()
itraxx_xo_old = bbg_data['ITRXEXE CBBT Curncy', 'PX_LAST'].dropna()
itraxx_fs_old = bbg_data['ITRXESE CBBT Curncy', 'PX_LAST'].dropna()
cdx_ig_new = scaled_cds_index_data['CDX NA IG'].dropna()
itraxx_ie_new = scaled_cds_index_data['iTraxx EU Main'].dropna()
itraxx_xo_new = scaled_cds_index_data['iTraxx EU Xover'].dropna()
itraxx_fs_new = scaled_cds_index_data['iTraxx EU SenFin'].dropna()

plt.subplots()
cdx_ig_old.plot()
cdx_ig_new.plot()
plt.subplots()
itraxx_ie_old.plot()
itraxx_ie_new.plot()
plt.subplots()
itraxx_xo_old.plot()
itraxx_xo_new.plot()
plt.subplots()
itraxx_fs_old.plot()
itraxx_fs_new.plot()

##############################################################################

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(vega_volumes_df['Vega Volume'], label='Vega Volume using Given IVs', linewidth=4)
ax.plot(vega_volumes_df['Vega Volume using Backed-Out IV'], label='Vega Volume using Backed-Out IVs')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Vega Volume (MM)')
ax.set_title('{} Options Daily Vega Volume'.format(NOTE_CODE))

#####################################

fig, ax = plt.subplots()
make_lineplot(vix.price(), 'VIX', ax=ax)
make_regime(vix.vol_regime()[2], 'High Vol Regime', 'r', 'Date', 'Index Level', 'VIX Vol Regimes', ax=ax)
make_regime(vix.vol_regime()[1], 'Low Vol Regime', 'g', 'Date', 'Index Level', 'VIX Vol Regimes', ax=ax)

plt.figure()
prices = vix.price()
prices.plot()
window = 126
low_threshold = 0.1
high_threshold = 0.9
rolling_low = prices.rolling(window).quantile(low_threshold).dropna()
rolling_high = prices.rolling(window).quantile(high_threshold).dropna()
rolling_low.plot()
rolling_high.plot()
regime, low_list, high_list = vix.vol_regime()
for interval in low_list:
    plt.axvspan(*interval, color='C1', alpha=0.5)
for interval in high_list:
    plt.axvspan(*interval, color='C2', alpha=0.5)

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

#%% Shape up the data for use

###############################################################################
### [DAILY] Premiums and volumes
## Obtain sorted index by option ticker and no extra columns
# Transpose
prem_vol_df_clean_T = prem_vol_df_clean.T
# Reset index
prem_vol_df_clean_T_noindex = prem_vol_df_clean_T.reset_index()
# Remove column name to reorganize
prem_vol_df_clean_T_noindex.columns = prem_vol_df_clean_T_noindex.columns.rename(None)
# Rename, reindex, and sort
prem_vol_df_clean_T_reindex = \
    prem_vol_df_clean_T_noindex.rename({'ticker':'opt_ticker'}, axis=1) \
                               .set_index('opt_ticker') \
                               .sort_index()

## Option ticker-indexed daily volumes
volume_df_indexed = \
    prem_vol_df_clean_T_reindex[prem_vol_df_clean_T_reindex['field']=='PX_VOLUME'] \
    .drop(columns='field')
volume_df_indexed = volume_df_indexed.fillna(value=0)   # Replace NaN with 0
## Option ticker-indexed daily premiums
prem_df_indexed = \
    prem_vol_df_clean_T_reindex[prem_vol_df_clean_T_reindex['field']=='PX_LAST'] \
    .drop(columns='field')
## Option ticker-index that returned usable information
chosen_index = volume_df_indexed.index
################################################################################

################################################################################
### [DAILY] Underlying prices
## Clean underlying ticker mapping by removing and renaming columns
undl_ticker_df_clean = \
    undl_ticker_df.drop(columns='field') \
                  .rename({'ticker':'opt_ticker', 'value':'undl_ticker'}, axis=1)
## Obtain price DF to merge into undl_ticker_df_clean
## NOTE: ensure reset_index() is always used as intended
# Make copy to retain clean original BBG pull
undl_price_df_copy = undl_price_df.copy()
# Remove unnecessary multiindex
undl_price_df_copy.columns = undl_price_df.columns.get_level_values(0).rename(None)
# Transpose
undl_price_df_T = undl_price_df_copy.T
# Remove column name to reorganize
undl_price_df_T.columns = undl_price_df_T.columns.rename(None)
# Rename index
undl_price_df_T.index = undl_price_df_T.index.rename('undl_ticker')
# Match formatting of undl_ticker_df_clean by resetting index
undl_price_df_clean = undl_price_df_T.reset_index()

## Option ticker-indexed daily underlying prices (from merging)
undl_price_df_indexed = \
    undl_ticker_df_clean.merge(undl_price_df_clean) \
    .drop(columns='undl_ticker') \
    .set_index('opt_ticker') \
    .sort_index()
################################################################################

################################################################################
### Expiration date, financing rate, and strike
## Clean by indexing by option ticker and isolating the two types of data
# Rename and index
exp_strike_rate_df_indexed = exp_rate_df.rename({'ticker':'opt_ticker'}, axis=1) \
                                               .set_index('opt_ticker')
# Isolate expiration date and financing rate
exp_df_indexed = \
    exp_strike_rate_df_indexed.loc[exp_strike_rate_df_indexed['field']=='OPT_EXPIRE_DT',
                                   'value']
rate_df_indexed = \
    exp_strike_rate_df_indexed.loc[exp_strike_rate_df_indexed['field']=='OPT_FINANCE_RT',
                                   'value'].astype(float) * 0.01    # Scale
# Create strike in same format
strike_df_indexed = \
    pd.Series(exp_df_indexed.index.map(lambda s: float(s.split(' ')[1])),
              index=exp_df_indexed.index)
# Create put or call indication in same format
pc_df_indexed = \
    pd.Series(exp_df_indexed.index.map(lambda s: s.split(' ')[0][-1]),
              index=exp_df_indexed.index)

## Option ticker-indexed expiration date, strike, and financing rate
constant_df_indexed = pd.DataFrame(dict(expiration=exp_df_indexed,
                                        strike=strike_df_indexed,
                                        putcall=pc_df_indexed,
                                        rate=rate_df_indexed)).sort_index()
################################################################################
