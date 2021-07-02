import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET
import matplotlib.pyplot as plt
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

LEGACY_DATA_DIR = 'Y:/Research/Research1/Parth/Fudon Project/Input Data/Raw Data (SAS Pulled)/Futures Cleared Trades/'
LEGACY_DATA_FILE = '{} Cleared Trades.csv'   # '23-Feb-18 Cleared Trades.csv'
MODERN_DATA_DIR = 'Y:/Research/Research1/Parth/Fudon Project/Input Data/Raw Data (Tableau Pulled)/Tableau Raw Data Contra Futures/VX/'
MODERN_DATA_FILE = '{} VX Contra Trade Data.csv'    # '20180320 VX Contra Trade Data.csv'

# Important dates
LEGACY_START, LEGACY_END = pd.Timestamp('2009-01-02'), pd.Timestamp('2018-02-23')
legacy_days = pd.date_range(start='2012-01-03', end=LEGACY_END, freq=BUSDAY_OFFSET)     # 2012 is what we'll mark as "reasonable VIX era"
MODERN_START, MODERN_END = pd.Timestamp('2018-03-20'), pd.Timestamp('2020-12-17')
modern_days = pd.date_range(start=MODERN_START, end=MODERN_END, freq=BUSDAY_OFFSET)
FIRST_DAY_3PM_SETTLEMENT = pd.Timestamp('2020-10-26')
FIRST_DAY_MINI_VIX = pd.Timestamp('2020-08-10')

#### Load "legacy" SAS data tables

LEGACY_USECOLS = ['Entry Time', 'Date', 'Expiry Date', 'Class', 'Trade Size', 'Trade Price']   # Class 'VXT' is TAS

legacy_data_list = []
for day in legacy_days:
    day_data = pd.read_csv(LEGACY_DATA_DIR + LEGACY_DATA_FILE.format(day.strftime('%d-%b-%y')),
                           usecols=LEGACY_USECOLS,
                           parse_dates=['Date', 'Entry Time', 'Expiry Date'])
    legacy_data_list.append(day_data)
    print(day.strftime('%Y-%m-%d'), "done")

# Clean into mega DataFrame
legacy_data = pd.concat(legacy_data_list).set_index('Entry Time').sort_index()
legacy_vix = legacy_data[legacy_data['Class'] == 'VX'].copy()   # Live, no-TAS
legacy_tas = legacy_data[legacy_data['Class'] == 'VXT'].copy()

#### Load crude data pulled from production table for 1 month of disconnect - 2018-02-23 to 2018-03-20

# There are 2 transact_time columns, and one of them needs to be hard-corrected to transact_date
missing_data = pd.read_csv(DOWNLOADS_DIR + 'futures_trades_missing_final.csv')
missing_data = missing_data.rename({'transact_time': 'transact_date', 'transact_time.1': 'transact_time'}, axis=1)
# Isolate VX from other roots
missing_data = missing_data[missing_data['futures_root'] == 'VX']
# Manual conversion to Timestamp
missing_data['transact_date'] = pd.to_datetime(missing_data['transact_date'])
missing_data['expire_date'] = pd.to_datetime(missing_data['expire_date'])
missing_data['transact_time_no_tz'] = pd.to_datetime(missing_data['transact_time'].apply(lambda s: s[:-3]))     # Remove the UTC offset - it's just daylight savings I think

MISSING_USECOLS = ['transact_time_no_tz', 'transact_date', 'expire_date', 'futures_root', 'size', 'price', 'tas']   # tas field 'Y' is TAS

# Clean into mega DataFrame
missing_data = missing_data.set_index('transact_time_no_tz').sort_index()
missing_vix = missing_data[missing_data['tas'] == 'N'].copy()   # Live, no-TAS
missing_tas = missing_data[missing_data['tas'] == 'Y'].copy()

#### Load "modern" Tableau data tables

MODERN_USECOLS = ['Trade Time', 'Trade Date', 'Expire Date', 'Class', 'Size', 'Price', 'Product Type', 'Bid Price', 'Ask Price']   # Product Type 'T' is TAS

modern_data_list = []
for day in modern_days:
    day_data = pd.read_csv(MODERN_DATA_DIR + MODERN_DATA_FILE.format(day.strftime('%Y%m%d')),
                           usecols=MODERN_USECOLS,
                           parse_dates=['Trade Date', 'Trade Time', 'Expire Date'])
    modern_data_list.append(day_data)
    print(day.strftime('%Y-%m-%d'), "done")

# Clean into mega DataFrame
modern_data = pd.concat(modern_data_list).set_index('Trade Time').sort_index()
modern_vix = modern_data[modern_data['Product Type'] == 'S'].copy()     # Live, no-TAS
modern_tas = modern_data[modern_data['Product Type'] == 'T'].copy()

#### Combine all 3 data sources

# # Observe that legacy data only shows one side of a contra (volume, i.e. trade size/2)
# test_1 = legacy_vix.between_time('14:45', '15:00').loc['2018-02-23']
# test_2 = missing_vix.between_time('14:45', '15:00').loc['2018-02-23']
# test_1 = test_1[['Expiry Date', 'Trade Size', 'Trade Price']]
# test_2 = test_2[['expire_date', 'size', 'price']]
# test_1 = test_1.reset_index().set_index(['Entry Time', 'Expiry Date']).sort_index()
# test_2 = test_2.reset_index().set_index(['transact_time_no_tz', 'expire_date']).sort_index()

# Rename columns and cut out extra info
# Want to end up with: 'Trade Time', 'Trade Date', 'Expire Date', 'Class', 'Size', 'Price' (6 fields, as named in modern Tableau)
legacy_rename = \
    (legacy_vix
     .rename({'Entry Time': 'Trade Time', 'Date': 'Trade Date', 'Expiry Date': 'Expire Date',
              'Class': 'Class', 'Trade Size': 'Size', 'Trade Price': 'Price'}, axis=1))
legacy_rename.index.name = 'Trade Time'     # From 'Entry Time'
missing_rename = \
    (missing_vix
     .rename({'transact_time_no_tz': 'Trade Time', 'transact_date': 'Trade Date', 'expire_date': 'Expire Date',
              'futures_root': 'Class', 'size': 'Size', 'price': 'Price'}, axis=1))
missing_rename.index.name = 'Trade Time'     # From 'transact_time_no_tz'
missing_rename = missing_rename.drop(['transact_time', 'tas'], axis=1)
modern_rename = modern_vix.drop('Product Type', axis=1)

# Make legacy trade sizes consistent with the others
legacy_rename['Size'] *= 2

# Combine
mega_df = pd.concat([legacy_rename.between_time('14:45', '15:15'),
                     missing_rename.between_time('14:45', '15:15').loc['2018-02-24':'2018-03-19 16:00:01'],
                     modern_rename.between_time('14:45', '15:15')], sort=False)
mega_df = mega_df[['Trade Date', 'Expire Date', 'Size', 'Price', 'Bid Price', 'Ask Price']]   # Useful

# Ensure fact that each trade date spans 2 days does not throw index date-time off from "trade date" column
idx_date = pd.to_datetime(pd.Series(mega_df.index.date, index=mega_df.index))
assert mega_df[(idx_date != mega_df['Trade Date'])].empty

#### Load new Mini-VIX data

MINI_DATA_DIR = 'Y:/Research/Research1/Parth/Fudon Project/Input Data/Raw Data (Tableau Pulled)/Tableau Raw Data Contra Futures/VXM/'
MINI_DATA_FILE = '{} VXM Contra Trade Data.csv'    # '201808 VXM Contra Trade Data.csv'
mini_data_list = []
for yearmonth_str in pd.date_range(FIRST_DAY_MINI_VIX, MODERN_END).strftime('%Y%m').unique():
    month_data = pd.read_csv(MINI_DATA_DIR + MINI_DATA_FILE.format(yearmonth_str),
                             usecols=MODERN_USECOLS,
                             parse_dates=['Trade Date', 'Trade Time', 'Expire Date'])   # Reuses MODERN_USECOLS
    mini_data_list.append(month_data)
    print(yearmonth_str, "done")
mini_data = pd.concat(mini_data_list).set_index('Trade Time').sort_index()
mini_vix = mini_data[mini_data['Product Type'] == 'S'].copy()   # Live, no-TAS
mini_tas = mini_data[mini_data['Product Type'] == 'T'].copy()

mini_df = mini_vix[['Trade Date', 'Expire Date', 'Size', 'Price', 'Bid Price', 'Ask Price']].between_time('14:45', '15:15')   # Useful

# Ensure fact that each trade date spans 2 days does not throw index date-time off from "trade date" column
idx_date = pd.to_datetime(pd.Series(mini_df.index.date, index=mini_df.index))
assert mini_df[(idx_date != mini_df['Trade Date'])].empty

#### Using knowledge from the future, I know there are weeklies included, so let's filter those out now

from options_futures_expirations_v3 import generate_expiries
vix_monthly_maturities = generate_expiries('2012-01-03', MODERN_END+pd.DateOffset(months=12), specific_product='VIX')
mega_weeklies = pd.Series(list(set(pd.Series(mega_df['Expire Date'].unique())) - set(vix_monthly_maturities))).sort_values()
# Also knowledge from the future, on trade date 2013-07-02, they have 2014-03-19 marked as maturity instead of 2014-03-18. On every other day, they use the correct Good Friday maturity, but for some reason, 7/2 is glitched.
mega_df.replace(pd.Timestamp('2014-03-19'), {'Expire Date': pd.Timestamp('2014-03-18')}, inplace=True)  # inplace is needed because not enough RAM lmao
mega_weeklies = mega_weeklies[mega_weeklies != pd.Timestamp('2014-03-19')].reset_index(drop=True)   # 2014-03-19 is a Good Friday mistake; should be 2014-03-18
mega_weeklies_df = mega_df[mega_df['Expire Date'].isin(mega_weeklies)].copy()
mega_df = mega_df[~mega_df['Expire Date'].isin(mega_weeklies)]  # Weeklies are taken out!

# Now do same with Minis
mini_weeklies_df = mini_df[mini_df['Expire Date'].isin(mega_weeklies)].copy()
mini_df = mini_df[~mini_df['Expire Date'].isin(mega_weeklies)]  # Weeklies are taken out!

###############################################################################

#### Project: Near-Settlement Data for Stuart Barton

# # NOTE: evaluating 3 sources separately because unsure of consistency
#
# # Legacy
# legacy_vix_245_300 = legacy_vix.between_time('14:45', '15:00')
# legacy_vix_300_315 = legacy_vix.between_time('15:00', '15:15')
# legacy_result_245_300 = legacy_vix_245_300.groupby('Date')['Trade Size'].sum()
# legacy_result_300_315 = legacy_vix_300_315.groupby('Date')['Trade Size'].sum()
#
# # Missing Transition
# missing_vix_245_300 = missing_vix.between_time('14:45', '15:00')
# missing_vix_300_315 = missing_vix.between_time('15:00', '15:15')
# missing_result_245_300 = missing_vix_245_300.groupby('transact_date')['size'].sum()
# missing_result_300_315 = missing_vix_300_315.groupby('transact_date')['size'].sum()
#
# # Modern
# modern_vix_245_300 = modern_vix.between_time('14:45', '15:00')
# modern_vix_300_315 = modern_vix.between_time('15:00', '15:15')
# modern_result_245_300 = modern_vix_245_300.groupby('Trade Date')['Size'].sum()
# modern_result_300_315 = modern_vix_300_315.groupby('Trade Date')['Size'].sum()

mega_245_300 = mega_df.between_time('14:45', '15:00')
mega_300_315 = mega_df.between_time('15:00', '15:15')
result_245 = mega_245_300.groupby('Trade Date')['Size'].sum()
result_300 = mega_300_315.groupby('Trade Date')['Size'].sum()
result_df = pd.DataFrame({'Volume 2:45pm-3pm': result_245, 'Volume 3pm-3:15pm': result_300})    # Automatically aligns index
# result_df.to_csv(DOWNLOADS_DIR + 'volume_near_settlement_REVISED.csv')

###############################################################################

#### Project: Near-Settlement Data for Stuart Barton Electric Boogaloo with TAS

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

# Filter out weeklies in TAS
mega_tas_weeklies_df = mega_tas_df[mega_tas_df['Expire Date'].isin(mega_weeklies)].copy()
mega_tas_df = mega_tas_df[~mega_tas_df['Expire Date'].isin(mega_weeklies)]  # Weeklies are taken out!

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
# result_tas_df.to_csv(DOWNLOADS_DIR + 'tas_volume_near_settlement.csv')

###############################################################################

# #### Troubleshoot mismatch
#
# # Get first and last time for each date
# # Modern
# modern_open_close_tuples_list = []
# for day in modern_days:
#     idx = modern_vix[modern_vix['Trade Date'] == day].index
#     modern_open_close_tuples_list.append((day, idx[0], idx[-1]))
# modern_oc_df = pd.DataFrame(modern_open_close_tuples_list,
#                             columns=['Trade Date', 'open', 'close'])    # Useful
# modern_subtract_df = modern_oc_df.subtract(modern_oc_df['Trade Date'], axis=0)
# modern_subtract_df['Trade Date'] = modern_oc_df['Trade Date']
# modern_subtract_df.set_index('Trade Date', inplace=True)    # Useful
# # Plot
# modern_subtract_df.plot()  # Oh God, every time there's a holiday, trading ends at 10:30am and it gets marked with the next business day as "Trade Date" - weird pattern
# (modern_vix[modern_vix['Trade Date'] == pd.Timestamp('2020-07-06')]
#  .reset_index()[['Trade Time', 'Trade Date']].plot())   # Uniquely bad date where July 4 is observed on the 3rd, trading ends at 10:30am, and there's a weekend after, and "Trade Date" is marked as 7/6
# # Missing
# missing_open_close_tuples_list = []
# for day in missing_vix['transact_date'].unique():
#     idx = missing_vix[missing_vix['transact_date'] == day].index
#     missing_open_close_tuples_list.append((day, idx[0], idx[-1]))
# missing_oc_df = pd.DataFrame(missing_open_close_tuples_list,
#                              columns=['transact_date', 'open', 'close'])    # Useful
# missing_subtract_df = missing_oc_df.subtract(missing_oc_df['transact_date'], axis=0)
# missing_subtract_df['transact_date'] = missing_oc_df['transact_date']
# missing_subtract_df.set_index('transact_date', inplace=True)   # Useful
# # Plot
# missing_subtract_df.plot()     # Very funky - regular pattern
# # Legacy
# legacy_open_close_tuples_list = []
# for day in legacy_days:
#     idx = legacy_vix[legacy_vix['Date'] == day].index
#     if idx.empty:
#         print(f"EMPTY {day.strftime('%Y-%m-%d')}")
#         continue
#     legacy_open_close_tuples_list.append((day, idx[0], idx[-1]))
# legacy_oc_df = pd.DataFrame(legacy_open_close_tuples_list,
#                             columns=['Date', 'open', 'close'])    # Useful
# legacy_subtract_df = legacy_oc_df.subtract(legacy_oc_df['Date'], axis=0)
# legacy_subtract_df['Date'] = legacy_oc_df['Date']
# legacy_subtract_df.set_index('Date', inplace=True)   # Useful
# # Plot
# legacy_subtract_df.plot()   # Lots of sawtoothing on open (really weird) with early close only on Thanksgiving
#
# # Specific overlap days analysis
# # Legacy
# legacy_overlap = legacy_vix[legacy_vix['Date'] == pd.Timestamp('2018-02-23')]
# missing_overlap_legacy = missing_vix[missing_vix['transact_date'] == pd.Timestamp('2018-02-23')]
# fig, axs = plt.subplots(2, 1, sharex='all', figsize=(19.2, 10.8))
# axs[0].set_title('Legacy (SAS)')
# axs[0].plot(legacy_overlap.between_time('14:00', '15:15')['Trade Price'], color='C0', label='Trade Price')
# ax0r = axs[0].twinx()
# ax0r.plot(legacy_overlap.between_time('14:00', '15:15')['Trade Size'], color='C1', label='Trade Size')
# axs[0].legend(loc=2)
# ax0r.legend(loc=1)
# axs[1].set_title('Missing (Production Tables)')
# axs[1].plot(missing_overlap_legacy.between_time('14:00', '15:15')['price'], color='C0', label='price')
# ax1r = axs[1].twinx()
# ax1r.plot(missing_overlap_legacy.between_time('14:00', '15:15')['size'], color='C1', label='size')
# axs[1].legend(loc=2)
# ax1r.legend(loc=1)
# fig.suptitle('2018-02-23 Comparison', size='x-large')
# # Modern
# modern_overlap = modern_vix[modern_vix['Trade Date'] == pd.Timestamp('2018-03-20')]
# missing_overlap_modern = missing_vix[missing_vix['transact_date'] == pd.Timestamp('2018-03-20')]
# fig, axs = plt.subplots(2, 1, sharex='all', figsize=(19.2, 10.8))
# axs[0].set_title('Modern (Tableau)')
# axs[0].plot(modern_overlap.between_time('14:00', '15:15')['Price'], color='C0', label='Price')
# ax0r = axs[0].twinx()
# ax0r.plot(modern_overlap.between_time('14:00', '15:15')['Size'], color='C1', label='Size')
# axs[0].legend(loc=2)
# ax0r.legend(loc=1)
# axs[1].set_title('Missing (Production Tables)')
# axs[1].plot(missing_overlap_modern.between_time('14:00', '15:15')['price'], color='C0', label='price')
# ax1r = axs[1].twinx()
# ax1r.plot(missing_overlap_modern.between_time('14:00', '15:15')['size'], color='C1', label='size')
# axs[1].legend(loc=2)
# ax1r.legend(loc=1)
# fig.suptitle('2018-03-20 Comparison', size='x-large')

# 2 overlap days - "missing overlap legacy" on 2018-02-23, "missing overlap modern" on 2018-03-20
legacy_overlap = legacy_rename.between_time('14:00', '15:15').loc['2018-02-23']
missing_overlap_legacy = missing_rename.between_time('14:00', '15:15').loc['2018-02-23']
legacy_overlap = legacy_overlap.reset_index().set_index(['Trade Time', 'Expire Date']).sort_index().reset_index('Expire Date')
missing_overlap_legacy = missing_overlap_legacy.reset_index().set_index(['Trade Time', 'Expire Date']).sort_index().reset_index('Expire Date')
modern_overlap = modern_rename.between_time('14:00', '15:15').loc['2018-03-20']
missing_overlap_modern = missing_rename.between_time('14:00', '15:15').loc['2018-03-20']
modern_overlap = modern_overlap.reset_index().set_index(['Trade Time', 'Expire Date']).sort_index().reset_index('Expire Date')
missing_overlap_modern = missing_overlap_modern.reset_index().set_index(['Trade Time', 'Expire Date']).sort_index().reset_index('Expire Date')

# Plot "missing overlap legacy", aka 2018-02-23
fig, axs = plt.subplots(2, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].set_title('Legacy (SAS)')
axs[0].plot(legacy_overlap['Price'], color='C0', label='Price')
ax0r = axs[0].twinx()
ax0r.plot(legacy_overlap['Size'], color='C1', label='Size')
# ax0r.plot(legacy_overlap.groupby('Trade Time')['Size'].sum(), color='C1', label='Size')
axs[0].legend(loc=2)
ax0r.legend(loc=1)
axs[1].set_title('Missing (Production Tables)')
axs[1].plot(missing_overlap_legacy['Price'], color='C0', label='Price')
ax1r = axs[1].twinx()
ax1r.plot(missing_overlap_legacy['Size'], color='C1', label='Size')
# ax1r.plot(missing_overlap_legacy.groupby('Trade Time')['Size'].sum(), color='C1', label='Size')
axs[1].legend(loc=2)
ax1r.legend(loc=1)
fig.suptitle('2018-02-23 Comparison', size='x-large')

# Plot "missing overlap modern", aka 2018-03-20
fig, axs = plt.subplots(2, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].set_title('Modern (Tableau)')
axs[0].plot(modern_overlap['Price'], color='C0', label='Price')
ax0r = axs[0].twinx()
# ax0r.plot(modern_overlap['Size'], color='C1', label='Size')
ax0r.plot(modern_overlap.groupby('Trade Time')['Size'].sum(), color='C1', label='Size')
axs[0].legend(loc=2)
ax0r.legend(loc=1)
axs[1].set_title('Missing (Production Tables)')
axs[1].plot(missing_overlap_modern['Price'], color='C0', label='Price')
ax1r = axs[1].twinx()
# ax1r.plot(missing_overlap_modern['Size'], color='C1', label='Size')
ax1r.plot(missing_overlap_modern.groupby('Trade Time')['Size'].sum(), color='C1', label='Size')
axs[1].legend(loc=2)
ax1r.legend(loc=1)
fig.suptitle('2018-03-20 Comparison', size='x-large')

###############################################################################

#### Project: Proposed VWAP Settlement vs. Actual Midpoint Settlement

# Set aside the 2 relevant time slices (3:15pm settlement and 3pm settlement), for both big VIX and mini VIX
vx_315 = mega_df.between_time('15:14:30', '15:15:00')   # .loc[:FIRST_DAY_3PM_SETTLEMENT-pd.Timedelta(days=1)]
vxm_315 = mini_df.between_time('15:14:30', '15:15:00')  # .loc[:FIRST_DAY_3PM_SETTLEMENT-pd.Timedelta(days=1)]
vx_300 = mega_df.between_time('14:59:30', '15:00:00')   # .loc[FIRST_DAY_3PM_SETTLEMENT:]
vxm_300 = mini_df.between_time('14:59:30', '15:00:00')  # .loc[FIRST_DAY_3PM_SETTLEMENT:]

# Calculate big VIX-only 3:15pm settlement
vx_315_vwap_price = \
    (vx_315.groupby(['Trade Date', 'Expire Date'])
     .apply(lambda df: (df['Size']*df['Price']).sum() / df['Size'].sum()))
vx_315_vwap_volume = \
    (vx_315.groupby(['Trade Date', 'Expire Date'])
     .apply(lambda df: df['Size'].sum() / 2))   # Want volume, not size, so divide by 2! Did not do above because it makes no difference in VWAP calc
vx_315_vwap = pd.DataFrame({'VWAP Settlement': vx_315_vwap_price, 'VWAP Volume': vx_315_vwap_volume})

# Calculate mini VIX-only 3:15pm settlement
vxm_315_vwap_price = \
    (vxm_315.groupby(['Trade Date', 'Expire Date'])
     .apply(lambda df: (df['Size']*df['Price']).sum() / df['Size'].sum()))
vxm_315_vwap_volume = \
    (vxm_315.groupby(['Trade Date', 'Expire Date'])
     .apply(lambda df: df['Size'].sum() / 2))   # Want volume
vxm_315_vwap = pd.DataFrame({'VWAP Settlement': vxm_315_vwap_price, 'VWAP Volume': vxm_315_vwap_volume})

# Calculate big VIX-only 3pm settlement
vx_300_vwap_price = \
    (vx_300.groupby(['Trade Date', 'Expire Date'])
     .apply(lambda df: (df['Size']*df['Price']).sum() / df['Size'].sum()))
vx_300_vwap_volume = \
    (vx_300.groupby(['Trade Date', 'Expire Date'])
     .apply(lambda df: df['Size'].sum() / 2))   # Want volume
vx_300_vwap = pd.DataFrame({'VWAP Settlement': vx_300_vwap_price, 'VWAP Volume': vx_300_vwap_volume})

# Calculate mini VIX-only 3pm settlement
vxm_300_vwap_price = \
    (vxm_300.groupby(['Trade Date', 'Expire Date'])
     .apply(lambda df: (df['Size']*df['Price']).sum() / df['Size'].sum()))
vxm_300_vwap_volume = \
    (vxm_300.groupby(['Trade Date', 'Expire Date'])
     .apply(lambda df: df['Size'].sum() / 2))   # Want volume
vxm_300_vwap = pd.DataFrame({'VWAP Settlement': vxm_300_vwap_price, 'VWAP Volume': vxm_300_vwap_volume})


# Calculate big+mini VIX 3:15pm settlement
vx_vxm_315_vwap_price = \
    ((vx_315_vwap['VWAP Settlement']*vx_315_vwap['VWAP Volume']*10
      + vxm_315_vwap['VWAP Settlement']*vxm_315_vwap['VWAP Volume'])
     / (vx_315_vwap['VWAP Volume']*10 + vxm_315_vwap['VWAP Volume']))
vx_vxm_315_vwap_volume = \
    vx_315_vwap['VWAP Volume'] + vxm_315_vwap['VWAP Volume']/10     # Scaled to big VIX; aka "Qualifying Contracts Traded"
vx_vxm_315_vwap = pd.DataFrame({'VWAP Settlement': vx_vxm_315_vwap_price,
                                'VWAP Volume': vx_vxm_315_vwap_volume}).dropna()

# Calculate big+mini VIX 3pm settlement
vx_vxm_300_vwap_price = \
    ((vx_300_vwap['VWAP Settlement']*vx_300_vwap['VWAP Volume']*10
      + vxm_300_vwap['VWAP Settlement']*vxm_300_vwap['VWAP Volume'])
     / (vx_300_vwap['VWAP Volume']*10 + vxm_300_vwap['VWAP Volume']))
vx_vxm_300_vwap_volume = \
    vx_300_vwap['VWAP Volume'] + vxm_300_vwap['VWAP Volume']/10     # Scaled to big VIX; aka "Qualifying Contracts Traded"
vx_vxm_300_vwap = pd.DataFrame({'VWAP Settlement': vx_vxm_300_vwap_price,
                                'VWAP Volume': vx_vxm_300_vwap_volume}).dropna()


# Segment #1: Legacy, Pre-Mini, Pre-Settlement Change (2012-01-03 - 2018-02-23)
vwap_1 = vx_315_vwap.loc[:'2018-02-23']
# Segment #2: Missing Production, Pre-Mini, Pre-Settlement Change (2018-02-26 - 2018-03-19)
vwap_2 = vx_315_vwap.loc['2018-02-26':'2018-03-19']
# Segment #3: Modern, Pre-Mini, Pre-Settlement Change (2018-03-20 - 2020-08-07)
vwap_3 = vx_315_vwap.loc['2018-03-20':'2020-08-07']
# Segment #4: Modern, with Mini, Pre-Settlement Change (2020-08-10 - 2020-10-23)
vwap_4_big = vx_315_vwap.loc['2020-08-10':'2020-10-23']
vwap_4_both = vx_vxm_315_vwap.loc['2020-08-10':'2020-10-23']    # Subset of big because minis are rarer
vwap_4 = vwap_4_big.copy()
vwap_4.update(vwap_4_both)
# Segment #5: Modern, with Mini, New 3pm Settlement (2020-10-26 - Present)
vwap_5_big = vx_300_vwap.loc['2020-10-26':]
vwap_5_both = vx_vxm_300_vwap.loc['2020-10-26':]    # Subset of big because minis are rarer
vwap_5 = vwap_5_big.copy()
vwap_5.update(vwap_5_both)
# Concatenated
vwap_final = pd.concat([vwap_1, vwap_2, vwap_3, vwap_4, vwap_5])

# Alternate organization
# Big only
vwap_big_315 = vx_315_vwap.loc[:FIRST_DAY_3PM_SETTLEMENT-pd.Timedelta(days=1)]
vwap_big_300 = vx_300_vwap.loc[FIRST_DAY_3PM_SETTLEMENT:]
vwap_big = pd.concat([vwap_big_315, vwap_big_300])
# Mini only
vwap_mini_315 = vxm_315_vwap.loc[:FIRST_DAY_3PM_SETTLEMENT-pd.Timedelta(days=1)]
vwap_mini_300 = vxm_300_vwap.loc[FIRST_DAY_3PM_SETTLEMENT:]
vwap_mini = pd.concat([vwap_mini_315, vwap_mini_300])
# Both big and mini only (subset of big only, since mini often doesn't trade)
vwap_both_315 = vx_vxm_315_vwap.loc[:FIRST_DAY_3PM_SETTLEMENT-pd.Timedelta(days=1)]
vwap_both_300 = vx_vxm_300_vwap.loc[FIRST_DAY_3PM_SETTLEMENT:]
vwap_both = pd.concat([vwap_both_315, vwap_both_300])
# Concatenate by layer
vwap_ultimate = \
    (vwap_big.join(vwap_mini, how='outer', rsuffix=' - Mini Only')
     .join(vwap_both, how='outer', rsuffix=' - Big and Mini (must have both)')
     .join(vwap_final, how='outer', rsuffix=' - Final (using Big and Mini where possible, Big Only where not)',
           lsuffix=' - Big Only'))    # Note the clever/obnoxious use of delayed lsuffix to rename columns
# vwap_ultimate.to_csv(DOWNLOADS_DIR + 'vwap_layers.csv')


# Pull VIX futures historical prices from Bloomberg
import futures_reader
good_ticker_list = \
    ['UXF12 Index', 'UXG12 Index', 'UXH12 Index', 'UXJ12 Index', 'UXK12 Index', 'UXM12 Index',
     'UXN12 Index', 'UXQ12 Index', 'UXU12 Index', 'UXV12 Index', 'UXX12 Index', 'UXZ12 Index',
     'UXF13 Index', 'UXG13 Index', 'UXH13 Index', 'UXJ13 Index', 'UXK13 Index', 'UXM13 Index',
     'UXN13 Index', 'UXQ13 Index', 'UXU13 Index', 'UXV13 Index', 'UXX13 Index', 'UXZ13 Index',
     'UXF14 Index', 'UXG14 Index', 'UXH14 Index', 'UXJ14 Index', 'UXK14 Index', 'UXM14 Index',
     'UXN14 Index', 'UXQ14 Index', 'UXU14 Index', 'UXV14 Index', 'UXX14 Index', 'UXZ14 Index',
     'UXF15 Index', 'UXG15 Index', 'UXH15 Index', 'UXJ15 Index', 'UXK15 Index', 'UXM15 Index',
     'UXN15 Index', 'UXQ15 Index', 'UXU15 Index', 'UXV15 Index', 'UXX15 Index', 'UXZ15 Index',
     'UXF16 Index', 'UXG16 Index', 'UXH16 Index', 'UXJ16 Index', 'UXK16 Index', 'UXM16 Index',
     'UXN16 Index', 'UXQ16 Index', 'UXU16 Index', 'UXV16 Index', 'UXX16 Index', 'UXZ16 Index',
     'UXF17 Index', 'UXG17 Index', 'UXH17 Index', 'UXJ17 Index', 'UXK17 Index', 'UXM17 Index',
     'UXN17 Index', 'UXQ17 Index', 'UXU17 Index', 'UXV17 Index', 'UXX17 Index', 'UXZ17 Index',
     'UXF18 Index', 'UXG18 Index', 'UXH18 Index', 'UXJ18 Index', 'UXK18 Index', 'UXM18 Index',
     'UXN18 Index', 'UXQ18 Index', 'UXU18 Index', 'UXV18 Index', 'UXX18 Index', 'UXZ18 Index',
     'UXF19 Index', 'UXG19 Index', 'UXH19 Index', 'UXJ19 Index', 'UXK19 Index', 'UXM19 Index',
     'UXN19 Index', 'UXQ19 Index', 'UXU19 Index', 'UXV19 Index', 'UXX19 Index', 'UXZ19 Index',
     'UXF20 Index', 'UXG20 Index', 'UXH20 Index', 'UXJ20 Index', 'UXK20 Index', 'UXM20 Index',
     'UXN20 Index', 'UXQ20 Index', 'UXU20 Index', 'UXV20 Index', 'UXX20 Index', 'UXZ20 Index',
     'UXF1 Index', 'UXG1 Index', 'UXH1 Index', 'UXJ1 Index', 'UXK1 Index', 'UXM1 Index', 'UXN1 Index']
# vix_futures = futures_reader.pull_fut_prices(
#     fut_codes='UX', start_datelike='2012-01-03', end_datelike=MODERN_END,
#     end_year_current=True, n_maturities_past_end=3,
#     contract_cycle='monthly', product_type='Index', ticker_list=good_ticker_list,
#     file_dir=DOWNLOADS_DIR, file_name='vix_futures_pull.csv',
#     bloomberg_con=None, verbose=True)
vix_futures = futures_reader.load_fut_prices(file_dir=DOWNLOADS_DIR, file_name='vix_futures_pull.csv')
vix_futures_yearmonth_list = [futures_reader.reverse_fut_ticker(ticker)[1] for ticker in good_ticker_list]
from options_futures_expirations_v3 import vix_thirty_days_before
vix_futures_expiries = [vix_thirty_days_before()(yearmonth_str) for yearmonth_str in vix_futures_yearmonth_list]
vix_ticker_expiry_dict = dict(zip(good_ticker_list, vix_futures_expiries))

vix_futures_formatted_list = []
for date in vix_futures.index:
    date_ser = vix_futures.loc[date].dropna()    # Has multiindex with ticker ('UXF12 Index', etc.) and field ('PX_LAST')
    date_tickers = date_ser.index.get_level_values('ticker')
    for date_ticker in date_tickers:
        date_ticker_price = date_ser.loc[date_ticker]['PX_LAST']
        formatted_el = (date, vix_ticker_expiry_dict[date_ticker], date_ticker_price, date_ticker)  # original date_ticker included as double-check
        vix_futures_formatted_list.append(formatted_el)
vix_futures_formatted_df = pd.DataFrame(vix_futures_formatted_list,
                                        columns=['Trade Date', 'Expire Date', 'Historical Settlement', 'BBG Ticker'])
vix_futures_formatted_df = vix_futures_formatted_df.set_index(['Trade Date', 'Expire Date']).sort_index()

# Line up VWAP settle with historical (mostly midpoint) settle
vwap_vs_midpoint = vwap_final.join(vix_futures_formatted_df, how='outer')

# Quick analysis
# 28/20,175 prices where VWAP was possible but no official settle price on BBG - all but (2013-07-02, 2014-03-19) is due to weeklies being erroneously included
# NOTE: Parth's "modern" data has weeklies and I don't currently have a field to filter them out;
#       I've visually picked them out via comparison to the BBG dates and will manually remove them for now
# NOTE: (2013-07-02, 2014-03-19) is wrong for a different reason - 2014-04 third Friday is Good Friday, so actual expiry date is 2014-03-18
no_midpoint = vwap_vs_midpoint.loc[vwap_vs_midpoint['Historical Settlement'].isna()]    # Not sure why these are missing settlements on BBG...
makeshift_weeklies_idx = no_midpoint.index[1:]
makeshift_vwap_vs_midpoint = vwap_vs_midpoint.drop(makeshift_weeklies_idx)
makeshift_vwap_vs_midpoint.loc[(pd.Timestamp('2013-07-02'), pd.Timestamp('2014-03-18')), ['VWAP Settlement', 'VWAP Volume']] = makeshift_vwap_vs_midpoint.loc[no_midpoint.index[0], ['VWAP Settlement', 'VWAP Volume']]
makeshift_vwap_vs_midpoint = makeshift_vwap_vs_midpoint.drop(no_midpoint.index[0])
# 80 expiry days - obviously we have no VWAP on that morning, but BBG provides a price.
# this is meaningless to our comparison, so we'll strike them out of the makeshift
test = makeshift_vwap_vs_midpoint.reset_index()
test = test.loc[test['Trade Date'] == test['Expire Date']].set_index(['Trade Date', 'Expire Date'])     # Tried >=, didn't have any >
makeshift_vwap_vs_midpoint = makeshift_vwap_vs_midpoint.drop(test.index)
# 2910 (vs. 20,067 total, 14.5%) prices where VWAP was not possible but historically we had a settle price on BBG
no_vwap = makeshift_vwap_vs_midpoint.loc[makeshift_vwap_vs_midpoint['VWAP Settlement'].isna()]

# Export
# makeshift_vwap_vs_midpoint.to_csv(DOWNLOADS_DIR + 'vwap_vs_historical.csv')
# no_vwap.to_csv(DOWNLOADS_DIR + 'no_vwap.csv')
vwap_vs_midpoint_only_matches = makeshift_vwap_vs_midpoint.dropna()
# vwap_vs_midpoint_only_matches.to_csv(DOWNLOADS_DIR + 'vwap_vs_historical_only_matches.csv')

# Using groupby() and head() and nth(), we can break down to near terms
# Fun fact: 2019-12-27 is the only date where 10 expiries were capable of VWAP (2020-01-22 to 2020-10-21)
first_n_terms_dict = {}
first_n_terms_dict_abs = {}
for n in range(1, 11):
    first_n_terms = vwap_vs_midpoint_only_matches.groupby('Trade Date').head(n).copy()
    first_n_terms['Rounded VWAP'] = first_n_terms['VWAP Settlement'].round(3)
    first_n_terms['Rounded Minus Historical'] = first_n_terms['Rounded VWAP'] - first_n_terms['Historical Settlement']
    first_n_terms['Abs Minus'] = first_n_terms['Rounded Minus Historical'].abs()
    first_n_terms_dict[n] = first_n_terms['Rounded Minus Historical'].describe().round(3)
    first_n_terms_dict_abs[n] = first_n_terms['Abs Minus'].describe().round(3)

# Analysis
# - focus on first 4 terms - the rest no one really cares about; limit all analysis to first 4
n = 4
first_n_terms = vwap_vs_midpoint_only_matches.groupby('Trade Date').head(n).copy()
first_n_terms['Rounded VWAP'] = first_n_terms['VWAP Settlement'].round(3)
first_n_terms['Rounded Minus Historical'] = first_n_terms['Rounded VWAP'] - first_n_terms['Historical Settlement']
first_n_terms['Abs Minus'] = first_n_terms['Rounded Minus Historical'].abs()
# - consider nickel ticks: at the tightest spread, midpoint settle is 2.5 cents off trade price, then 5 cents, then 7.5 cents, etc.
#   we should check number of occurrences for each of these thresholds for VWAP-historical difference
# More than 2.5 cents
more_than_25 = first_n_terms[first_n_terms['Abs Minus'] > 0.025]
yearmonth_occurrences_list = []
for year in range(2012, 2021):
    for month in range(1, 13):
        yearmonth_str = f"{year}-{month}"
        n_occurrences = len(more_than_25.loc[yearmonth_str])
        n_total = len(first_n_terms.loc[yearmonth_str])
        print(yearmonth_str, ": ", n_occurrences)
        yearmonth_occurrences_list.append((pd.Timestamp(yearmonth_str), n_occurrences, n_occurrences/n_total))
yearmonth_occurrences_25_df = pd.DataFrame(yearmonth_occurrences_list, columns=['Month', 'Occurrences of > 2.5 cents difference', 'Percentage']).set_index('Month') # Plot this for visual
for year in range(2012, 2021):
    print(f"{year}: ", len(more_than_25.loc[str(year)]))
# More than 5 cents
more_than_50 = first_n_terms[first_n_terms['Abs Minus'] > 0.05]
yearmonth_occurrences_list = []
for year in range(2012, 2021):
    for month in range(1, 13):
        yearmonth_str = f"{year}-{month}"
        n_occurrences = len(more_than_50.loc[yearmonth_str])
        n_total = len(first_n_terms.loc[yearmonth_str])
        print(yearmonth_str, ": ", n_occurrences)
        yearmonth_occurrences_list.append((pd.Timestamp(yearmonth_str), n_occurrences, n_occurrences/n_total))
yearmonth_occurrences_50_df = pd.DataFrame(yearmonth_occurrences_list, columns=['Month', 'Occurrences of > 5 cents difference', 'Percentage']).set_index('Month') # Plot this for visual
for year in range(2012, 2021):
    print(f"{year}: ", len(more_than_50.loc[str(year)]))
# More than 7.5 cents
more_than_75 = first_n_terms[first_n_terms['Abs Minus'] > 0.075]
yearmonth_occurrences_list = []
for year in range(2012, 2021):
    for month in range(1, 13):
        yearmonth_str = f"{year}-{month}"
        n_occurrences = len(more_than_75.loc[yearmonth_str])
        n_total = len(first_n_terms.loc[yearmonth_str])
        print(yearmonth_str, ": ", n_occurrences)
        yearmonth_occurrences_list.append((pd.Timestamp(yearmonth_str), n_occurrences, n_occurrences/n_total))
yearmonth_occurrences_75_df = pd.DataFrame(yearmonth_occurrences_list, columns=['Month', 'Occurrences of > 7.5 cents difference', 'Percentage']).set_index('Month') # Plot this for visual
for year in range(2012, 2021):
    print(f"{year}: ", len(more_than_75.loc[str(year)]))
# More than 10 cents
more_than_100 = first_n_terms[first_n_terms['Abs Minus'] > 0.10]
yearmonth_occurrences_list = []
for year in range(2012, 2021):
    for month in range(1, 13):
        yearmonth_str = f"{year}-{month}"
        n_occurrences = len(more_than_100.loc[yearmonth_str])
        n_total = len(first_n_terms.loc[yearmonth_str])
        print(yearmonth_str, ": ", n_occurrences)
        yearmonth_occurrences_list.append((pd.Timestamp(yearmonth_str), n_occurrences, n_occurrences/n_total))
yearmonth_occurrences_100_df = pd.DataFrame(yearmonth_occurrences_list, columns=['Month', 'Occurrences of > 10 cents difference', 'Percentage']).set_index('Month') # Plot this for visual
for year in range(2012, 2021):
    print(f"{year}: ", len(more_than_100.loc[str(year)]))
# Plot
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title("Per Month Frequency of Big VWAP-Midpoint Differences (In Front 4 Maturities)")
ax.plot(yearmonth_occurrences_25_df['Percentage']*100, label=yearmonth_occurrences_25_df.columns[0], color='C0')
ax.plot(yearmonth_occurrences_50_df['Percentage']*100, label=yearmonth_occurrences_50_df.columns[0], color='C1')
ax.plot(yearmonth_occurrences_75_df['Percentage']*100, label=yearmonth_occurrences_75_df.columns[0], color='C2')
ax.plot(yearmonth_occurrences_100_df['Percentage']*100, label=yearmonth_occurrences_100_df.columns[0], color='C3')
ax.set_ylabel("Percentage of Total Date-Maturities (%)")
ax.legend()
fig.tight_layout()
# Complement
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title("Per Month Match of VWAP vs. Midpoint Settle (Front 4 Maturities)")
ax.plot((1-yearmonth_occurrences_25_df['Percentage'])*100, label='$\leq$2.5 cents difference', color='C0')
ax.axhline(100, color='k', linestyle='--')
ax.plot((1-yearmonth_occurrences_50_df['Percentage'])*100, label='$\leq$5 cents difference', color='C1')
ax.plot((1-yearmonth_occurrences_75_df['Percentage'])*100, label='$\leq$7.5 cents difference', color='C2')
ax.plot((1-yearmonth_occurrences_100_df['Percentage'])*100, label='$\leq$10 cents difference', color='C3')
ax.set_ylabel("Percentage of Total Date-Maturities (%)")
ax.legend()
fig.tight_layout()
# - we want case studies for recent big (by amount) difference days
print(first_n_terms.loc['2020', 'Abs Minus'].describe())
big_days = first_n_terms.loc['2020'][first_n_terms.loc['2020', 'Abs Minus'] > 0.2]
# Looks like 2020-09-03 and 2020-09-08 (big SPX drop in early Sep) and 2020-10-28 are prime candidates
# There are a bunch of big days in March as well, so we can figure out what was happening at beginning of COVID
# Plot
from cboe_exchange_holidays_v3 import datelike_to_timestamp
from mpl_tools import save_fig
def big_diff_case_study_plot(trade_datelike, expiry_datelike, intraday_dataset, big_days_dataset):
    trade_date = datelike_to_timestamp(trade_datelike)
    expiry_date = datelike_to_timestamp(expiry_datelike)
    trade_date_data = intraday_dataset.loc[trade_datelike]  # Doesn't work if Timestamp is used!
    trade_date_expiry_data = trade_date_data[trade_date_data['Expire Date'] == expiry_date]
    # Plot
    func_fig, func_ax = plt.subplots(figsize=(19.2, 10.8))
    func_ax.set_title(f"{trade_date.strftime('%Y-%m-%d')}, Maturity {expiry_date.strftime('%Y-%m-%d')} "
                      f"Settlement Price: VWAP vs. Historical")
    func_ax.plot(trade_date_expiry_data['Bid Price'], color='C1', label='Bid Price')
    func_ax.plot(trade_date_expiry_data['Ask Price'], color='C2', label='Ask Price')
    func_ax.plot(trade_date_expiry_data['Price'], marker='o', linestyle='none', color='C0', label='Trade Price')
    # func_ax.plot(trade_date_expiry_data['Price'], marker='o', linestyle='-', color='C0', alpha=0.4, label='Trade Price')
    func_axr = func_ax.twinx()
    size_resample = trade_date_expiry_data.resample('100ms')['Size'].sum()  # Do resample-sum since matplotlib will not automatically aggregate over time
    func_axr.bar(size_resample.index, size_resample, width=1/(24*60*60*10), align='edge',
                 color='C3', alpha=0.5, label='Size (2x Volume)')   # width is fraction of a day, so I'm trying to match 100ms
    func_axr.grid(False, which='major', axis='y')   # In mplstyles that show twinx grids, we want to turn off secondary horizontal lines
    quick_stats = big_days_dataset.loc[(trade_date, expiry_date)]
    func_ax.axhline(y=quick_stats['Historical Settlement'], color='k', alpha=0.6, linestyle='--',
                    label=f"Historical Settle {quick_stats['Historical Settlement']:.3f}")
    func_ax.axhline(y=quick_stats['Rounded VWAP'], color='C4', alpha=0.6, linestyle='--',
                    label=f"VWAP Settle {quick_stats['Rounded VWAP']:.3f} ({quick_stats['Rounded Minus Historical']:+.3f})")
    func_ax.legend(loc=2)
    func_axr.legend(loc=1)
    return func_fig
# Case 1.
# 2020-09-03: 10/21 had VWAP 38.466 vs. 39.025 (-0.559)
#             11/18 had VWAP 34.899 vs. 35.15 (-0.251)
fig_1a = big_diff_case_study_plot('2020-09-03', '2020-10-21', vx_315, big_days)
fig_1b = big_diff_case_study_plot('2020-09-03', '2020-11-18', vx_315, big_days)
save_fig(fig_1a, f"VWAP Settle Case Study 2020-09-03 Exp 2020-10-21.png", DOWNLOADS_DIR)
save_fig(fig_1b, f"VWAP Settle Case Study 2020-09-03 Exp 2020-11-18.png", DOWNLOADS_DIR)
# Case 2
# 2020-09-08: 11/18 had VWAP 32.041 vs. 31.825 (+0.216)
fig_2 = big_diff_case_study_plot('2020-09-08', '2020-11-18', vx_315, big_days)
save_fig(fig_2, f"VWAP Settle Case Study 2020-09-08 Exp 2020-11-18.png", DOWNLOADS_DIR)
# Case 3
# 2020-10-28: 12/16 had VWAP 34.516 vs. 34.725 (-0.209)
fig_3 = big_diff_case_study_plot('2020-10-28', '2020-12-16', vx_300, big_days)
save_fig(fig_3, f"VWAP Settle Case Study 2020-10-28 Exp 2020-12-16.png", DOWNLOADS_DIR)
# COVID March
covid_days = big_days.index.get_level_values('Trade Date')[:-4].unique()
for day in covid_days:
    for exp_day in big_days.loc[day].index:
        covid_fig = big_diff_case_study_plot(day.strftime('%Y-%m-%d'), exp_day, vx_315, big_days)
        save_fig(covid_fig, f"VWAP Settle Case Study {day.strftime('%Y-%m-%d')} Exp {exp_day.strftime('%Y-%m-%d')}.png", DOWNLOADS_DIR)

# Same case studies can be repeated for 2019, but no day went over 9 pennies difference
big_days_2019 = first_n_terms.loc['2019'][first_n_terms.loc['2019', 'Abs Minus'] > 0.07]
