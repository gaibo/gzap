import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET, generate_expiries
import matplotlib.pyplot as plt
plt.style.use('cboe-fivethirtyeight')

# [MANUAL] Configure end date
END_DATE = pd.Timestamp('2021-02-03')

# Makeshift output folder
DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

# Parth's Y drive general data folder
PARTH_DIR = 'Y:/Research/Research1/Parth/Fudon Project/Input Data/'

#### Load VIX #################################################################
# We currently must use 3 data sources:
#   1) SAS tables for old stuff (LEGACY)
#   2) a random CSV Charlie Barrett pulled from production data tables (MISSING)
#   3) Tableau/Data Platform for new stuff (MODERN)
# Each source has filters for isolating TAS, thankfully. Data issues that required additional code to clean:
#   !) On trade date 2013-07-02, 2014-03-19 is marked as a maturity instead of 2014-03-18.
#      Good Friday pushed SPX expiry 2014-04-18 to 17, and apparently this was not fixed in SAS tables.
#      Interestingly though, on every other trade date, they already have the corrected 2014-03-18 maturity.
#   2) Modern data includes Weeklys, with no built-in filter between them and Monthlys,
#      so I wrote code to take them out against a list of generated maturities.
#   3) Legacy data is one-sided, so "size" refers to volume. The other data is two-sided,
#      as in there are rows for both buyer and seller, so volume is "size"/2.
# The bulk of this code is just formatting the 3 sources together.
# For obvious reasons, "MODERN" data is the most useful for analysis and should
# always be ready for use. Loading all data is very time/memory intensive.

LEGACY_DATA_DIR = PARTH_DIR + 'Raw Data (SAS Pulled)/Futures Cleared Trades/'
LEGACY_DATA_FILE = '{} Cleared Trades.csv'   # '23-Feb-18 Cleared Trades.csv'
MISSING_DATA_DIR = DOWNLOADS_DIR
MISSING_DATA_FILE = 'futures_trades_missing_final.csv'  # Literally just one offhand file
MODERN_DATA_DIR = PARTH_DIR + 'Raw Data (Tableau Pulled)/Tableau Raw Data Contra Futures/VX/'
MODERN_DATA_FILE = '{} VX Contra Trade Data.csv'    # '20180320 VX Contra Trade Data.csv'

# Important dates
LEGACY_START, LEGACY_END = pd.Timestamp('2009-01-02'), pd.Timestamp('2018-02-23')
RELEVANT_START = pd.Timestamp('2012-01-03')   # 2012 to present is what we'll mark as "reasonable VIX era"
legacy_days = pd.date_range(start=RELEVANT_START, end=LEGACY_END, freq=BUSDAY_OFFSET)
MISSING_START, MISSING_END = pd.Timestamp('2018-02-22'), pd.Timestamp('2018-03-21')    # Midnight cutoffs instead of 5pm
missing_days = pd.date_range(start=MISSING_START, end=MISSING_END, freq=BUSDAY_OFFSET)
MODERN_START, MODERN_END = pd.Timestamp('2018-03-20'), END_DATE
modern_days = pd.date_range(start=MODERN_START, end=MODERN_END, freq=BUSDAY_OFFSET)
all_days = pd.date_range(start=RELEVANT_START, end=MODERN_END, freq=BUSDAY_OFFSET)
FIRST_DAY_3PM_SETTLEMENT = pd.Timestamp('2020-10-26')
FIRST_DAY_MINI_VIX = pd.Timestamp('2020-08-10')

#### 1) Load "legacy" SAS data tables

LEGACY_USECOLS = ['Entry Time', 'Date', 'Expiry Date',
                  'Class', 'Trade Size', 'Trade Price']   # Class 'VXT' is TAS

legacy_data_list = []
for day in legacy_days:
    day_data = pd.read_csv(LEGACY_DATA_DIR + LEGACY_DATA_FILE.format(day.strftime('%d-%b-%y')),
                           usecols=LEGACY_USECOLS, parse_dates=['Date', 'Entry Time', 'Expiry Date'])
    legacy_data_list.append(day_data)
    print(day.strftime('%Y-%m-%d'), "done")

# Sort
print("LEGACY loaded. Sorting...")
legacy_data = pd.concat(legacy_data_list).set_index('Entry Time').sort_index()
# Deal with 2014-03-19 data issue; inplace because not enough RAM
print("Fixing 2014-03-19 maturity...")
legacy_data.replace(pd.Timestamp('2014-03-19'), {'Expiry Date': pd.Timestamp('2014-03-18')}, inplace=True)
# Split non-TAS from TAS
legacy_vix = legacy_data[legacy_data['Class'] == 'VX'].copy()   # Live, no-TAS
legacy_tas = legacy_data[legacy_data['Class'] == 'VXT'].copy()

#### 2) Load "missing" from production tables for 1 month of disconnect - 2018-02-23 to 2018-03-20

missing_data = pd.read_csv(DOWNLOADS_DIR + MISSING_DATA_FILE)

# There are 2 transact_time columns, and one of them needs to be hard-corrected to transact_date
missing_data = missing_data.rename({'transact_time': 'transact_date', 'transact_time.1': 'transact_time'}, axis=1)
# Isolate VX from other roots
missing_data = missing_data[missing_data['futures_root'] == 'VX']
# Convert to Timestamp
missing_data['transact_date'] = pd.to_datetime(missing_data['transact_date'])
missing_data['expire_date'] = pd.to_datetime(missing_data['expire_date'])
# Remove the UTC offset - it's just daylight savings I think and confounds the data
missing_data['transact_time_no_tz'] = pd.to_datetime(missing_data['transact_time'].apply(lambda s: s[:-3]))

MISSING_USECOLS = ['transact_time_no_tz', 'transact_date', 'expire_date',
                   'futures_root', 'size', 'price',
                   'tas']   # tas 'Y' is TAS

# Sort
print("MISSING loaded. Sorting...")
missing_data = missing_data.set_index('transact_time_no_tz').sort_index()
# Split non-TAS from TAS
missing_vix = missing_data[missing_data['tas'] == 'N'].copy()   # Live, no-TAS
missing_tas = missing_data[missing_data['tas'] == 'Y'].copy()

#### 3) Load "modern" Tableau data tables
# NOTE: there are many extra fields for firm-level contras analysis
# NOTE: empirically, it is most memory efficient by far to do date parsing inside the for-loop,
#       prior to concatenating.

# MODERN_USECOLS = ['Trade Time', 'Trade Date', 'Expire Date',
#                   'Product Type', 'Side',
#                   'On Behalf Of', 'Name', 'CTI',
#                   'Contra On Behalf Of', 'Contra Name', 'Contra CTI',
#                   'Price', 'Size', 'Bid Price', 'Ask Price']    # Product Type 'T' is TAS

MODERN_USECOLS = ['Trade Time', 'Trade Date', 'Expire Date',
                  'Product Type', 'Side',
                  'On Behalf Of', 'CTI',
                  'Price', 'Size', 'Bid Price', 'Ask Price']

modern_data_list = []
for day in modern_days:
    day_data = pd.read_csv(MODERN_DATA_DIR + MODERN_DATA_FILE.format(day.strftime('%Y%m%d')),
                           usecols=MODERN_USECOLS, parse_dates=['Trade Date', 'Trade Time', 'Expire Date'])
    modern_data_list.append(day_data)
    print(day.strftime('%Y-%m-%d'), "done")

# Sort
print("MODERN loaded. Converting to Timestamp and sorting...")
modern_data = pd.concat(modern_data_list).set_index('Trade Time').sort_index()
modern_data = modern_data[MODERN_USECOLS[1:]]   # Enforce column order
# Deal with unfilterable Weeklys
print("Filtering out Weeklys...")
# Generate set of VIX Monthly maturities
vix_monthly_maturities = generate_expiries('2012-01-03', MODERN_END+pd.DateOffset(months=12), specific_product='VIX')
# Find maturities in modern_data not in those maturities
detected_weeklies = pd.Series(list(set(modern_data['Expire Date'].drop_duplicates())
                                   - set(vix_monthly_maturities))).sort_values().reset_index(drop=True)
# Remove detected weeklies
modern_weeklies_df = modern_data[modern_data['Expire Date'].isin(detected_weeklies)].copy()
modern_data = modern_data[~modern_data['Expire Date'].isin(detected_weeklies)]  # Weeklies are taken out!
# Split non-TAS from TAS
modern_vix = modern_data[modern_data['Product Type'] == 'S'].drop('Product Type', axis=1)   # Live, no-TAS
modern_tas = modern_data[modern_data['Product Type'] == 'T'].copy()


#### Combine all 3 data sources for VIX volume-specific analysis ##############
# Cut out info not related to volumes and rename columns
# Want to end up with: 'Trade Time', 'Trade Date', 'Expire Date', 'Size', 'Price', 'Bid Price', 'Ask Price'
# (as named in modern Tableau data)

#### VIX price-influencing trading, no TAS

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
mega_df = pd.concat([legacy_rename.between_time('14:00', '15:15'),
                     missing_rename.between_time('14:00', '15:15').loc['2018-02-24':'2018-03-19 16:00:01'],
                     modern_rename.between_time('14:00', '15:15')], sort=False)
mega_df['Volume'] = mega_df['Size'] / 2
mega_df = mega_df[['Trade Date', 'Expire Date', 'Size', 'Volume', 'Price', 'Bid Price', 'Ask Price']]   # Useful
print("mega_df for settlement volume analysis created.")

# Ensure fact that each trade date spans 2 days does not throw index date-time off from "trade date" column
# We are only taking a subset of each trading day, so we shouldn't be dealing with this yet
idx_date = pd.to_datetime(pd.Series(mega_df.index.date, index=mega_df.index))
assert mega_df[(idx_date != mega_df['Trade Date'])].empty

#### TAS

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

# Combine (calling it "mega" because 1) it's big VIX not mini and 2) it's a HUGE dataset
mega_tas_df = pd.concat([legacy_tas_rename.between_time('14:00', '15:15'),
                         missing_tas_rename.between_time('14:00', '15:15').loc['2018-02-24':'2018-03-19 16:00:01'],
                         modern_tas_rename.between_time('14:00', '15:15')], sort=False)
mega_tas_df['Volume'] = mega_tas_df['Size'] / 2
mega_tas_df = mega_tas_df[['Trade Date', 'Expire Date', 'Size', 'Volume', 'Price', 'Bid Price', 'Ask Price']]   # Useful
print("mega_tas_df for settlement volume analysis created.")

#### Load Mini-VIX ############################################################
# Mini-VIX doesn't have to deal with 3 data sources, since it began 2020-08-10.
# It also does not have Weeklys or TAS yet.
# This data is also fit for firm-level analysis.

MINI_DATA_DIR = PARTH_DIR + 'Raw Data (Tableau Pulled)/Tableau Raw Data Contra Futures/VXM/'
MINI_DATA_FILE = '{} VXM Contra Trade Data.csv'    # '20180831 VXM Contra Trade Data.csv'
mini_data_list = []
# mini_months = pd.date_range(FIRST_DAY_MINI_VIX, MODERN_END).strftime('%Y%m').unique()     # Obsolete format
mini_months = pd.date_range(FIRST_DAY_MINI_VIX, MODERN_END).to_period('M').to_timestamp('M').unique().strftime('%Y%m%d')
for yearmonth_str in mini_months:
    month_data = pd.read_csv(MINI_DATA_DIR + MINI_DATA_FILE.format(yearmonth_str),
                             usecols=MODERN_USECOLS, parse_dates=['Trade Date', 'Trade Time', 'Expire Date'])
    mini_data_list.append(month_data)
    print(yearmonth_str, "done")
mini_data = pd.concat(mini_data_list).set_index('Trade Time').sort_index()
# Deal with unfilterable Weeklys
print("Filtering out Weeklys...")
# Find maturities in modern_data not in those maturities
mini_detected_weeklies = pd.Series(list(set(mini_data['Expire Date'].drop_duplicates())
                                        - set(vix_monthly_maturities))).sort_values().reset_index(drop=True)
# Remove detected weeklies
mini_weeklies_df = mini_data[mini_data['Expire Date'].isin(mini_detected_weeklies)].copy()
mini_data = mini_data[~mini_data['Expire Date'].isin(mini_detected_weeklies)]  # Weeklies are taken out!
# Split non-TAS from TAS
mini_vix = mini_data[mini_data['Product Type'] == 'S'].copy()   # Live, no-TAS
mini_tas = mini_data[mini_data['Product Type'] == 'T'].copy()

# Subset for volume analysis, parallel to mega_df
mini_df = mini_vix.between_time('14:00', '15:15')
mini_df['Volume'] = mini_df['Size'] / 2
mini_df = mini_df[['Trade Date', 'Expire Date', 'Size', 'Volume', 'Price', 'Bid Price', 'Ask Price']]   # Useful
print("mini_df for settlement volume analysis created.")

# Ensure fact that each trade date spans 2 days does not throw index date-time off from "trade date" column
# We are only taking a subset of each trading day, so we shouldn't be dealing with this yet
idx_date = pd.to_datetime(pd.Series(mini_df.index.date, index=mini_df.index))
assert mini_df[(idx_date != mini_df['Trade Date'])].empty


###############################################################################

from cboe_exchange_holidays_v3 import datelike_to_timestamp
def big_diff_case_study_plot(trade_datelike, expiry_datelike, intraday_dataset, settle_price=None, last_price=None):
    trade_date = datelike_to_timestamp(trade_datelike)
    expiry_date = datelike_to_timestamp(expiry_datelike)
    trade_date_data = intraday_dataset.loc[trade_datelike]  # Doesn't work if Timestamp is used!
    trade_date_expiry_data = trade_date_data[trade_date_data['Expire Date'] == expiry_date]
    # Plot
    func_fig, func_ax = plt.subplots(figsize=(19.2, 10.8))
    func_ax.set_title(f"{trade_date.strftime('%Y-%m-%d')}, Maturity {expiry_date.strftime('%Y-%m-%d')} Settlement")
    func_ax.plot(trade_date_expiry_data['Bid Price'], color='C1', label='Bid Price')
    func_ax.plot(trade_date_expiry_data['Ask Price'], color='C2', label='Ask Price')
    func_ax.plot(trade_date_expiry_data['Price'], marker='o', linestyle='none', color='C0', label='Trade Price')
    # func_ax.plot(trade_date_expiry_data['Price'], marker='o', linestyle='-', color='C0', alpha=0.4, label='Trade Price')
    func_axr = func_ax.twinx()
    size_resample = trade_date_expiry_data.resample('100ms')['Size'].sum()/2  # Do resample-sum since matplotlib will not automatically aggregate over time
    func_axr.bar(size_resample.index, size_resample, width=1/(24*60*60*10), align='edge',
                 color='C3', alpha=0.5, label='Volume')   # width is fraction of a day, so I'm trying to match 100ms
    func_axr.grid(False, which='major', axis='y')   # In mplstyles that show twinx grids, we want to turn off secondary horizontal lines
    if settle_price is not None:
        func_ax.axhline(y=settle_price, color='k', alpha=0.6, linestyle='--',
                        label=f"VWAP Settle {settle_price:.4f}")    # Assume is VWAP
    if last_price is not None:
        func_ax.axhline(y=last_price, color='C4', alpha=0.6, linestyle='--',
                        label=f"Last Price {last_price:.2f}")
    func_ax.legend(loc=2)
    func_axr.legend(loc=1)
    return func_fig, func_ax

vix1 = modern_vix[(modern_vix['Trade Date'] == pd.Timestamp('2021-01-27')) & (modern_vix['Expire Date'] == pd.Timestamp('2021-02-17'))]
tas1 = modern_tas[(modern_tas['Trade Date'] == pd.Timestamp('2021-01-27')) & (modern_tas['Expire Date'] == pd.Timestamp('2021-02-17'))]
vix2 = modern_vix[(modern_vix['Trade Date'] == pd.Timestamp('2021-01-27')) & (modern_vix['Expire Date'] == pd.Timestamp('2021-03-17'))]
tas2 = modern_tas[(modern_tas['Trade Date'] == pd.Timestamp('2021-01-27')) & (modern_tas['Expire Date'] == pd.Timestamp('2021-03-17'))]
vix1_zoom = vix1.between_time('14:59', '15:00')
tas1_zoom = tas1.between_time('14:59', '15:00')
vix2_zoom = vix2.between_time('14:59', '15:00')
tas2_zoom = tas2.between_time('14:59', '15:00')
tas1_zoom_edited = tas1_zoom.copy()
tas1_zoom_edited['Price'] = tas1_zoom['Price'] - 31.5342
tas2_zoom_edited = tas2_zoom.copy()
tas2_zoom_edited['Price'] = tas2_zoom['Price'] - 31.2313

big_diff_case_study_plot('2021-01-27', '2021-02-17', vix1_zoom, 31.5342)
fig, ax = big_diff_case_study_plot('2021-01-27', '2021-02-17', tas1_zoom_edited, None, 0.04)
ax.set_ylim(-0.01, 0.05)

big_diff_case_study_plot('2021-01-27', '2021-03-17', vix2_zoom, 31.2313)
fig, ax = big_diff_case_study_plot('2021-01-27', '2021-03-17', tas2_zoom_edited, None, -0.22)
# ax.set_ylim(-0.01, 0.05)
