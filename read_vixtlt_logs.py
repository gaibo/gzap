import pandas as pd
from pathlib import Path
from options_data_tools import add_t_to_exp, add_rate, add_forward, lookup_val_in_col
import matplotlib.pyplot as plt
plt.style.use('cboe-fivethirtyeight')

LOGS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Gaibo/VIXTLT/VIXTLTP Log Pulls/')
FILE_NAME_STR = 'VXTLTPE_2021_09_24.log'  # CHANGE MANUALLY; e.g. 'VXTLTPE_2021_09_20_10am_11am.log'
EVENT_NAME_STR = '2021-09-24'    # CHANGE MANUALLY; e.g. '2021-09-20 10AM Large Changes'

# Load raw data as CSV
data_raw = pd.read_csv(LOGS_DIR/FILE_NAME_STR, names=range(8))
# Break down messy first column
data_raw_0_split = data_raw[0].str.split('::')
data_raw_0_df = pd.DataFrame(row_list for row_list in data_raw_0_split)
# Extract info from first column
data = data_raw.copy()
data['Calc Datetime'] = pd.to_datetime(data_raw_0_df[1].apply(lambda s: s[:-4]+'.'+s[-3:]))     # Change : to .
data_raw_0_4_strip = data_raw_0_df[4].str.strip()    # Strip rogue spaces
data['Is Price'] = ((data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Near Term Calls')
                    | (data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Near Term Puts')
                    | (data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Next Term Calls')
                    | (data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Next Term Puts'))
data.loc[((data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Near Term Calls')
         | (data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Near Term Puts')),
         'Term'] = 'Near'
data.loc[((data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Next Term Calls')
         | (data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Next Term Puts')),
         'Term'] = 'Next'
data.loc[((data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Near Term Calls')
         | (data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Next Term Calls')),
         'CP'] = 'C'
data.loc[((data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Near Term Puts')
         | (data_raw_0_4_strip == 'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0 Next Term Puts')),
         'CP'] = 'P'
# Focus on consistently-formatted price data
data = data[data['Is Price']]
data['Strike'] = data[2].astype(float)
data['Expiry'] = pd.to_datetime(data[3]) + pd.to_timedelta('15:00:00')    # 3pm expiry time - need minute precision
data['Bid'] = data[5].astype(float)
data['Ask'] = data[7].astype(float)
# Clean up for final format
nncp = data[['Calc Datetime', 'Term', 'Expiry', 'CP', 'Strike', 'Bid', 'Ask']].copy()  # Near next call put

# Use options_data_tools on the data
nncp['Mid'] = (nncp['Bid'] + nncp['Ask']) / 2
nncp = add_t_to_exp(nncp, 'Calc Datetime', 'Expiry', 't_to_exp')
nncp = add_rate(nncp, 'Calc Datetime', 't_to_exp', 'rate')
# Need to group inputs by 15-second interval so we can 1) calculate forward, 2) get K0
# NOTE: options_data_tools was written for doing one calc per day (at close) so used date in a multiindex to group
#       data; the more precise datetime cannot currently be used - we need a separate date column;
#       just adding .normalize() to options_data_tools isn't a great solution because 2021-11-09 date could include
#       2021-11-08 17:40:30.321 prices - probably need to modify options_data_tools with separate datetime column
nncp['Calc Group'] = (nncp['Calc Datetime'].diff() > pd.Timedelta(seconds=5)).cumsum() + 1  # +1 so 1st group is 1
nncp['Calc Date'] = nncp['Calc Datetime'].dt.normalize()    # TLT doesn't have GTH, so .normalize() works
grouped_nncp = nncp.set_index('Calc Group')
# add_forward() usage complicated by intraday - calculate forwards for each group
# NOTE: forward is unique to group and each term/expiry within
forwards_nncp = grouped_nncp.copy()     # Prepare new DF to collect results; prob not the cleverest way
for group in grouped_nncp.index.unique():
    nncp_group_forwards = add_forward(grouped_nncp.loc[group], 'Calc Date', 'Expiry', 'Strike', 'CP',
                                      'Mid', 't_to_exp', 'rate', 'forward')
    forwards_nncp.loc[group, 'forward'] = nncp_group_forwards['forward'].values     # Overwrite in new DF
# Calculate K_0 (strike just under forward) for each forward
unique_forwards = forwards_nncp.groupby(['Calc Group', 'Term'])['forward'].first()
forwards_nncp = forwards_nncp.set_index('Term', append=True)    # Now multtindex is (Calc Group, Term)
for group_term_idx in unique_forwards.index:
    unindexed_data_slice = forwards_nncp.loc[group_term_idx].reset_index()  # options_data_tools needs unindexed
    forward = unique_forwards.loc[group_term_idx]
    forwards_nncp.loc[group_term_idx, 'K_0'] = \
        lookup_val_in_col(unindexed_data_slice, forward, 'Strike', leq_only=True)['Strike']

# Use K_0 to select only OTM data
otm_nncp = forwards_nncp.loc[((forwards_nncp['CP'] == 'C') & (forwards_nncp['Strike'] >= forwards_nncp['K_0']))
                             | ((forwards_nncp['CP'] == 'P') & (forwards_nncp['Strike'] <= forwards_nncp['K_0']))] \
           .copy()

# Trim 0-bids to focus on meaningful data
# Logic is to spot consecutive 0-bid index by using Booleans & with next OTM, then crop from there
# Beyond that, we must also remove all remaining strikes with 0-bids
indexed_otm_nncp = (otm_nncp.reset_index().set_index(['Calc Group', 'Term', 'CP', 'Strike', 'Calc Datetime'])
                    .sort_index(ascending=[True, True, False, True, True]))   # [['Bid', 'Ask', 'Mid']] for clarity
trimmed_nncp_list = []  # Prepare list to collect modified DFs; prob not the cleverest way
for group in indexed_otm_nncp.index.get_level_values('Calc Group').unique():
    for term in ['Near', 'Next']:
        for cp in ['P', 'C']:
            if cp == 'P':
                # OTM direction is high strike to low strike - upward
                puts = indexed_otm_nncp.loc[(group, term, cp)]
                puts_0bid = puts['Bid'] == 0
                puts_consec_0bid = puts_0bid & puts_0bid.shift(1)   # .shift(1) shifts downward
                if puts_consec_0bid.sum() == 0:
                    # Rare: no consecutive 0-bids! No trimming needed
                    trimmed = puts.copy()
                else:
                    # Perform trimming
                    # [::-1] to reverse strikes for puts
                    first_puts_consec_0bid_idx = puts_consec_0bid[::-1].idxmax()
                    # .iloc[1:] to avoid 0-bid itself
                    trimmed = puts.loc[first_puts_consec_0bid_idx:].iloc[1:].copy()
            else:
                # OTM direction is low strike to high strike - downward
                calls = indexed_otm_nncp.loc[(group, term, cp)]
                calls_0bid = calls['Bid'] == 0
                calls_consec_0bid = calls_0bid & calls_0bid.shift(-1)  # .shift(-1) shifts upward
                if calls_consec_0bid.sum() == 0:
                    # Rare: no consecutive 0-bids! No trimming needed
                    trimmed = calls.copy()
                else:
                    # Perform trimming
                    first_calls_consec_0bid_idx = calls_consec_0bid.idxmax()
                    # .iloc[:-1] to avoid 0-bid itself
                    trimmed = calls.loc[:first_calls_consec_0bid_idx].iloc[:-1].copy()
            trimmed[['Calc Group', 'Term', 'CP']] = group, term, cp
            trimmed_nncp_list.append(trimmed)
trimmed_nncp = pd.concat(trimmed_nncp_list)
trimmed_nncp = trimmed_nncp[trimmed_nncp['Bid'] != 0].copy()
trimmed_nncp = (trimmed_nncp.reset_index().set_index(['Calc Group', 'Term', 'CP', 'Strike', 'Calc Datetime'])
                .sort_index(ascending=[True, True, False, True, True]))

# Now that we have our desired data from filtering by group, re-index to see how the groups progressed over time
progression_nncp = (trimmed_nncp.reset_index().set_index(['Term', 'Expiry', 'CP', 'Strike', 'Calc Datetime'])
                    [['Bid', 'Ask', 'Mid', 't_to_exp', 'rate', 'forward', 'K_0']]
                    .sort_index(ascending=[True, True, False, True, True]))

####

# Export 1: progression of strike range
# Want range and number of strikes in puts and calls respectively for each group
# Deliverable: DataFrame + 1 plot visual
# NOTE: apparently there are times with no usable puts or calls!
strike_range_progression_list = []
for group in trimmed_nncp.index.get_level_values('Calc Group').unique():
    group_data = trimmed_nncp.loc[group]
    group_datetime = group_data.index.get_level_values('Calc Datetime').min()
    for term in ['Near', 'Next']:
        group_term_data = trimmed_nncp.loc[(group, term)]
        strikes = group_term_data.index.get_level_values('Strike')
        strike_low, strike_high, n_strikes = strikes.min(), strikes.max(), len(strikes)
        cp_grouped_strikes = group_term_data.reset_index().groupby('CP')['Strike']
        try:
            strike_low_put, strike_high_put, n_strikes_put = \
                cp_grouped_strikes.min()['P'], cp_grouped_strikes.max()['P'], cp_grouped_strikes.count()['P']
        except KeyError:
            print(f"WARNING: Group {group}, Term {term} has no puts!")
            strike_low_put, strike_high_put, n_strikes_put = None, None, None
        try:
            strike_low_call, strike_high_call, n_strikes_call = \
                cp_grouped_strikes.min()['C'], cp_grouped_strikes.max()['C'], cp_grouped_strikes.count()['C']
        except KeyError:
            print(f"WARNING: Group {group}, Term {term} has no calls!")
            strike_low_call, strike_high_call, n_strikes_call = None, None, None
        strike_range_progression_list.append((group_datetime, term,
                                              strike_low, strike_high, n_strikes,
                                              strike_low_put, strike_high_put, n_strikes_put,
                                              strike_low_call, strike_high_call, n_strikes_call))
strike_range_progression_df = pd.DataFrame(strike_range_progression_list,
                                           columns=['Datetime', 'Term',
                                                    'Strike Low', 'Strike High', '# Strikes',
                                                    'Put Strike Low', 'Put Strike High', 'Put # Strikes',
                                                    'Call Strike Low', 'Call Strike High', 'Call # Strikes'])
strike_range_progression_df = strike_range_progression_df.set_index(['Datetime', 'Term'])
strike_range_progression_df.to_csv(LOGS_DIR/f'{EVENT_NAME_STR}_strike_range_progression.csv')

# Plot near term, with VIXTLT behavior in separate subplot
near_strike_range_progression_df = strike_range_progression_df.xs('Near', level='Term')
next_strike_range_progression_df = strike_range_progression_df.xs('Next', level='Term')
# Extract VIXTLT from raw data
data_raw_0_4_split = data_raw_0_4_strip.str.split(':')
data_raw_0_4_df = pd.DataFrame(row_list for row_list in data_raw_0_4_split)
data_raw_datetimes = pd.to_datetime(data_raw_0_df[1].apply(lambda s: s[:-4]+'.'+s[-3:]))
data_raw_0_4_df['Calc Datetime'] = data_raw_datetimes
vixtlt = data_raw_0_4_df.loc[(data_raw_0_4_df[0] ==
                              'TLT.0.CBOE_VIX_QUOTE_FILTERING.BLACK.TWO.0.0.MONTHLY.0** CBOE VIX result'),
                             ['Calc Datetime', 1]]
vixtlt[1] = vixtlt[1].astype(float)
vixtlt = vixtlt.rename({1: 'VIXTLT'}, axis=1).set_index('Calc Datetime').squeeze()
# Create figure - near
_, axs = plt.subplots(2, 1, figsize=(19.2, 10.8), sharex=True)
axs[0].set_title(f'VIXTLT {EVENT_NAME_STR} Near Term Strike Range Progression')
axs[0].plot(near_strike_range_progression_df['Strike Low'], label='Lowest Strike', color='C0')
axs[0].plot(near_strike_range_progression_df['Strike High'], label='Highest Strike', color='C1')
axs[0].legend(loc=2)
axs0r = axs[0].twinx()
axs0r.bar(near_strike_range_progression_df['# Strikes'].index,
          near_strike_range_progression_df['# Strikes'],
          label='# of Strikes', color='C2', alpha=0.2, width=0.0001)
axs0r.grid(False, which='major', axis='both')
axs0r.legend(loc=1)
axs[1].plot(vixtlt, label='VIXTLT', color='C3')
axs[1].legend(loc=2)
# Create figure - next
_, axs = plt.subplots(2, 1, figsize=(19.2, 10.8), sharex=True)
axs[0].set_title(f'VIXTLT {EVENT_NAME_STR} Next Term Strike Range Progression')
axs[0].plot(next_strike_range_progression_df['Strike Low'], label='Lowest Strike', color='C0')
axs[0].plot(next_strike_range_progression_df['Strike High'], label='Highest Strike', color='C1')
axs[0].legend(loc=2)
axs0r = axs[0].twinx()
axs0r.bar(next_strike_range_progression_df['# Strikes'].index,
          next_strike_range_progression_df['# Strikes'],
          label='# of Strikes', color='C2', alpha=0.2, width=0.0001)
axs0r.grid(False, which='major', axis='both')
axs0r.legend(loc=1)
axs[1].plot(vixtlt, label='VIXTLT', color='C3')
axs[1].legend(loc=2)

# Export 2: progression of prices within each strike
# Want timeseries of prices in each (term, strike (ATM put vs. call matters, but nowhere else))
# Deliverable: DataFrame for detail + 100s of plots for visual?
progression_nncp.to_csv(LOGS_DIR/f'{EVENT_NAME_STR}_price_progression.csv')
# For each term-cp-strike, calculate volatility of prices to rank prices to look at?
# NOTE: manually check the sorted price vols for 10cent+ std, especially away from the money!
near_strike_price_vol = (progression_nncp.reset_index('Expiry').loc['Near']
                         .groupby(['CP', 'Strike'])['Mid'].std().sort_index(ascending=[False, True]))
sorted_near_strike_price_vol = near_strike_price_vol.sort_values(ascending=False)
print(f"Near Term 10+ Cent Price Vol Watchlist:\n"
      f"{sorted_near_strike_price_vol[sorted_near_strike_price_vol > 0.10]}")
next_strike_price_vol = (progression_nncp.reset_index('Expiry').loc['Next']
                         .groupby(['CP', 'Strike'])['Mid'].std().sort_index(ascending=[False, True]))
sorted_next_strike_price_vol = next_strike_price_vol.sort_values(ascending=False)
print(f"Next Term 10+ Cent Price Vol Watchlist:\n"
      f"{sorted_next_strike_price_vol[sorted_next_strike_price_vol > 0.10]}")
