import pandas as pd
from pathlib import Path
from options_data_tools import add_t_to_exp, add_rate, add_forward, lookup_val_in_col

LOGS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Gaibo/VIXTLT/VIXTLTP Log Pulls/')
FILE_NAME_STR = 'VXTLTPE_2021_09_20_10am_11am.log'  # CHANGE MANUALLY

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
# NOTE: options_data_tools was written for doing one calc per day (at close), so needed date-only column,
#       as indexing behind the scenes is screwed up by precision of datetime
nncp['Calc Group'] = (nncp['Calc Datetime'].diff() > pd.Timedelta(seconds=5)).cumsum() + 1  # +1 so 1st group is 1
nncp['Calc Date'] = nncp['Calc Datetime'].dt.normalize()
# add_forward() usage complicated by intraday - collect unique forward for each group and each term/expiry within
nncp = nncp.set_index('Calc Group')
nncp_forwards = nncp.copy()
for group in nncp.index.unique():
    nncp_group_forward = add_forward(nncp.loc[group], 'Calc Date', 'Expiry', 'Strike', 'CP',
                                     'Mid', 't_to_exp', 'rate', 'forward')
    nncp_forwards.loc[group, 'forward'] = nncp_group_forward['forward'].values
unique_forwards = nncp_forwards.groupby(['Calc Group', 'Expiry'])['forward'].first()
nncp_forwards = nncp_forwards.set_index('Expiry', append=True)
for group_exp_idx in unique_forwards.index:
    unindexed_data_slice = nncp_forwards.loc[group_exp_idx].reset_index()
    forward = unique_forwards.loc[group_exp_idx]
    nncp_forwards.loc[group_exp_idx, 'K_0'] = \
        lookup_val_in_col(unindexed_data_slice, forward, 'Strike', leq_only=True)['Strike']

# Use K_0 to select only OTM
nncp_otm = (nncp_forwards.loc[((nncp_forwards['CP'] == 'C') & (nncp_forwards['Strike'] >= nncp_forwards['K_0']))
                              | ((nncp_forwards['CP'] == 'P') & (nncp_forwards['Strike'] <= nncp_forwards['K_0']))]
            .copy())

# Trim 0-bids to focus on meaningful data
# Note that we can cleverly spot consecutive 0-bids by using Boolean & with next OTM
nncp_otm_clean = (nncp_otm.reset_index().set_index(['Calc Group', 'Term', 'CP', 'Strike', 'Calc Datetime'])
                  .sort_index(ascending=[True, True, False, True, True])[['Bid', 'Ask', 'Mid']])
nncp_trim_list = []
for group in range(1, 241):
    for term in ['Near', 'Next']:
        for cp in ['P', 'C']:
            if cp == 'P':
                # OTM direction is high strike to low strike - upward
                puts = nncp_otm_clean.loc[(group, term, cp)]
                puts_0bid = puts['Bid'] == 0
                puts_consec_0bid = puts_0bid & puts_0bid.shift(1)   # .shift(1) shifts downward
                first_puts_consec_0bid_idx = puts_consec_0bid[::-1].idxmax()     # [::-1] to reverse strikes for puts
                trimmed = puts.loc[first_puts_consec_0bid_idx:].iloc[1:].copy()     # .iloc[1:] to avoid 0-bid itself
            else:
                # OTM direction is low strike to high strike - downward
                calls = nncp_otm_clean.loc[(group, term, cp)]
                calls_0bid = calls['Bid'] == 0
                calls_consec_0bid = calls_0bid & calls_0bid.shift(-1)  # .shift(-1) shifts upward
                first_calls_consec_0bid_idx = calls_consec_0bid.idxmax()
                trimmed = calls.loc[:first_calls_consec_0bid_idx].iloc[:-1].copy()  # .iloc[:-1] to avoid 0-bid itself
            trimmed[['Calc Group', 'Term', 'CP']] = group, term, cp
            nncp_trim_list.append(trimmed)
nncp_trim = pd.concat(nncp_trim_list)
nncp_trim = (nncp_trim.reset_index().set_index(['Calc Group', 'Term', 'CP', 'Strike', 'Calc Datetime'])
             .sort_index(ascending=[True, True, False, True, True]))
