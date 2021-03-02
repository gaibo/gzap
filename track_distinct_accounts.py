import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

# IBHY
ibhy = pd.read_csv(DOWNLOADS_DIR + 'IBHY_Daily_Volume_data_2021-01-26.csv',
                   parse_dates=['Month, Day, Year of Trading Dt'])
ibhy = ibhy.rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1)

# HYG, IBXXIBHY, IBY1 (1st term futures prices)
hyg = pd.read_csv(DOWNLOADS_DIR + 'HYG_bbg_2021-01-26.csv',
                  parse_dates=['Date'], index_col='Date')
ibxxibhy = pd.read_csv(DOWNLOADS_DIR + 'IBXXIBHY_bbg_2021-01-26.csv',
                       parse_dates=['Date'], index_col='Date')
iby1 = pd.read_csv(DOWNLOADS_DIR + 'IBY1_bbg_2021-01-26.csv',
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

#### Unique new accounts each day for IBHY

# Step 1: Aggregate size by date, CTI, firm, account; each row will be unique
ibhy_new = ibhy.groupby(['Trade Date', 'CTI', 'Name', 'Account '])['Size'].sum()

# Step 2: Generate "known" set of accounts for each day
ibhy_days = ibhy_new.index.get_level_values('Trade Date').unique()
known_set_dict = {ibhy_days[0]: set()}  # No known accounts on first day
for day, prev_day in zip(ibhy_days[1:], ibhy_days[:-1]):
    known_set_dict[day] = \
        (known_set_dict[prev_day]
         | set(ibhy_new.loc[prev_day].index.get_level_values('Account ')))

# Step 3: Mark accounts each day that were not known at the beginning of the day
ibhy_new_reset = ibhy_new.reset_index('Account ')
for day in ibhy_days:
    # .values is great for doing stuff with no unique indexes
    ibhy_new_reset.loc[day, 'New Account'] = \
        (~ibhy_new_reset.loc[day]['Account '].isin(known_set_dict[day])).values

# NOTE: Account names SHOULD be unique across firms, and each account SHOULD only trade in one CTI.
#       HOWEVER, there are examples of firms using wrong CTI and having to make correction,
#       so code should ideally be able to mark this.
#       If an account on its debut day trades in 2 CTIs (very specific, rare), then new account rows will be 1 too many.
# NOTE: Account names are absolutely not unique across trading sessions. In fact, it is useless trying
#       to track "new accounts by session". For GTH analysis, stick to active accounts/volume per session.

# Step 4: Aggregate for final results
ibhy_new_accounts = ibhy_new_reset.groupby(['Trade Date', 'CTI'])['New Account'].sum().unstack()
# Use .fillna(method='ffill') on cumsum to fill NaNs, but NaNs actually give you info on what days the CTI had volume
ibhy_new_accounts_cumsum = ibhy_new_accounts.cumsum()
# Exportable
ibhy_new_accounts_export = ibhy_new_reset

#### Okay let's functionize that

def evaluate_new_accounts(data):
    # Step 0: Check if any accounts switch CTI
    ctiswitch_set = set(data[data['CTI'] == 4]['Account ']) & set(data[data['CTI'] != 4]['Account '])
    if len(ctiswitch_set) != 0:
        print(f"Accounts that switch CTI: {ctiswitch_set}")

    # Step 1: Aggregate size by date, CTI, firm, account; each row will be unique
    data_agg = data.groupby(['Trade Date', 'CTI', 'Name', 'Account '])['Size'].sum()

    # Step 2: Generate "known" set of accounts for each day
    data_days = data_agg.index.get_level_values('Trade Date').unique()
    known_set_dict = {data_days[0]: set()}  # No known accounts on first day
    for day, prev_day in zip(data_days[1:], data_days[:-1]):
        known_set_dict[day] = \
            (known_set_dict[prev_day]
             | set(data_agg.loc[prev_day].index.get_level_values('Account ')))

    # Step 3: Mark accounts each day that were not known at the beginning of the day
    data_agg_reset = data_agg.reset_index('Account ')
    for day in data_days:
        # .values is great for doing stuff with no unique indexes
        data_agg_reset.loc[day, 'New Account'] = \
            (~data_agg_reset.loc[day]['Account '].isin(known_set_dict[day])).values
    return data_agg_reset


# def evaluate_volumes(data):
#     # Step 0: Check if any accounts switch CTI
#     ctiswitch_set = set(data[data['CTI'] == 4]['Account ']) & set(data[data['CTI'] != 4]['Account '])
#     if len(ctiswitch_set) != 0:
#         print(f"Accounts that switch CTI: {ctiswitch_set}")
#
#     # Step 1: Aggregate size by date, CTI, firm, account; each row will be unique
#     data_agg = data.groupby(['Trade Date', 'CTI', 'Name', 'Account '])['Size'].sum()
#
#     # Step 2: Generate "known" set of accounts for each day
#     data_days = data_agg.index.get_level_values('Trade Date').unique()
#     known_set_dict = {data_days[0]: set()}  # No known accounts on first day
#     for day, prev_day in zip(data_days[1:], data_days[:-1]):
#         known_set_dict[day] = \
#             (known_set_dict[prev_day]
#              | set(data_agg.loc[prev_day].index.get_level_values('Account ')))
#
#     # Step 3: Mark accounts each day that were not known at the beginning of the day
#     data_agg_reset = data_agg.reset_index('Account ')
#     for day in data_days:
#         # .values is great for doing stuff with no unique indexes
#         data_agg_reset.loc[day, 'New Account'] = \
#             (~data_agg_reset.loc[day]['Account '].isin(known_set_dict[day])).values
#     return data_agg_reset


# Try it on Mini VIX
# 1) Load
vxm = (pd.read_csv(DOWNLOADS_DIR + 'VXM_Daily_Volume_data_2021-03-02.csv',
                   parse_dates=['Month, Day, Year of Trading Dt'])
       .rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1))
# 2) Run function
vxm_new_accounts = evaluate_new_accounts(vxm)
# 3) Aggregate for Python visuals (step 2 already makes Excel-pivot-ready sheet)
vxm_new_accounts_CTI = vxm_new_accounts.groupby(['Trade Date', 'CTI'])['New Account'].sum().unstack()
vxm_new_accounts_cumsum = vxm_new_accounts_CTI.cumsum()
# 4) Export
vxm_new_accounts.to_csv(DOWNLOADS_DIR + 'vxm_new_accounts_2021-03-02.csv')

# Try it on IBIG
# 1) Load
ibig = (pd.read_csv(DOWNLOADS_DIR + 'IBIG_Daily_Volume_data_2021-03-02.csv',
                    parse_dates=['Month, Day, Year of Trading Dt'])
        .rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1))
# 2) Run function
ibig_new_accounts = evaluate_new_accounts(ibig)
# 3) Aggregate for Python visuals (step 2 already makes Excel-pivot-ready sheet)
ibig_new_accounts_CTI = ibig_new_accounts.groupby(['Trade Date', 'CTI'])['New Account'].sum().unstack()
ibig_new_accounts_cumsum = ibig_new_accounts_CTI.cumsum()
# 4) Export
ibig_new_accounts.to_csv(DOWNLOADS_DIR + 'ibig_new_accounts_2021-03-02.csv')

# Try it on IBHY
# 1) Load
ibhy = (pd.read_csv(DOWNLOADS_DIR + 'IBHY_Daily_Volume_data_2021-03-02.csv',
                    parse_dates=['Month, Day, Year of Trading Dt'])
        .rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1))
# 2) Run function
ibhy_new_accounts = evaluate_new_accounts(ibhy)
# 3) Aggregate for Python visuals (step 2 already makes Excel-pivot-ready sheet)
ibhy_new_accounts_CTI = ibhy_new_accounts.groupby(['Trade Date', 'CTI'])['New Account'].sum().unstack()
ibhy_new_accounts_cumsum = ibhy_new_accounts_CTI.cumsum()
# 4) Export
ibhy_new_accounts.to_csv(DOWNLOADS_DIR + 'ibhy_new_accounts_2021-03-02.csv')

# Try it on big VIX
# 1) Load
vx = (pd.read_csv(DOWNLOADS_DIR + 'VX_Daily_Volume_data_2021-01-29.csv',
                  parse_dates=['Month, Day, Year of Trading Dt'])
      .rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1))
# 2) Run function
vx_new_accounts = evaluate_new_accounts(vx)
# 3) Aggregate for Python visuals (step 2 already makes Excel-pivot-ready sheet)
vx_new_accounts_CTI = vx_new_accounts.groupby(['Trade Date', 'CTI'])['New Account'].sum().unstack()
vx_new_accounts_cumsum = vx_new_accounts_CTI.cumsum()
