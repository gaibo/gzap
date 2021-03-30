import pandas as pd
import matplotlib.pyplot as plt
from options_futures_expirations_v3 import days_between
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'


# General workflow:
# 1) Go to https://bi.cboe.com/#/views/ModifiedDailyiBoxxFuturesVolumeforDataPulling/iBoxxDailyVolume?:iid=1
#    and Download->Data->Summary tab->Download all rows as a text file.
#    Save it in DOWNLOADS_DIR as f'{product}_Daily_Volume_data_{USE_DATE}.csv'.
# 2) Run this script and check DOWNLOADS_DIR for f'{product}_new_accounts_{USE_DATE}.csv'.
# 3) Open output file, filter to new dates, copy and paste them into f'{product}_new_accounts_{USE_DATE}.xlsx',
#    which is a maintained Excel Pivot Table.

# NOTE: Account names SHOULD be unique across firms, and each account SHOULD only trade in one CTI.
#       HOWEVER, there are examples of firms using wrong CTI and having to make correction,
#       so code should ideally be able to mark this.
#       If an account on its debut day trades in 2 CTIs (very specific, rare), then new account rows will be 1 too many.
# NOTE: Account names are absolutely not unique across trading sessions. In fact, it is useless trying
#       to track "new accounts by session". For GTH analysis, stick to active accounts/volume per session.
def evaluate_new_accounts(data):
    """ Track new accounts through DataFrame history and return separately-formatted DataFrame
    :param data: DataFrame with columns:
                 ['Trade Date', 'Trading Session', 'CTI', 'Account', 'Operator Id', 'Name', 'Complex Trade', 'Size']
    :return: DataFrame with columns:
             ['Trade Date', 'CTI', 'Name', 'Account', 'Size', 'New Account']
    """
    # Step 0: Check if any accounts switch CTI
    ctiswitch_set = set(data[data['CTI'] == 4]['Account ']) & set(data[data['CTI'] != 4]['Account '])
    if len(ctiswitch_set) != 0:
        print(f"Accounts that switch CTI: {ctiswitch_set}")

    # Step 1: Aggregate size by date, CTI, firm, account; each row will be unique
    data_agg = data.groupby(['Trade Date', 'CTI', 'Name', 'Account '])['Size'].sum()

    # Step 2: Generate "known" set of accounts for each day
    # Step 2+: Track join date for each account
    data_days = data_agg.index.get_level_values('Trade Date').unique()
    known_set_dict = {data_days[0]: set()}  # No known accounts on first day
    account_join_dict = {}
    for day, prev_day in zip(data_days[1:], data_days[:-1]):
        prev_day_account_set = set(data_agg.loc[prev_day].index.get_level_values('Account '))
        known_set_dict[day] = known_set_dict[prev_day] | prev_day_account_set
        prev_day_account_join_dict = dict.fromkeys(prev_day_account_set, prev_day)
        account_join_dict = {**prev_day_account_join_dict, **account_join_dict}  # Order matters - main dict overwrites
    # Need an extra iteration for final day
    final_day_account_set = set(data_agg.loc[data_days[-1]].index.get_level_values('Account '))
    final_day_account_join_dict = dict.fromkeys(final_day_account_set, data_days[-1])
    account_join_dict = {**final_day_account_join_dict, **account_join_dict}

    # Step 3: Mark accounts each day that were not known at the beginning of the day
    data_agg_reset = data_agg.reset_index('Account ')
    for day in data_days:
        # .values is great for doing stuff with no unique indexes
        data_agg_reset.loc[day, 'New Account'] = \
            (~data_agg_reset.loc[day]['Account '].isin(known_set_dict[day])).values

    # Step 4: Add account join date and trade-days-since-join context
    data_agg_full_reset = data_agg_reset.reset_index()
    def account_join_apply_helper(row):
        join_date = account_join_dict[row['Account ']]
        if row['Trade Date'] < join_date:
            days_since_join = None
        else:
            days_since_join = days_between(row['Trade Date'], join_date, use_busdays=True)
        return join_date, days_since_join
    tuple_result = data_agg_full_reset.apply(account_join_apply_helper, axis=1)
    concat_df = pd.DataFrame(tuple_result.to_list(),
                             columns=['Account Join Date', 'Trade Days Since Join'])
    data_agg_reindexed = (pd.concat([data_agg_full_reset, concat_df], axis=1)
                          .set_index(['Trade Date', 'CTI', 'Name']))

    return data_agg_reindexed


# -----------------------------------------------------------------------------

# [CONFIGURE] Setup
USE_DATE = '2021-03-29'
PRODUCTS = ['VXM', 'IBHY', 'IBIG']  # Default ['VXM', 'IBHY', 'IBIG']
# Initialize storage
data_dict = {}
new_accounts_dict = {}
new_accounts_CTI_dict = {}
new_accounts_cumsum_dict = {}

# All product workflow
for product in PRODUCTS:
    # 1) Load
    data_dict[product] = \
        (pd.read_csv(DOWNLOADS_DIR + f'{product}_Historical_Daily_Volume_data_{USE_DATE}.csv',
                     parse_dates=['Month, Day, Year of Trading Dt'])
         .rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1))
    print(f"{product} loaded.")
    # 2) Run function
    new_accounts_dict[product] = evaluate_new_accounts(data_dict[product])
    new_accounts_dict[product].to_csv(DOWNLOADS_DIR + f'{USE_DATE}_{product}_new_accounts.csv')  # Export!
    # 3) Aggregate for Python visuals (step 2 already makes Excel-pivot-ready sheet)
    new_accounts_CTI_dict[product] = \
        new_accounts_dict[product].groupby(['Trade Date', 'CTI'])['New Account'].sum().unstack()
    new_accounts_cumsum_dict[product] = new_accounts_CTI_dict[product].cumsum()
