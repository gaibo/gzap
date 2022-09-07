import pandas as pd
from pathlib import Path

DOWNLOADS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/')

# Load raw volume data
data = pd.read_csv(DOWNLOADS_DIR / 'Account_Daily_BS_data.csv',
                   dtype={'CTI': int, 'Account': str, 'Expiry': str, 'Side': str, 'Trade Date': str,
                          'AMV': int, 'Size': int},
                   thousands=',',
                   parse_dates=['Expiry', 'Trade Date'])

bs_both_check = data.groupby(['CTI', 'Account', 'Trade Date', 'Expiry'])['Size'].count()    # Both B and S
account_day_expiry_min = data.groupby(['CTI', 'Account', 'Trade Date', 'Expiry'])['Size'].min()     # Matched B-S
account_day_expiry_join = account_day_expiry_min.to_frame('Min').join(bs_both_check.to_frame('BS'))
account_day_expiry = account_day_expiry_join.loc[account_day_expiry_join['BS'] == 2, 'Min']

account_day = account_day_expiry.groupby(['CTI', 'Account', 'Trade Date']).sum()

account_day_sum = account_day.groupby(['CTI', 'Account']).sum()
account_day_count = account_day.groupby(['CTI', 'Account']).count()
account_agg = account_day_sum.to_frame('Size').join(account_day_count.to_frame('Count'))
top_accounts = account_agg.sort_values('Size', ascending=False)     # Sanity check - 1071 total trade dates

top_test_days = account_day.xs('690133583', level='Account').groupby('Trade Date').sum()
top_test_months = top_test_days.groupby([lambda x: x.year, lambda x: x.month]).sum()    # Pretty damn close to official
top_test_months_double = top_test_months * 2    # Day Trading counts both sides
top_test_volume = top_test_months_double - 20_000

# CTI 4 $1.40, else $1 or $0.90 depending on TPH status (do we want to dig that deep?)

# Load in the account context information
burr = pd.read_csv(DOWNLOADS_DIR / 'AMV_Big_List_data.csv')
account_context_dupe = burr[burr.columns[:7]].drop_duplicates()  # Account may have duplicates - change over time
account_context = burr[burr.columns[:7]].drop_duplicates(['Account']).set_index('Account')

# Do account-by-account evaluation first (rebate in old system vs. rebate in new system); % change in volume
# NOTE: need to record for each account for each CTI (4 or not) their volume
top_50_accounts = top_accounts.iloc[:50].index.get_level_values('Account')
all_accounts = data['Account'].unique()

account_cti_monthly = account_day.groupby(['Account', 'CTI', lambda x: x[2].year, lambda x: x[2].month]).sum()
account_cti_monthly.index = account_cti_monthly.index.rename(['Account', 'CTI', 'Year', 'Month'])
account_cti_mean_monthly = account_cti_monthly.groupby(['Account', 'CTI']).mean()
account_cti_max_monthly = account_cti_monthly.groupby(['Account', 'CTI']).max()
account_cti_best_list = account_cti_max_monthly.sort_values(ascending=False)    # Only 77 would have EVER qualified
# Let's take top 100
top_100_accounts = account_cti_best_list.index.get_level_values('Account')[:100]
regrouped = account_cti_monthly.groupby(['Account', 'Year', 'Month']).sum()
reshaped = regrouped.unstack([1, 2])
top_reshaped = reshaped.reindex(top_100_accounts)
top_reshaped = top_reshaped[top_reshaped.columns.sort_values()]
# Join in account context
final = top_reshaped.join(account_context).set_index(['Port Owner', 'Member', 'Member Name',
                                                      'OCC ID', 'Clearing Member', 'EFID'], append=True)
final.columns = top_reshaped.columns    # Reformat destroyed Multiindex
# Still need CTI (put into context), and these lost the sorting!

# Then link top accounts to info so we know who gets affected

####

# Load in Thomas Mueller's spreadsheet for official rebate numbers
# foo = pd.read_excel(DOWNLOADS_DIR / 'cfe_day_trading_invoiced_since_2018_02 v5.xlsx')
