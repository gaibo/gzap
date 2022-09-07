import pandas as pd
from pathlib import Path

DOWNLOADS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/')

# Load raw volume data for Day Trading
data = pd.read_csv(DOWNLOADS_DIR / 'Account_Daily_BS_(Full)_data_FULL_2022-06.csv',
                   dtype={'CTI': int, 'Account': str, 'Expiry': str, 'Side': str, 'Trade Date': str,
                          'AMV': int, 'Size': int},
                   thousands=',',
                   parse_dates=['Expiry', 'Trade Date'])
data['Customer'] = (data['CTI'] == 4)   # We generally only care about customer RPC vs non-customer RPC

# Load data for Tier Rebate
data_tier = pd.read_csv(DOWNLOADS_DIR / 'VX_Historical_Daily_Volume_Breakdown_data_2022-06-30.csv',
                        parse_dates=['Trade Date'])
data_tier = data_tier.rename({'User': 'Member', 'User Name': 'Member Name', 'Clearing EFID': 'EFID'}, axis=1)
data_tier['Customer'] = (data_tier['CTI'] == 4)

################################################################################

# Filter to Day Trading volume
# Day Trading specs: must be B and S on same day
account_by_bs = data.groupby(['Account', 'CTI', 'Trade Date', 'Expiry', 'Side'])['Size'].sum()
bs_both_count = account_by_bs.groupby(['Account', 'CTI', 'Trade Date', 'Expiry']).count()   # Count 2 = B+S
account_cti_day_expiry_min = account_by_bs.groupby(['Account', 'CTI', 'Trade Date', 'Expiry']).min()    # Matched B-S
account_cti_day_expiry_join = account_cti_day_expiry_min.to_frame('Min').join(bs_both_count.to_frame('BS'))
account_cti_day_expiry = account_cti_day_expiry_join.loc[account_cti_day_expiry_join['BS'] == 2, 'Min'] * 2  # 2x for BS

# CTI 4 $1.40, else $1 or $0.90 depending on TPH status (do we want to dig that deep?)

# Load in the account context information
account_context_dupe = data[data.columns[:8]].drop_duplicates().set_index('Account')    # Duplicates = change over time
account_context = data[data.columns[:8]].drop_duplicates(['Account'], keep='last').set_index('Account')  # No dupe

# Do account-by-account evaluation first (rebate in old system vs. rebate in new system); % change in volume
# NOTE: need to record for each account for each CTI (4 or not) their volume
account_cti_monthly = account_cti_day_expiry.groupby(['Account', 'CTI',
                                                      lambda x: x[2].year, lambda x: x[2].month]).sum()
account_cti_monthly.index = account_cti_monthly.index.rename(['Account', 'CTI', 'Year', 'Month'])
account_cti_mean_monthly = account_cti_monthly.groupby(['Account', 'CTI']).mean()
account_cti_max_monthly = account_cti_monthly.groupby(['Account', 'CTI']).max()
account_cti_best_list = account_cti_max_monthly.sort_values(ascending=False)    # Only 108 would have EVER qualified

# Separate CTI 4 accounts from MM accounts and sort them descending
clone = account_cti_monthly.to_frame('Size')
clone['Customer'] = (clone.index.get_level_values('CTI') == 4)
account_monthly = clone.groupby(['Customer', 'Account', 'Year', 'Month'])['Size'].sum()
customer_account_monthly = account_monthly.xs(True)
mm_account_monthly = account_monthly.xs(False)
customer_reshaped = customer_account_monthly.unstack([1, 2])
customer_reshaped = customer_reshaped[customer_reshaped.columns.sort_values()]
mm_reshaped = mm_account_monthly.unstack([1, 2])
mm_reshaped = mm_reshaped[mm_reshaped.columns.sort_values()]
customer_reshaped = customer_reshaped.reindex(customer_reshaped.max(axis=1).sort_values(ascending=False).index)
mm_reshaped = mm_reshaped.reindex(mm_reshaped.max(axis=1).sort_values(ascending=False).index)

# Calculate
pass    # Done manually in Excel

################################################################################

# Separately, evaluate rebate based on Tier volume - need 1) ADV and 2) % of monthly total
volume = data_tier.groupby(['Customer', 'Member', 'Member Name', 'EFID', 'Trade Date'])['Size'].sum()
mm_volume = volume.xs(False)
mm_volume_monthly = \
    mm_volume.groupby(['Member', 'Member Name', 'EFID', lambda x: x[3].year, lambda x: x[3].month]).mean()
mm_volume_reshaped = mm_volume_monthly.unstack([3, 4])
mm_volume_reshaped = mm_volume_reshaped[mm_volume_reshaped.columns.sort_values()]
mm_volume_reshaped = mm_volume_reshaped.reindex(mm_volume_reshaped.max(axis=1).sort_values(ascending=False).index)

################################################################################

# Load in Thomas Mueller's spreadsheet for official rebate numbers
# foo = pd.read_excel(DOWNLOADS_DIR / 'cfe_day_trading_invoiced_since_2018_02 v5.xlsx')
