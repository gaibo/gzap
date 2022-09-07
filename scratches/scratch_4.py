import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load data from Historical Daily Volume Breakdown
DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
# NOTE: purposefully do not parse dates to save time since it will end up in Excel
vx_master = pd.read_csv(DOWNLOADS_DIR+'VX_Historical_Daily_Volume_Breakdown_data_2021-04-28.csv')
# Raw data is 1.5MM rows too large for Excel, so must distill via groupby().
# Aggregating Size as a "value" is a given and fairly straightforward, but it is not so easy to distill Accounts.
# Each trade comes with an account and the user firm that the account is trading on behalf of. However, note that each
# account can trade on behalf of many users, and each user can trade through many accounts. This prohibits hierarchy
# between Account and User Name, i.e. if we groupby User Name and get # Accounts, we must also consider groupby Account
# and get # Users.
# Another point is that while each Account is theoretically matched to just one CTI (in reality, there are some errors),
# each Account can trade in multiple Sessions. Thus, Sessions is incompatible for grouping Users/Accounts.
# Trade Date, CTI, Product Type, Complex are all valid for indexing; Session, Account, Operator, User,
#
# and 2) # accounts;
# those 2 data points for each grouping is enough to answer most questions, but
# we lose data on
nunique_test = vx_master.groupby(['Trade Date', 'User Name', 'Session', 'CTI', 'Product Type', 'Complex'])['Account'].nunique()

# Plot - Various Window Rolls
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('GTH Share of VIX Futures Volume')
ax.plot(gth_share, label='GTH Share')
ax.plot(gth_share.rolling(21, center=True).mean(), label='Rolling 1-Month Mean')
axr = ax.twinx()
axr.plot(volume_view.rolling(21, center=True).mean()/2, label='Rolling 1-Month VX Volume', color='C2', alpha=0.5)
ax.legend(loc=2)
axr.legend(loc=1)
axr.grid(False)
axr.set_yticklabels(['{:,}'.format(int(x)) for x in axr.get_yticks().tolist()])

# Plot - Various Window Rolls
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('RTH: CTI 4 Share of VIX Futures Volume')
ax.plot(cti_4_rth/2, label='CTI 4 Volume')
ax.plot((cti_4_rth/2).rolling(21, center=True).mean(), label='Rolling 1-Month Mean')
axr = ax.twinx()
axr.plot((cti_4_rth/(rth_share*volume_view)).rolling(21, center=True).mean(), label='Rolling 1-Month CTI 4 Share', color='C2', alpha=0.5)
# axr.plot(volume_view.rolling(21, center=True).mean()/2, label='Rolling 1-Month VX Volume', color='C2', alpha=0.5)
ax.legend(loc=2)
axr.legend(loc=1)
axr.grid(False)
ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('GTH: CTI 4 Share of VIX Futures Volume')
ax.plot(cti_4_gth/2, label='CTI 4 Volume')
ax.plot((cti_4_gth/2).rolling(21, center=True).mean(), label='Rolling 1-Month Mean')
axr = ax.twinx()
axr.plot((cti_4_gth/(gth_share*volume_view)).rolling(21, center=True).mean(), label='Rolling 1-Month CTI 4 Share', color='C2', alpha=0.5)
# axr.plot(volume_view.rolling(21, center=True).mean()/2, label='Rolling 1-Month VX Volume', color='C2', alpha=0.5)
ax.legend(loc=2)
axr.legend(loc=1)
axr.grid(False)
ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])

for n in [1, 3, 6]:
    ax.plot(ibhy1_hyg_corr_dict[n],
            label=f'{n}-Month Rolling Correlation')
overall_corr = pd.DataFrame({'IBHY': ibhy1_change, 'HYG': hyg_change}).corr().iloc[1, 0]
ax.axhline(overall_corr, label=f'Overall Correlation ({overall_corr*100:.1f}%)',
           color='k', linestyle='--')
ax.legend()
