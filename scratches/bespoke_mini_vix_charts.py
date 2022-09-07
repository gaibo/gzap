import pandas as pd
import matplotlib.pyplot as plt

# Testing out usage of custom FiveThirtyEight
plt.style.use('cboe-fivethirtyeight')

DATA_DIR_STABILIS = 'Y:/Research/Research1/Gaibo/Stabilis Starter Pack/Data/'
DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
# retail_file_name = 'Customer_Accounts_by_Firm_Full_Data_Edited.csv'
# vxm_file_name = 'Volume_Summary_Full_Data_edited.csv'
retail_file_name = 'Customer_Accounts_by_Firm_Full_Data_data (1).csv'
vxm_file_name = 'Volume_Summary_Full_Data_data (3).csv'

# Read data
data = pd.read_csv(DOWNLOADS_DIR+retail_file_name)
# FCST "behalf" used for both INTL FC Stone and StoneX lmao
# "Firm Id" is not "behalf"... e.g. CQCI whatever that is covers 9 firms
subset = data[['Name', 'Trading Dt', 'Trading Session', 'Distinct Accounts', 'Size', 'Account']].copy()
subset['Trading Dt'] = pd.to_datetime(subset['Trading Dt'])
size_ser = subset.groupby('Trading Dt')['Size'].sum()
# accounts_ser = subset.groupby('Trading Dt')['Distinct Accounts'].sum()
accounts_ser = subset.groupby('Trading Dt')['Account'].apply(lambda df: len(df.unique()))
accounts_ser.index = pd.to_datetime(accounts_ser.index)
accounts_ser = accounts_ser.sort_index()

# Read total VXM data
vxm_data = pd.read_csv(DOWNLOADS_DIR+vxm_file_name, parse_dates=[[0,1,2]])
vxm_data = vxm_data.rename({vxm_data.columns[0]: 'Trading Dt'}, axis=1)
vxm_data = vxm_data.drop('Futures Root', axis=1)
vxm_size_ser = vxm_data.groupby('Trading Dt')['Size'].sum()

# # Plot
# fig, ax = plt.subplots(figsize=(19.2, 10.8))
# ax.plot(size_ser, label='Retail Trade Size')
# ax.plot(accounts_ser, label='Unique Accounts Trading')
# axr = ax.twinx()
# axr.plot(size_ser/vxm_size_ser * 100, color='C2', label='Retail as Percent of Total')
# axr.grid(False, which='major', axis='both')
# axr.set_ylabel('Percent of Total VXM Size (%)')
# ax.set_xlabel('Date')
# ax.set_title('Retail Trading of Mini-VIX Futures')
# ax.legend(loc=2)
# axr.legend(loc=1)

# Plot version 2
fig, axs = plt.subplots(2, 1, sharex='all', figsize=(19.2, 10.8))
axs[0].plot(size_ser, color='C0', label='Retail Trade Size')
axs0r = axs[0].twinx()
axs0r.plot(size_ser/vxm_size_ser * 100, color='C2', label='Retail as Percent of Total')
axs0r.grid(False, which='major', axis='both')
axs0r.set_ylabel('Percent of Total VXM Size (%)')
axs[1].plot(accounts_ser, color='C1', label='Unique Accounts Trading')
axs[1].set_xlabel('Date')
axs[0].set_title('Retail Trading of Mini-VIX Futures')
axs[0].legend(loc=2)
axs0r.legend(loc=4)     # Bespoke to not get in the way
axs[1].legend(loc=2)
fig.set_tight_layout(True)

# # Plot of full VXM trading
# fig, ax = plt.subplots(figsize=(19.2, 10.8))
# ax.plot(vxm_size_ser, color='C4', label='Total VXM Trade Size (2xVolume)')
# ax.legend()
# ax.set_xlabel('Date')
# ax.set_title('Trend of Total VXM Size')

# Plot version 3 - bars with unique accounts
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.bar(size_ser.index, size_ser, color='C0', label='Retail Trade Size')
ax.set_yticklabels([f'{int(x):,}' for x in ax.get_yticks().tolist()])
ax.legend(loc=2)
ax.set_title('Retail Trading of Mini-VIX Futures')
axr = ax.twinx()
# axs0r.plot(size_ser/vxm_size_ser * 100, color='C2', label='Retail as Percent of Total')
# axr.set_ylabel('Percent of Total VXM Size (%)')
axr.plot(accounts_ser, color='C1', label='Unique Accounts Trading')
axr.grid(False, which='major', axis='both')
axr.legend(loc=1)
# axs0r.legend(loc=4)     # Bespoke to not get in the way
# axs[1].legend(loc=2)
fig.set_tight_layout(True)
