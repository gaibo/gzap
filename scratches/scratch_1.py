import pandas as pd

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

# Load horrible makeshift iBoxx OI because inception-2018-11-07 is not in CFEVOLOI lmao
oi_file_name = 'Current_Open_Interest_and_Volume_by_Futures_Root_F_data (3).csv'  # [CONFIGURE]
oi_data = pd.read_csv(DOWNLOADS_DIR+oi_file_name, index_col='Trading Date', parse_dates=True).sort_index()
ibhy_oi_data = oi_data[oi_data['Product'] == 'IBHY']
ibhy_oi = ibhy_oi_data.groupby('Trading Date')['Current Open Interest'].sum()
ibig_oi_data = oi_data[oi_data['Product'] == 'IBIG']
ibig_oi = ibig_oi_data.groupby('Trading Date')['Current Open Interest'].sum()
vx_oi_data = oi_data[oi_data['Product'] == 'VX']
vx_oi = vx_oi_data.groupby('Trading Date')['Current Open Interest'].sum()
vxm_oi_data = oi_data[oi_data['Product'] == 'VXM']
vxm_oi = vxm_oi_data.groupby('Trading Date')['Current Open Interest'].sum()

#### iBoxx Volumes Breakdown ##################################################

# Load data
# IBHY
ibhy_file_name = 'IBHY_Daily_Volume_data_2021-01-29.csv'     # [CONFIGURE]
ibhy_data = (pd.read_csv(DOWNLOADS_DIR+ibhy_file_name, parse_dates=['Month, Day, Year of Trading Dt'])
             .rename({'Month, Day, Year of Trading Dt': 'Trading Date'}, axis=1))
# IBIG
ibig_file_name = 'IBIG_Daily_Volume_data_2021-01-29.csv'     # [CONFIGURE]
ibig_data = (pd.read_csv(DOWNLOADS_DIR+ibig_file_name, parse_dates=['Month, Day, Year of Trading Dt'])
             .rename({'Month, Day, Year of Trading Dt': 'Trading Date'}, axis=1))

# Process IBHY
ibhy_size = ibhy_data.set_index(['Trading Date', 'CTI', 'Name']).sort_index()
ibhy_day_cti_volume = ibhy_size.groupby(['Trading Date', 'CTI'])['Size'].sum()/2
ibhy_days = ibhy_day_cti_volume.index.get_level_values(0).unique()
ibhy_volume_breakdown = pd.DataFrame(index=ibhy_days,
                                     columns=['CTI 1', 'CTI 2', 'CTI 4', 'Total'])
ibhy_volume_breakdown['CTI 1'] = ibhy_day_cti_volume.xs(1, level='CTI')
ibhy_volume_breakdown['CTI 2'] = ibhy_day_cti_volume.xs(2, level='CTI')
ibhy_volume_breakdown['CTI 4'] = ibhy_day_cti_volume.xs(4, level='CTI')
ibhy_volume_breakdown['Total'] = ibhy_volume_breakdown[['CTI 1', 'CTI 2', 'CTI 4']].sum(axis=1)
ibhy_volume_breakdown.index.name = 'Trading Date'
# Open interest separately
ibhy_volume_breakdown['Open Interest'] = ibhy_oi    # Can't use cfevoloi['IBHY OI'] because missing data
# ibhy_volume_breakdown = ibhy_volume_breakdown.drop(pd.Timestamp('2021-01-21'))

# Aggregate by month (must do manually because # days varies)
unique_yearmonths = ibhy_volume_breakdown.index.strftime('%Y-%m').unique()
# Volumes
volume_sums = pd.DataFrame(columns=['CTI 1', 'CTI 2', 'CTI 4', 'Total'])
for yearmonth in unique_yearmonths:
    volume_sums.loc[yearmonth] = ibhy_volume_breakdown.loc[yearmonth, ['CTI 1', 'CTI 2', 'CTI 4', 'Total']].sum()

# Break down by firms
ibhy_firm = ibhy_size.reset_index()
ibhy_unique_firms = ibhy_firm.groupby(['Trading Date', 'CTI'])['Name'].nunique()
unique_customers_daily = ibhy_unique_firms.xs(4, level='CTI')
ibhy_monthly = pd.DataFrame(columns=['MM Volume', 'Customer Volume', 'Total Volume', 'Unique Customers'])
ibhy_firm_indexed = ibhy_firm.set_index('Trading Date')
for yearmonth in unique_yearmonths:
    cti_volume = ibhy_firm_indexed.loc[yearmonth].groupby('CTI')['Size'].sum()
    mm_volume = cti_volume[cti_volume.index != 4].sum()
    customer_volume = cti_volume[cti_volume.index == 4].sum()
    cti_unique_names = ibhy_firm_indexed.loc[yearmonth].groupby('CTI')['Account '].nunique()
    cti_4 = cti_unique_names[cti_unique_names.index == 4]
    if cti_4.empty:
        unique_customers = 0
    else:
        unique_customers = cti_4.squeeze()
    ibhy_monthly.loc[yearmonth] = (mm_volume, customer_volume, mm_volume+customer_volume, unique_customers)

# Export
ibhy_volume_breakdown.to_csv(DOWNLOADS_DIR + 'ibhy_volume_breakdown_2021-01-29.csv')
volume_sums.to_csv(DOWNLOADS_DIR + 'ibhy_volume_sums_2021-01-29.csv')
ibhy_monthly.to_csv(DOWNLOADS_DIR + 'ibhy_unique_customers_2021-01-29.csv')

###############################################################################
# VIX Volumes Breakdown

vx_file_name = 'VX_Daily_Volume_data_2021-01-29.csv'     # [CONFIGURE]

vx_mega = pd.read_csv(DOWNLOADS_DIR + vx_file_name,
                      parse_dates=['Month, Day, Year of Trading Dt'])
vx_mega = vx_mega.rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1)
# iBoxx style analysis
vx_day_cti_volume = vx_mega.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vx_days = vx_day_cti_volume.index.get_level_values(0).unique()
vx_volume_breakdown = pd.DataFrame(index=vx_days, columns=['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total'])
vx_volume_breakdown['CTI 1'] = vx_day_cti_volume.xs(1, level='CTI')
vx_volume_breakdown['CTI 2'] = vx_day_cti_volume.xs(2, level='CTI')
vx_volume_breakdown['CTI 3'] = vx_day_cti_volume.xs(3, level='CTI')
vx_volume_breakdown['CTI 4'] = vx_day_cti_volume.xs(4, level='CTI')
vx_volume_breakdown['Total'] = vx_volume_breakdown[['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4']].sum(axis=1)
# vx_volume_breakdown.index.name = 'Trade Date'
# Open interest separately
vx_volume_breakdown['Open Interest'] = vx_oi    # Can't use cfevoloi['IBHY OI'] because missing data
# ibhy_volume_breakdown = ibhy_volume_breakdown.drop(pd.Timestamp('2021-01-21'))

# Aggregate by month (must do manually because # days varies)
unique_yearmonths = vx_volume_breakdown.index.strftime('%Y-%m').unique()
# Volumes
vx_volume_sums = pd.DataFrame(columns=['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total'])
for yearmonth in unique_yearmonths:
    vx_volume_sums.loc[yearmonth] = vx_volume_breakdown.loc[yearmonth, ['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total']].sum()

# Break down by firms
vx_unique_accounts = vx_mega.groupby(['Trade Date', 'CTI'])['Account '].nunique()
vx_unique_customer_accounts_daily = vx_unique_accounts.xs(4, level='CTI')
vx_monthly = pd.DataFrame(columns=['MM Volume', 'Customer Volume', 'Total Volume', 'Unique Customer Accounts'])
vx_mega_indexed = vx_mega.set_index('Trade Date')
for yearmonth in unique_yearmonths:
    cti_volume = vx_mega_indexed.loc[yearmonth].groupby('CTI')['Size'].sum()/2  # Litch's thing has 2xVolume
    mm_volume = cti_volume[cti_volume.index != 4].sum()
    customer_volume = cti_volume[cti_volume.index == 4].sum()
    cti_unique_names = vx_mega_indexed.loc[yearmonth].groupby('CTI')['Account '].nunique()
    cti_4 = cti_unique_names[cti_unique_names.index == 4]
    if cti_4.empty:
        unique_customers = 0
    else:
        unique_customers = cti_4.squeeze()
    vx_monthly.loc[yearmonth] = (mm_volume, customer_volume, mm_volume+customer_volume, unique_customers)

# Export
vx_volume_breakdown.to_csv(DOWNLOADS_DIR + 'vx_volume_breakdown_2021-01-29.csv')
volume_sums.to_csv(DOWNLOADS_DIR + 'vx_volume_sums_2021-01-29.csv')
vx_monthly.to_csv(DOWNLOADS_DIR + 'vx_unique_customers_2021-01-29.csv')

###############################################################################
# Mini VIX

vxm_file_name = 'VXM_Daily_Volume_data_2021-01-29.csv'     # [CONFIGURE]

vxm_mega = pd.read_csv(DOWNLOADS_DIR + vxm_file_name,
                       parse_dates=['Month, Day, Year of Trading Dt'])
vxm_mega = vxm_mega.rename({'Month, Day, Year of Trading Dt': 'Trade Date'}, axis=1)
# iBoxx style analysis
vxm_day_cti_volume = vxm_mega.groupby(['Trade Date', 'CTI'])['Size'].sum()/2
vxm_days = vxm_day_cti_volume.index.get_level_values(0).unique()
vxm_volume_breakdown = pd.DataFrame(index=vxm_days, columns=['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total'])
vxm_volume_breakdown['CTI 1'] = vxm_day_cti_volume.xs(1, level='CTI')
vxm_volume_breakdown['CTI 2'] = vxm_day_cti_volume.xs(2, level='CTI')
vxm_volume_breakdown['CTI 3'] = vxm_day_cti_volume.xs(3, level='CTI')
vxm_volume_breakdown['CTI 4'] = vxm_day_cti_volume.xs(4, level='CTI')
vxm_volume_breakdown['Total'] = vxm_volume_breakdown[['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4']].sum(axis=1)
# vxm_volume_breakdown.index.name = 'Trade Date'
# Open interest separately
vxm_volume_breakdown['Open Interest'] = vxm_oi    # Can't use cfevoloi['IBHY OI'] because missing data
# ibhy_volume_breakdown = ibhy_volume_breakdown.drop(pd.Timestamp('2021-01-21'))

# Aggregate by month (must do manually because # days varies)
unique_yearmonths = vxm_volume_breakdown.index.strftime('%Y-%m').unique()
# Volumes
vxm_volume_sums = pd.DataFrame(columns=['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total'])
for yearmonth in unique_yearmonths:
    vxm_volume_sums.loc[yearmonth] = vxm_volume_breakdown.loc[yearmonth, ['CTI 1', 'CTI 2', 'CTI 3', 'CTI 4', 'Total']].sum()

# Break down by firms
vxm_unique_accounts = vxm_mega.groupby(['Trade Date', 'CTI'])['Account '].nunique()
vxm_unique_customer_accounts_daily = vxm_unique_accounts.xs(4, level='CTI')
vxm_monthly = pd.DataFrame(columns=['MM Volume', 'Customer Volume', 'Total Volume', 'Unique Customer Accounts'])
vxm_mega_indexed = vxm_mega.set_index('Trade Date')
for yearmonth in unique_yearmonths:
    cti_volume = vxm_mega_indexed.loc[yearmonth].groupby('CTI')['Size'].sum()/2  # Litch's thing has 2xVolume
    mm_volume = cti_volume[cti_volume.index != 4].sum()
    customer_volume = cti_volume[cti_volume.index == 4].sum()
    cti_unique_names = vxm_mega_indexed.loc[yearmonth].groupby('CTI')['Account '].nunique()
    cti_4 = cti_unique_names[cti_unique_names.index == 4]
    if cti_4.empty:
        unique_customers = 0
    else:
        unique_customers = cti_4.squeeze()
    vxm_monthly.loc[yearmonth] = (mm_volume, customer_volume, mm_volume+customer_volume, unique_customers)

# Export
vxm_volume_breakdown.to_csv(DOWNLOADS_DIR + 'vxm_volume_breakdown_2021-01-29.csv')
volume_sums.to_csv(DOWNLOADS_DIR + 'vxm_volume_sums_2021-01-29.csv')
vxm_monthly.to_csv(DOWNLOADS_DIR + 'vxm_unique_customers_2021-01-29.csv')
