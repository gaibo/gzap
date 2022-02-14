import pandas as pd
from futures_reader import create_bloomberg_connection
# from futures_reader import reformat_pdblp

DOWNLOADS_DIR = 'C:/Users/gzhang/OneDrive - CBOE/Downloads/'
con = create_bloomberg_connection()

# NOTE: 1709583D US Equity is old, matured VXX; VXX US Equity is "series B" and only goes back to 2018-01
VIX_ETPS = ['XIV US Equity', 'SVXY US Equity',
            '1709583D US Equity', 'VXX US Equity', 'VIXY US Equity',
            'UVXY US Equity', 'TVIXF US Equity',
            '00677U TT Equity', '1552 JP Equity']

# ETP_LEVERAGE = {'XIV US Equity': -1, 'SVXY US Equity': -0.5,
#                 '1709583D US Equity': 1, 'VXX US Equity': 1, 'VIXY US Equity': 1,
#                 'UVXY US Equity': 1.5, 'TVIXF US Equity': 2,
#                 '00677U TT Equity': 1, '1552 JP Equity': 1}

# ETP_CHART_COLOR = {'SVXY US Equity': 'C0',
#                    'VXX US Equity': 'C1',
#                    'VIXY US Equity': 'C2',
#                    'UVXY US Equity': 'C3',
#                    'TVIXF US Equity': 'C4',
#                    '00677U TT Equity': 'C5',
#                    '1552 JP Equity': 'C6'}

###############################################################################
# Recreate AUM chart

START_DATE = pd.Timestamp('2015-01-01')
END_DATE = pd.Timestamp('2022-01-31')

# Get AUM data from Bloomberg
etps_raw = con.bdh(VIX_ETPS, ['FUND_TOTAL_ASSETS', 'PX_VOLUME'],
                   f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}",
                   elms=[('currency', 'USD')])
etps_raw = etps_raw[VIX_ETPS].copy()    # Enforce column order

# Back out exchange rates for foreign-issued ETPs to normalize volumes
foreign_aums = con.bdh(['00677U TT Equity', '1552 JP Equity'], 'FUND_TOTAL_ASSETS',
                       f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}")
TWD_ratio = foreign_aums['00677U TT Equity']['FUND_TOTAL_ASSETS'] / etps_raw['00677U TT Equity']['FUND_TOTAL_ASSETS']
JPY_ratio = foreign_aums['1552 JP Equity']['FUND_TOTAL_ASSETS'] / etps_raw['1552 JP Equity']['FUND_TOTAL_ASSETS']

# Clean
# NOTE: the foreign ETP volume scaling seems very hokey, but we're doing it to be consistent with legacy;
#       the objectively better way to look at it is with VIX Futures Equivalent Volume (vega associated with volume)
aums = etps_raw.xs('FUND_TOTAL_ASSETS', level=1, axis=1).copy()
aums['VXX US Equity'] = aums[['1709583D US Equity', 'VXX US Equity']].dropna(how='all').sum(axis=1)     # Combine VXXs
aums = aums.drop('1709583D US Equity', axis=1)  # Drop matured VXX ETN for cleanliness
volumes = etps_raw.xs('PX_VOLUME', level=1, axis=1).copy()
volumes['VXX US Equity'] = volumes[['1709583D US Equity', 'VXX US Equity']].dropna(how='all').sum(axis=1)   # Combine
volumes = volumes.drop('1709583D US Equity', axis=1)  # Drop matured VXX ETN for cleanliness
volumes['00677U TT Equity'] /= TWD_ratio    # Scale foreign volumes down
volumes['1552 JP Equity'] /= JPY_ratio  # Scale foreign volumes down
# =============================================================================
# volumes['SVXY US Equity'] *= 0.5  # Scale levered volumes
# volumes['UVXY US Equity'] *= 1.5  # Scale levered volumes
# volumes['TVIXF US Equity'] *= 2  # Scale levered volumes
# =============================================================================

# Average by month (must do manually because # days varies)
unique_yearmonths = aums.index.strftime('%Y-%m').unique()
# AUMs
aum_means = pd.DataFrame(columns=aums.columns)
for yearmonth in unique_yearmonths:
    aum_means.loc[yearmonth] = aums.loc[yearmonth].mean()
# Volumes
volume_means = pd.DataFrame(columns=volumes.columns)
for yearmonth in unique_yearmonths:
    volume_means.loc[yearmonth] = volumes.loc[yearmonth].mean()

# Group into bigger categories
BIGGER_CATEGORIES = ['Inverse', 'Long Vol', 'Levered', 'Asian Long']
# AUMs - good for export
aum_means['Inverse'] = aum_means[['XIV US Equity', 'SVXY US Equity']].dropna(how='all').sum(axis=1)
aum_means['Long Vol'] = aum_means[['VXX US Equity', 'VIXY US Equity']].dropna(how='all').sum(axis=1)
aum_means['Levered'] = aum_means[['UVXY US Equity', 'TVIXF US Equity']].dropna(how='all').sum(axis=1)
aum_means['Asian Long'] = aum_means[['00677U TT Equity', '1552 JP Equity']].dropna(how='all').sum(axis=1)
aum_means['Total'] = aum_means[BIGGER_CATEGORIES].dropna(how='all').sum(axis=1)
# Volumes - good for export
volume_means['Inverse'] = volume_means[['XIV US Equity', 'SVXY US Equity']].dropna(how='all').sum(axis=1)
volume_means['Long Vol'] = volume_means[['VXX US Equity', 'VIXY US Equity']].dropna(how='all').sum(axis=1)
volume_means['Levered'] = volume_means[['UVXY US Equity', 'TVIXF US Equity']].dropna(how='all').sum(axis=1)
volume_means['Asian Long'] = volume_means[['00677U TT Equity', '1552 JP Equity']].dropna(how='all').sum(axis=1)
volume_means['Total'] = volume_means[BIGGER_CATEGORIES].dropna(how='all').sum(axis=1)

# Only big categories
big_cat_aum_means = aum_means[BIGGER_CATEGORIES+['Total']].copy()
big_cat_volume_means = volume_means[BIGGER_CATEGORIES+['Total']].copy()

# Export
aum_means.to_csv(DOWNLOADS_DIR + f'aum_means_{END_DATE.strftime("%Y-%m-%d")}.csv')
volume_means.to_csv(DOWNLOADS_DIR + f'volume_means_{END_DATE.strftime("%Y-%m-%d")}.csv')
big_cat_aum_means.to_csv(DOWNLOADS_DIR + f'big_cat_aum_means_{END_DATE.strftime("%Y-%m-%d")}.csv')
big_cat_volume_means.to_csv(DOWNLOADS_DIR + f'big_cat_volume_means_{END_DATE.strftime("%Y-%m-%d")}.csv')

# =============================================================================
# #%% Looking at volumes (in particular, TVIX after delisting)
#
# # Get AUM data from Bloomberg
# volume_price_raw = con.bdh(VIX_ETPS, ['PX_VOLUME', 'PX_LAST'],
#                            '20200101', '20201028',
#                            elms=[('currency','USD')])
#
# # Clean with reformat_pdblp
# volume_price = reformat_pdblp(volume_price_raw, is_bdh=True)
# volume = volume_price['PX_VOLUME']
# price = volume_price['PX_LAST']
#
# #%% Plot volumes
#
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(figsize=(19.2, 10.8))
# for etp in VIX_ETPS:
#     ax.plot(volume.xs(etp, level='ticker').dropna()/1e6, label=etp)
# ax.legend(fontsize='x-large')
# ax.set_title('Volumes', fontsize='x-large')
# ax.set_xlabel('Date', fontsize='x-large')
# ax.set_ylabel('Shares Traded (MM)', fontsize='x-large')
# fig.set_tight_layout(True)
#
# #%% Plot AUM changes
#
# fig, ax = plt.subplots()
# for etp in VIX_ETPS:
#     changes = aums[etp].dropna().diff()
#     rolling_changes = changes.rolling(21, center=True).mean()   # 1-month
# #    ax.plot(changes, color=ETP_CHART_COLOR[etp], alpha=0.3)
#     ax.plot(rolling_changes, label=etp, color=ETP_CHART_COLOR[etp], alpha=1)
# ax.legend(fontsize='x-large')
# ax.set_title('AUM Changes', fontsize='x-large')
# ax.set_xlabel('Date', fontsize='x-large')
# ax.set_ylabel('Daily Change in AUM (MM)', fontsize='x-large')
# fig.set_tight_layout(True)
# =============================================================================
