# I have worked with 3 kinds of futures, and they've all had very different maturity date setups:
#   1) VIX Futures - same as options, 30 calendar days (1 more for holiday) before
#                    3rd Friday SPX options expiry
#   2) Treasury Futures - 2-year: last business day of quarterly month
#                         5-year: last business day of quarterly month
#                         10-year: 7th business day preceding last business day of quarterly month
#                         30-year: 7th business day preceding last business day of quarterly month
#      NOTE: Treasury Options are different - last Friday at least 2 business days
#            preceding last business day of month
#   3) iBoxx Futures - first business day of the month
# So we need bespoke functions for each type of expiry - one should exist in
# options_futures_expirations_v3.py for each of the cases listed above, in the "monthly expiration
# date functions" section.

# With that done, we can generate a list of iBoxx maturities and weave together Bloomberg's generic 1st
# term futures (e.g. IBY1 Comdty) with 2nd term futures (e.g. IBY2 Comdty) at correct date prior to expiry.

import pandas as pd
from options_futures_expirations_v3 import BUSDAY_OFFSET, generate_expiries
from futures_reader import create_bloomberg_connection, stitch_bloomberg_futures
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use('cboe-fivethirtyeight')

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'
# DATA_DIR = 'C:/Users/gzhang/Downloads/iBoxx Bloomberg Pulls/'
START_DATE = pd.Timestamp('2000-01-01')
END_DATE = pd.Timestamp('now').normalize() - BUSDAY_OFFSET  # Default yesterday
ROLL_N_BEFORE_EXPIRY = 3    # Default 3
con = create_bloomberg_connection()

###############################################################################

# Pull from Bloomberg
iby1 = con.bdh('IBY1 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
iby2 = con.bdh('IBY2 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ibxxibhy = con.bdh('IBXXIBHY Index', ['PX_LAST'],
                   f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ihb1 = con.bdh('IHB1 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ihb2 = con.bdh('IHB2 Index', ['PX_LAST', 'PX_VOLUME', 'OPEN_INT', 'FUT_AGGTE_VOL', 'FUT_AGGTE_OPEN_INT'],
               f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
ibxxibig = con.bdh('IBXXIBIG Index', ['PX_LAST'],
                   f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)

# Generate iBoxx maturities and surrounding "useful" dates
iboxx_maturities = generate_expiries(START_DATE, END_DATE, specific_product='iBoxx')
iboxx_maturities_df = pd.DataFrame({'Maturity': iboxx_maturities,
                                    'Selected Roll Date': iboxx_maturities - ROLL_N_BEFORE_EXPIRY*BUSDAY_OFFSET,
                                    'Post-Roll Return Date': iboxx_maturities - (ROLL_N_BEFORE_EXPIRY-1)*BUSDAY_OFFSET,
                                    'Bloomberg Stitch Date': iboxx_maturities + BUSDAY_OFFSET}).set_index('Maturity')
iboxx_maturities_df = iboxx_maturities_df[iby1.index[0]:iby1.index[-1]].copy()  # Crop

# This is literally insane, but have to fix maturity date settlement values from Bloomberg
iby1.loc[iboxx_maturities_df.index, 'PX_LAST'] = \
    ibxxibhy.loc[iboxx_maturities_df.index, 'PX_LAST'].round(2).values
ihb1.loc[ihb1.index & iboxx_maturities, 'PX_LAST'] = \
    ibxxibig.loc[ihb1.index & iboxx_maturities, 'PX_LAST'].round(2).values

###############################################################################

# [EXPERIMENTAL] Testing out functionalized version of this script
iby_roll_df_v2 = stitch_bloomberg_futures(iby1['PX_LAST'], iby2['PX_LAST'], iboxx_maturities_df)
iby_roll_df_v2_alt = stitch_bloomberg_futures(iby1['PX_LAST'], iby2['PX_LAST'], roll_n_before_expiry=3,
                                              start_datelike=START_DATE, end_datelike=END_DATE,
                                              specific_product='iBoxx')
# ihb_roll_df = stitch_bloomberg_futures(ihb1['PX_LAST'], ihb2['PX_LAST'], iboxx_maturities_df)
ihb_roll_df = stitch_bloomberg_futures(ihb1['PX_LAST'], ihb2['PX_LAST'], roll_n_before_expiry=3,
                                       start_datelike=START_DATE, end_datelike=END_DATE,
                                       specific_product='iBoxx')

# Go through trade date history, performing rolls and Bloomberg data stitches
# Goal is to end up with:
#   - a percent return history with reasonable stitching
#   - a scaled price history starting at 100 or something
#   - running roll cost
iby_roll_df = pd.DataFrame({'Bloomberg 1st': iby1['PX_LAST'], 'Bloomberg 2nd': iby2['PX_LAST']})
iby_roll_df.index.name = 'Trade Date'
iby_roll_df = iby_roll_df.dropna(how='all')     # Remove holiday NaNs; note first term maturity has more NaNs
# NOTE: 1 NaN price causes 2 consecutive NaN changes - day of and day after - but it SHOULD never affect us
iby_roll_df['1st Change'] = iby_roll_df['Bloomberg 1st'].pct_change(fill_method=None)
iby_roll_df['2nd Change'] = iby_roll_df['Bloomberg 2nd'].pct_change(fill_method=None)
for maturity_date in iboxx_maturities_df.index:
    if maturity_date in iby_roll_df.index:
        # Perform roll-related tasks
        task_dates = iboxx_maturities_df.loc[maturity_date]
        roll_cost = (iby_roll_df.loc[task_dates['Selected Roll Date'], 'Bloomberg 2nd']
                     - iby_roll_df.loc[task_dates['Selected Roll Date'], 'Bloomberg 1st'])
        post_roll_pre_stitch_returns = \
            iby_roll_df.loc[task_dates['Post-Roll Return Date']:maturity_date, '2nd Change']
        stitch_date_return = \
            (iby_roll_df.loc[task_dates['Bloomberg Stitch Date'], 'Bloomberg 1st']
             - iby_roll_df.loc[maturity_date, 'Bloomberg 2nd']) / iby_roll_df.loc[maturity_date, 'Bloomberg 2nd']
        # Write to DF
        iby_roll_df.loc[task_dates['Selected Roll Date'], 'Roll Cost'] = roll_cost
        iby_roll_df.loc[task_dates['Post-Roll Return Date']:maturity_date, 'Stitched Change from 2nd'] = \
            post_roll_pre_stitch_returns.values
        iby_roll_df.loc[task_dates['Bloomberg Stitch Date'], 'Stitched Change from (1st-2nd)/2nd'] = \
            stitch_date_return
no_stitch_returns_idx = (iby_roll_df['Stitched Change from 2nd'].isna()
                         & iby_roll_df['Stitched Change from (1st-2nd)/2nd'].isna())
iby_roll_df.loc[no_stitch_returns_idx, 'Stitched Change from 1st'] = \
    iby_roll_df.loc[no_stitch_returns_idx, '1st Change'].values

# Combine purposefully separated 3 components to create stitched percent returns
iby_roll_df['Stitched Change'] = \
    (iby_roll_df['Stitched Change from 2nd']
     .combine_first(iby_roll_df['Stitched Change from (1st-2nd)/2nd'])
     .combine_first(iby_roll_df['Stitched Change from 1st']))

# Run stitched returns on 100 to get scaled price history
iby_roll_df['Scaled Price'] = (iby_roll_df['Stitched Change']+1).cumprod() * 100
iby_roll_df.loc[iby_roll_df.index[0], 'Scaled Price'] = 100

# Sum roll costs
iby_roll_df['Cumulative Roll Cost'] = iby_roll_df['Roll Cost'].cumsum().ffill()

###############################################################################

# Try IBHY crafted first month vs. HYG correlation
ibhy1_change = iby_roll_df['Stitched Change']
hyg = con.bdh('HYG US Equity', ['PX_LAST', 'PX_VOLUME', 'TOT_RETURN_INDEX_GROSS_DVDS'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
hyg_tr = hyg['TOT_RETURN_INDEX_GROSS_DVDS']
hyg_change = hyg_tr.pct_change()
# Subtle: must line up the series' indexes
# NOTE: this is not a joke - ibhy1_change.rolling(n*21, center=False).corr(hyg_change).dropna()
#       does not work unless both series have identical indexes, no NaNs; otherwise, the
#       cleanest method is literally
#       ibhy1_hyg_change_df.iloc[:, 0].rolling(n*21, center=False).corr(ibhy1_hyg_change_df.iloc[:, 1]).dropna()
ibhy1_hyg_change_df = pd.DataFrame({'IBHY1': ibhy1_change, 'HYG': hyg_change}).dropna(how='any')
ibhy1_hyg_corr_dict = {}
for n in [1, 2, 3, 6]:
    ibhy1_hyg_corr_dict[n] = \
        ibhy1_hyg_change_df.iloc[:, 0].rolling(n*21, center=False).corr(ibhy1_hyg_change_df.iloc[:, 1]).dropna()
ibhy1_hyg_corr_df = pd.DataFrame({f'Rolling {n} Month': ibhy1_hyg_corr_dict[n] for n in [1, 2, 3, 6]})
ibhy1_hyg_corr_df.index.name = 'Trade Date'

# Plot - Various Window Rolls
_, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Correlation: IBHY Futures 1st Month vs. HYG Total Return')
for n in [1, 3, 6]:
    ax.plot(ibhy1_hyg_corr_dict[n].loc['2020-01-01':'2021-03-31'],
            label=f'{n}-Month Rolling Correlation')
overall_corr = ibhy1_hyg_change_df.loc['2020-01-01':'2021-03-31'].corr().iloc[1, 0]
ax.axhline(overall_corr, label=f'Overall Correlation ({overall_corr*100:.1f}%)',
           color='k', linestyle='--')
ax.legend()

# Try IBIG crafted first month vs. LQD correlation
ibig1_change = ihb_roll_df['Stitched Change']
lqd = con.bdh('LQD US Equity', ['PX_LAST', 'PX_VOLUME', 'TOT_RETURN_INDEX_GROSS_DVDS'],
              f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}").droplevel(0, axis=1)
lqd_tr = lqd['TOT_RETURN_INDEX_GROSS_DVDS']
lqd_change = lqd_tr.pct_change()
# Subtle: must line up the series' indexes
# NOTE: this is not a joke - ibhy1_change.rolling(n*21, center=False).corr(hyg_change).dropna()
#       does not work unless both series have identical indexes, no NaNs; otherwise, the
#       cleanest method is literally
#       ibhy1_hyg_change_df.iloc[:, 0].rolling(n*21, center=False).corr(ibhy1_hyg_change_df.iloc[:, 1]).dropna()
ibig1_lqd_change_df = pd.DataFrame({'IBIG1': ibig1_change, 'LQD': lqd_change}).dropna(how='any')
ibig1_lqd_corr_dict = {}
for n in [1, 2, 3, 6]:
    ibig1_lqd_corr_dict[n] = \
        ibig1_lqd_change_df.iloc[:, 0].rolling(n*21, center=False).corr(ibig1_lqd_change_df.iloc[:, 1]).dropna()
ibig1_lqd_corr_df = pd.DataFrame({f'Rolling {n} Month': ibig1_lqd_corr_dict[n] for n in [1, 2, 3, 6]})
ibig1_lqd_corr_df.index.name = 'Trade Date'

# Plot - Various Window Rolls
_, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Correlation: IBIG Futures 1st Month vs. LQD Total Return')
for n in [1, 3, 6]:
    ax.plot(ibig1_lqd_corr_dict[n].loc['2020-01-01':'2021-03-31'],
            label=f'{n}-Month Rolling Correlation')
overall_corr = ibig1_lqd_change_df.loc['2020-01-01':'2021-03-31'].corr().iloc[1, 0]
ax.axhline(overall_corr, label=f'Overall Correlation ({overall_corr*100:.1f}%)',
           color='k', linestyle='--')
ax.legend()

# Export roll_df
iby_roll_df.to_csv(DOWNLOADS_DIR+'iby_roll_df.csv')
ihb_roll_df.to_csv(DOWNLOADS_DIR+'ihb_roll_df.csv')
ibhy1_hyg_corr_df.to_csv(DOWNLOADS_DIR+'IBHY1_HYGTR_corr_rolling.csv')
ibig1_lqd_corr_df.to_csv(DOWNLOADS_DIR+'IBIG1_LQDTR_corr_rolling.csv')
