import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model.data_structures import ETF, Futures, Index, VolatilityIndex
from utility.graph_utilities import \
    share_dateindex, make_basicstatstable, make_lineplot, make_fillbetween,\
    make_scatterplot, make_histogram

# Load raw data
spx_data = pd.read_csv('data/spxt_data.csv', index_col='Date', parse_dates=True)
vix_data = pd.read_csv('data/vix_data.csv', index_col='Date', parse_dates=True)
ty1_futures_data = pd.read_csv('data/ty1_futures_data.csv', index_col='Date', parse_dates=True)
tyvix_data = pd.read_csv('data/tyvix_data.csv', index_col='Date', parse_dates=True)
tyvix_bp_data = pd.read_csv('data/tyvix_bp_data.csv', index_col='Trade Date', parse_dates=True)
ief_data = pd.read_csv('data/bbg_ief_data.csv', index_col='Date', parse_dates=True)
three_month_t_bill = pd.read_csv('data/three_month_t_bill.csv', index_col='Date', parse_dates=True)

# Create data structures
spx = Index(spx_data['PX_LAST'], 'SPX')
vix = VolatilityIndex(vix_data['Close'], spx, 'VIX')
ty1 = Futures(ty1_futures_data['PX_LAST'], None, 'TY1')
tyvix = VolatilityIndex(tyvix_data['Close'], ty1, 'TYVIX')
tyvix_bp = VolatilityIndex(tyvix_bp_data['BP TYVIX'], ty1, 'TYVIX BP')
ief = ETF(ief_data['TOT_RETURN_INDEX_GROSS_DVDS'], 'IEF')
risk_free_rate = three_month_t_bill['DTB3'].replace('.', np.NaN).dropna().astype(float) / 100

# Line plot comparing cumulative returns of SPX to IEF
[truncd_spx, truncd_ief] = share_dateindex([spx.price(), ief.price()])
make_lineplot([truncd_spx/truncd_spx[0], truncd_ief/truncd_ief[0]],
              ['SPX cumulative return', 'IEF cumulative return'])

# Sharpe ratios of SPX and IEF
daily_risk_free_rate = np.exp(risk_free_rate*1/252) - 1
[truncd_spx_ret, truncd_ief_ret, truncd_rfr] = \
    share_dateindex([spx.price_return(), ief.price_return(), daily_risk_free_rate])
spx_excess_ret = truncd_spx_ret - truncd_rfr
ief_excess_ret = truncd_ief_ret - truncd_rfr
spx_sharpe = spx_excess_ret.mean() / spx_excess_ret.std() * np.sqrt(252)
ief_sharpe = ief_excess_ret.mean() / ief_excess_ret.std() * np.sqrt(252)
print("SPX Sharpe Ratio: {}\nIEF Sharpe Ratio: {}".format(spx_sharpe, ief_sharpe))

# Level deciles + distribution chart
vix_deciles = vix.price().quantile(np.arange(0, 1.1, 0.1))
tyvix_deciles = tyvix.price().quantile(np.arange(0, 1.1, 0.1))
print("VIX Deciles:\n{}".format(vix_deciles))
print("TYVIX Deciles:\n{}".format(tyvix_deciles))
make_histogram(vix.price(), n_bins=100,
               xlabel='Volatility Index (%)', ylabel='Probability',
               title='VIX Level Distribution')
make_histogram(tyvix.price(), n_bins=100,
               xlabel='Volatility Index (%)', ylabel='Probability',
               title='TYVIX Level Distribution')

# Realized vol level deciles + distribution chart
vix_rv_deciles = vix.undl_realized_vol().quantile(np.arange(0, 1.1, 0.1)) * 100
tyvix_rv_deciles = tyvix.undl_realized_vol().quantile(np.arange(0, 1.1, 0.1)) * 100
print("VIX Underlying Realized Vol Deciles:\n{}".format(vix_rv_deciles))
print("TYVIX Underlying Realized Vol Deciles:\n{}".format(tyvix_rv_deciles))
make_histogram((vix.undl_realized_vol()*100).dropna(), n_bins=100,
               xlabel='Realized Volatility (%)', ylabel='Probability',
               title='VIX Underlying Realized Vol Distribution')
make_histogram((tyvix.undl_realized_vol()*100).dropna(), n_bins=100,
               xlabel='Realized Volatility (%)', ylabel='Probability',
               title='TYVIX Underlying Realized Vol Distribution')

# Level vs. RV difference deciles + distribution chart
vix_diff = (vix.price() - vix.undl_realized_vol(do_shift=True)*100).dropna()
vix_diff_abs_deciles = vix_diff.abs().quantile(np.arange(0, 1.1, 0.1))
tyvix_diff = (tyvix.price() - tyvix.undl_realized_vol(do_shift=True)*100).dropna()
tyvix_diff_abs_deciles = tyvix_diff.abs().quantile(np.arange(0, 1.1, 0.1))
print("VIX vs Underlying Realized Vol Difference Deciles:\n{}".format(vix_diff_abs_deciles))
print("TYVIX vs Underlying Realized Vol Difference Deciles:\n{}".format(tyvix_diff_abs_deciles))
make_histogram(vix_diff, n_bins=100,
               xlabel='Volatility (%)', ylabel='Probability',
               title='VIX - Underlying Realized Vol (Implied Vol Premium) Distribution')
make_histogram(tyvix_diff, n_bins=100,
               xlabel='Volatility (%)', ylabel='Probability',
               title='TYVIX - Underlying Realized Vol (Implied Vol Premium) Distribution')

# Rolling 6-month vol of VIX deciles
six_month_vol_of_vix = np.sqrt(vix.price_return().rolling(6*21).var(ddof=0) * 252).dropna()
six_month_vol_of_tyvix = np.sqrt(tyvix.price_return().rolling(6*21).var(ddof=0) * 252).dropna()
vol_of_vix_deciles = six_month_vol_of_vix.quantile(np.arange(0, 1.1, 0.1))
vol_of_tyvix_deciles = six_month_vol_of_tyvix.quantile(np.arange(0, 1.1, 0.1))
print("Vol of VIX Deciles:\n{}".format(vol_of_vix_deciles))
print("Vol of TYVIX Deciles:\n{}".format(vol_of_tyvix_deciles))

# Daily range in levels deciles

# Rolling 6-month correlation between VIX and TYVIX

# Rolling 6-month correlation with underlying

# Levels divided by sqrt(12) to show implied change range (68% confidence level)
