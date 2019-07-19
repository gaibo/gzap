import pandas as pd
import matplotlib.pyplot as plt

from model.data_structures import ETF, Index, VolatilityIndex
from utility.graph_utilities import \
    share_dateindex, make_basicstatstable, make_lineplot, make_fillbetween,\
    make_scatterplot, make_histogram, make_scatter_matrix, make_correlation_matrix


# Import data sources with pandas
bbg_data = pd.read_csv('data/bbg_automated_pull.csv',
                       index_col=0, parse_dates=True, header=[0, 1])
creditvix_data = pd.read_csv('data/creditvix_pc_bp_missing_4_months.csv',
                             index_col='Date', parse_dates=True)
tyvix_bp_data = pd.read_csv('data/tyvix_bp.csv',
                            index_col='Trade Date', parse_dates=True)
jgbvix_bp_data = pd.read_csv('data/jgbvix_bp.csv',
                             index_col='Trade Date', parse_dates=True)  # Rough estimate

# Create data objects
spx = Index(bbg_data['SPX Index', 'PX_LAST'], 'SPX')
vix = VolatilityIndex(bbg_data['VIX Index', 'PX_LAST'], spx, 'VIX')
ty1 = Index(bbg_data['TY1 Comdty', 'PX_LAST'], 'TY1')
tyvix = VolatilityIndex(bbg_data['TYVIX Index', 'PX_LAST'], ty1, 'TYVIX')
tyvix_bp = VolatilityIndex(tyvix_bp_data['BP TYVIX'], ty1, 'BP TYVIX')
cdx_ig = Index(bbg_data['IBOXUMAE CBBT Curncy', 'PX_LAST'], 'CDX NA IG')
vixig = VolatilityIndex(creditvix_data['VIXIG Percent'], cdx_ig, 'VIXIG')
vixig_bp = VolatilityIndex(creditvix_data['VIXIG Basis Point'], cdx_ig, 'BP VIXIG')
cdx_hy = Index(bbg_data['IBOXHYSE CBBT Curncy', 'PX_LAST'], 'CDX NA HY')
vixhy = VolatilityIndex(creditvix_data['VIXHY Percent'], cdx_hy, 'VIXHY')
vixhy_bp = VolatilityIndex(creditvix_data['VIXHY Basis Point'], cdx_hy, 'BP VIXHY')
itraxx_ie = Index(bbg_data['ITRXEBE CBBT Curncy', 'PX_LAST'], 'iTraxx EU Main')
vixie = VolatilityIndex(creditvix_data['VIXIE Percent'], itraxx_ie, 'VIXIE')
vixie_bp = VolatilityIndex(creditvix_data['VIXIE Basis Point'], itraxx_ie, 'BP VIXIE')
itraxx_xo = Index(bbg_data['ITRXEXE CBBT Curncy', 'PX_LAST'], 'iTraxx EU Xover')
vixxo = VolatilityIndex(creditvix_data['VIXXO Percent'], itraxx_xo, 'VIXXO')
vixxo_bp = VolatilityIndex(creditvix_data['VIXXO Basis Point'], itraxx_xo, 'BP VIXXO')
itraxx_fs = Index(bbg_data['ITRXESE CBBT Curncy', 'PX_LAST'], 'iTraxx EU SenFin')
vixfs = VolatilityIndex(creditvix_data['VIXFS Percent'], itraxx_fs, 'VIXFS')
vixfs_bp = VolatilityIndex(creditvix_data['VIXFS Basis Point'], itraxx_fs, 'BP VIXFS')
jb1 = Index(bbg_data['JB1 Comdty', 'PX_LAST'], 'JB1')
jgbvix = VolatilityIndex(bbg_data['SPJGBV Index', 'PX_LAST'], jb1, 'JGB VIX')
jgbvix_bp = VolatilityIndex(jgbvix_bp_data['BP JGBVIX'], jb1, 'BP JGB VIX')
usfs0110 = Index(bbg_data['USFS0110 CMPN Curncy', 'PX_LAST'], '1Y-10Y Forward Swap Rate')
srvix = VolatilityIndex(bbg_data['SRVIX Index', 'PX_LAST'], usfs0110, 'SRVIX')
sx5e = Index(bbg_data['SX5E Index', 'PX_LAST'], 'Euro Stoxx 50')
vstoxx = VolatilityIndex(bbg_data['V2X Index', 'PX_LAST'], sx5e, 'VSTOXX')
spx_tr = Index(bbg_data['SPX Index', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'SPX')
agg_tr = ETF(bbg_data['AGG Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'AGG')
hyg_tr = ETF(bbg_data['HYG Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'HYG')
ihyg_tr = ETF(bbg_data['IHYG EU Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'IHYG')
ief_tr = ETF(bbg_data['IEF Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'IEF')
lqd_tr = ETF(bbg_data['LQD Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'LQD')

# FI VIX Empirical Primer Document Reproduction

# S&P500 Index and AGG Total Return
[truncd_spx, truncd_agg] = share_dateindex([spx_tr.price(), agg_tr.price()])
make_lineplot([truncd_spx/truncd_spx[0], truncd_agg/truncd_agg[0]],
              ['SPX total return', 'AGG total return'],
              ylabel='Normalized Level', title='S&P500 Index and AGG Total Return')

# North American Credit VIXs with VIX Index
[truncd_vix, truncd_vixig, truncd_vixhy] = share_dateindex([vix.price(), vixig.price(), vixhy.price()])
make_lineplot([truncd_vix, truncd_vixig, truncd_vixhy],
              ['S&P500 VIX', 'VIXIG', 'VIXHY'],
              ylabel='Volatility Index', title='North American Credit VIXs with VIX Index')

# European Credit VIXs with VIX Index
[truncd_vix, truncd_vixie, truncd_vixxo, truncd_vixfs] = \
    share_dateindex([vix.price(), vixie.price(), vixxo.price(), vixfs.price()])
make_lineplot([truncd_vix, truncd_vixie, truncd_vixxo, truncd_vixfs],
              ['S&P500 VIX', 'VIXIE', 'VIXXO', 'VIXFS'],
              ylabel='Volatility Index', title='European Credit VIXs with VIX Index')

# VIXs in Rates Group with VIX Index
start_date = tyvix.price().index[0]
[truncd_vix, truncd_tyvix, truncd_jgbvix, truncd_srvix] = \
    map(lambda p: p.truncate(start_date), [vix.price(), tyvix.price(), jgbvix.price(), srvix.price()])
_, axleft = plt.subplots()
axleft.plot(truncd_vix, label='S&P500 VIX')
axleft.plot(truncd_tyvix, label='TYVIX')
axleft.plot(truncd_jgbvix, label='JGB VIX')
axleft.legend(loc=2)
axleft.set_ylabel('Volatility Index (% Price)')
axleft.set_title('VIXs in Rates Group with VIX Index')
axright = axleft.twinx()
axright.plot(truncd_srvix, label='SRVIX', color='C3')
axright.legend(loc=1)
axright.set_ylabel('Volatility Index (SRVIX) (bps)')
axright.set_ylim(20, 115)

# Basic Statistics for Credit VIX Indexes
make_basicstatstable([vix, vixig, vixhy, vixie, vixxo, vixfs]) \
    .to_csv('Basic Statistics for Credit VIX Indexes.csv')

# Basic Statistics for Basis Point Credit VIX Indexes
make_basicstatstable([vix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]) \
    .to_csv('Basic Statistics for Basis Point Credit VIX Indexes.csv')

# Basic Statistics for Interest Rate VIX Indexes
make_basicstatstable([vix, tyvix, jgbvix, srvix, tyvix_bp]) \
    .to_csv('Basic Statistics for Interest Rate VIX Indexes.csv')

# Basis Point Version of Credit Group
make_lineplot([vixig_bp.price(), vixhy_bp.price(), vixie_bp.price(), vixxo_bp.price(), vixfs_bp.price()],
              ['BP VIXIG', 'BP VIXHY', 'BP VIXIE', 'BP VIXXO', 'BP VIXFS'],
              ylabel='Volatility Index (bps)', title='Basis Point Version of Credit Group')

# Basis Point Version of Rates Group
make_lineplot([tyvix_bp.price(), jgbvix_bp.price(), srvix.price()],
              ['BP TYVIX', 'BP JGB VIX', 'SRVIX'],
              ylabel='Volatility Index (bps)', title='Basis Point Version of Rates Group')

# VIXIG Daily % Change vs. CDX NAIG Index Daily bps Change
make_scatterplot(vixig.price_return(False), cdx_ig.price_return(False),
                 xlabel='% Change', ylabel='bps Change',
                 title='VIXIG Daily % Change vs. CDX NAIG Index Daily bps Change')

# SRVIX Daily Change vs. LQD Daily Change
make_scatterplot(srvix.price_return(False), lqd_tr.price_return(False),
                 xlabel='% Change', ylabel='% Change',
                 title='SRVIX Daily % Change vs. LQD Daily % Change')

# Change per 10% Change in Implied Vol Index Table
# TODO

# Four Scatter Plots
_, axs = plt.subplots(2, 2)
make_scatterplot(vixhy_bp.price_return(False), hyg_tr.price_return(False),
                 xlabel='% Change', ylabel='% Change',
                 title='BP VIXHY Daily % Change vs. HYG Daily % Change',
                 ax=axs[0, 0])
make_scatterplot(vixxo_bp.price_return(False), ihyg_tr.price_return(False),
                 xlabel='% Change', ylabel='% Change',
                 title='BP VIXXO Daily % Change vs. IHYG Daily % Change',
                 ax=axs[0, 1])
make_scatterplot(tyvix_bp.price_return(False), ief_tr.price_return(False),
                 xlabel='% Change', ylabel='% Change',
                 title='BP TYVIX Daily % Change vs. IEF Daily % Change',
                 ax=axs[1, 0])
make_scatterplot(srvix.price_return(False), ief_tr.price_return(False),
                 xlabel='% Change', ylabel='% Change',
                 title='SRVIX Daily % Change vs. IEF Daily % Change',
                 ax=axs[1, 1])

# Credit VIX Difference Charts
_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixig.price(), 100*vixig.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXIG', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXIG with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
make_fillbetween(difference.index, joined_index, joined_undl_rv,
                 label='Difference', color='g', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixhy.price(), 100*vixhy.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXHY', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXHY with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
make_fillbetween(difference.index, joined_index, joined_undl_rv,
                 label='Difference', color='g', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixie.price(), 100*vixie.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXIE', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXIE with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
make_fillbetween(difference.index, joined_index, joined_undl_rv,
                 label='Difference', color='g', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixxo.price(), 100*vixxo.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXXO', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXXO with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
make_fillbetween(difference.index, joined_index, joined_undl_rv,
                 label='Difference', color='g', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixfs.price(), 100*vixfs.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXFS', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXFS with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
make_fillbetween(difference.index, joined_index, joined_undl_rv,
                 label='Difference', color='g', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

# Credit VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
_, ax_prem = make_histogram(joined_index - joined_undl_rv, hist=False,
                            label='VIX', xlabel='Volatility Premium (%)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixig.price(), 100*vixig.undl_realized_vol(do_shift=True)])
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='VIXIG', xlabel='Volatility Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C1', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixhy.price(), 100*vixhy.undl_realized_vol(do_shift=True)])
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='VIXHY', xlabel='Volatility Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C2', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixie.price(), 100*vixie.undl_realized_vol(do_shift=True)])
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='VIXIE', xlabel='Volatility Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C3', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixxo.price(), 100*vixxo.undl_realized_vol(do_shift=True)])
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='VIXXO', xlabel='Volatility Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C4', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixfs.price(), 100*vixfs.undl_realized_vol(do_shift=True)])
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='VIXFS', xlabel='Volatility Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C5', ax=ax_prem)

# Risk Premium Tails Table
# TODO

# Interest Rate VIX Difference Charts
_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([tyvix.price(), 100*tyvix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['TYVIX', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='TYVIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
make_fillbetween(difference.index, joined_index, joined_undl_rv,
                 label='Difference', color='g', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([jgbvix.price(), 100*jgbvix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['JGB VIX', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='JGB VIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
make_fillbetween(difference.index, joined_index, joined_undl_rv,
                 label='Difference', color='g', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([srvix.price(), 100*srvix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['SRVIX', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='SRVIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
make_fillbetween(difference.index, joined_index, joined_undl_rv,
                 label='Difference', color='g', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['S&P500 VIX', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='S&P500 VIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
make_fillbetween(difference.index, joined_index, joined_undl_rv,
                 label='Difference', color='g', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

# Interest Rate VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([tyvix.price(), 100*tyvix.undl_realized_vol(do_shift=True)])
_, ax_prem = make_histogram(joined_index - joined_undl_rv, hist=False,
                            label='TYVIX', xlabel='Volatility Premium (%)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
[joined_index, joined_undl_rv] = \
    share_dateindex([jgbvix.price(), 100*jgbvix.undl_realized_vol(do_shift=True)])
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='JGB VIX', xlabel='Volatility Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C1', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([srvix.price(), 100*srvix.undl_realized_vol(do_shift=True)])
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='SRVIX', xlabel='Volatility Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C2', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='S&P500 VIX', xlabel='Volatility Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C3', ax=ax_prem)

# Credit VIX Scatter Matrix
instr_list = [vix, vstoxx, vixig, vixhy, vixie, vixxo, vixfs]
data_list = list(map(lambda instr: instr.price_return(), instr_list))
color_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
label_list = list(map(lambda instr: instr.name, instr_list))
make_scatter_matrix(data_list, color_list, label_list=label_list, title='Credit VIX Scatter Matrix')

# Credit VIX Correlation Matrix
make_correlation_matrix(data_list, color_list, label_list=label_list, title='Credit VIX Correlation Matrix')

# Interest Rate VIX Scatter Matrix
instr_list = [vix, tyvix, jgbvix, srvix]
data_list = list(map(lambda instr: instr.price_return(), instr_list))
color_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
label_list = list(map(lambda instr: instr.name, instr_list))
make_scatter_matrix(data_list, color_list, label_list=label_list, title='Interest Rate VIX Scatter Matrix')

# Interest Rate VIX Correlation Matrix
make_correlation_matrix(data_list, color_list, label_list=label_list, title='Interest Rate VIX Correlation Matrix')
