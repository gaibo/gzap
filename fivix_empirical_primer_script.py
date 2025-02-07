from load_data_script import *

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from utility.universal_tools import share_dateindex, get_best_fit
from utility.mpl_graph_tools import make_basicstatstable, make_lineplot, \
    make_fillbetween, make_scatterplot, make_histogram, make_scatter_matrix, \
    make_correlation_matrix

register_matplotlib_converters()

##############################################################################

# FI VIX Empirical Primer Document Reproduction

# [Figure 1] S&P500 Index and AGG Total Return
[truncd_spx, truncd_agg] = share_dateindex([spx_tr.price(), agg_tr.price()])
make_lineplot([truncd_spx/truncd_spx[0], truncd_agg/truncd_agg[0]],
              ['SPX total return', 'AGG total return'],
              ylabel='Normalized Level', title='S&P500 Index and AGG Total Return')

# [Figure 2] North American Credit VIXs with VIX Index
[truncd_vix, truncd_vixig, truncd_vixhy] = share_dateindex([vix.price(), vixig.price(), vixhy.price()])
make_lineplot([truncd_vix, truncd_vixig, truncd_vixhy],
              ['S&P500 VIX', 'VIXIG', 'VIXHY'],
              ylabel='Volatility Index', title='North American Credit VIXs with VIX Index')

# [Figure 3] European Credit VIXs with VIX Index
[truncd_vix, truncd_vixie, truncd_vixxo, truncd_vixfs] = \
    share_dateindex([vix.price(), vixie.price(), vixxo.price(), vixfs.price()])
make_lineplot([truncd_vix, truncd_vixie, truncd_vixxo, truncd_vixfs],
              ['S&P500 VIX', 'VIXIE', 'VIXXO', 'VIXFS'],
              ylabel='Volatility Index', title='European Credit VIXs with VIX Index')

# [Figure 4] VIXs in Rates Group with VIX Index
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

# [Table] Volatility of Volatility Table for Percent and Basis Point Vol Indexes
pct_vol_of_vol_list = [vix, tyvix, jgbvix, srvix, vixig, vixhy, vixie, vixxo, vixfs]
pct_vol_of_vol_names = [v.name for v in pct_vol_of_vol_list]
aligned_pct_list = share_dateindex([v.price_return(False) for v in pct_vol_of_vol_list])
pct_vol_of_vol_nums = [(pr.var(ddof=0)*252)**0.5 * 100 for pr in aligned_pct_list]
pct_vol_of_vol_df = \
    pd.DataFrame(zip(pct_vol_of_vol_names, pct_vol_of_vol_nums),
                 columns=['Vol Index', '% Vol of % Vol']) \
    .set_index('Vol Index').sort_values('% Vol of % Vol', ascending=False)
pct_vol_of_vol_df.to_csv('Volatility of Volatility Table for Percent Change Indexes.csv')
bps_vol_of_vol_list = [tyvix_bp, jgbvix_bp, srvix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]
bps_vol_of_vol_names = [v.name for v in bps_vol_of_vol_list]
aligned_bps_list = share_dateindex([v.price().diff() for v in bps_vol_of_vol_list])
bps_vol_of_vol_nums = [((pr**2).mean()*252)**0.5 for pr in aligned_bps_list]
bps_vol_of_vol_df = \
    pd.DataFrame(zip(bps_vol_of_vol_names, bps_vol_of_vol_nums),
                 columns=['Vol Index', 'bps Vol of bps Vol']) \
    .set_index('Vol Index').sort_values('bps Vol of bps Vol', ascending=False)
bps_vol_of_vol_df.to_csv('Volatility of Volatility Table for Basis Point Indexes.csv')

# [Table] Basic Statistics Table for Credit VIX Indexes
make_basicstatstable([vix, vixig, vixhy, vixie, vixxo, vixfs]) \
    .to_csv('Basic Statistics for Credit VIX.csv')

# [Table] Basic Statistics Table for Basis Point Credit VIX Indexes
make_basicstatstable([vix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]) \
    .to_csv('Basic Statistics for Basis Point Credit VIX.csv')

# [Table] Basic Statistics Table for Interest Rate VIX Indexes
make_basicstatstable([vix, tyvix, jgbvix, srvix, tyvix_bp, jgbvix_bp]) \
    .to_csv('Basic Statistics for Interest Rate VIX.csv')

# [Figure 5] Basis Point Version of Credit Group
make_lineplot([vixig_bp.price(), vixhy_bp.price(), vixie_bp.price(), vixxo_bp.price(), vixfs_bp.price()],
              ['BP VIXIG', 'BP VIXHY', 'BP VIXIE', 'BP VIXXO', 'BP VIXFS'],
              ylabel='Volatility Index (bps)', title='Basis Point Version of Credit Group')

# [Figure 6] Basis Point Version of Rates Group
make_lineplot([tyvix_bp.price(), jgbvix_bp.price(), srvix.price()],
              ['BP TYVIX', 'BP JGB VIX', 'SRVIX'],
              ylabel='Volatility Index (bps)', title='Basis Point Version of Rates Group')

# [Figure 7] VIXIG Daily % Change vs. CDX NAIG Index Daily bps Change
make_scatterplot(vixig.price_return(False)*100, cdx_ig_normalized.price().diff(),
                 xlabel='% Change', ylabel='bps Change',
                 title='VIXIG Daily % Change vs. CDX NAIG Index Daily bps Change')

# [Figure 8] SRVIX Daily Change vs. LQD Daily Change
make_scatterplot(srvix.price_return(False)*100, lqd_tr.price_return(False)*100,
                 xlabel='% Change', ylabel='% Change',
                 title='SRVIX Daily % Change vs. LQD Daily % Change')

# [Table] Credit VIX Change per 10% Change in Implied Vol Index Table
_, vixig_slope, _ = \
    get_best_fit(vixig.price_return(False)*100, vixig.underlying.price().diff(), fit_intercept=False)
_, vixhy_slope, _ = \
    get_best_fit(vixhy.price_return(False)*100, vixhy.underlying.price().diff(), fit_intercept=False)
_, vixie_slope, _ = \
    get_best_fit(vixie.price_return(False)*100, vixie.underlying.price().diff(), fit_intercept=False)
_, vixxo_slope, _ = \
    get_best_fit(vixxo.price_return(False)*100, vixxo.underlying.price().diff(), fit_intercept=False)
_, vixfs_slope, _ = \
    get_best_fit(vixfs.price_return(False)*100, vixfs.underlying.price().diff(), fit_intercept=False)
_, vixig_bp_slope, _ = \
    get_best_fit(vixig_bp.price_return(False)*100, vixig_bp.underlying.price().diff(), fit_intercept=False)
_, vixhy_bp_slope, _ = \
    get_best_fit(vixhy_bp.price_return(False)*100, vixhy_bp.underlying.price().diff(), fit_intercept=False)
_, vixie_bp_slope, _ = \
    get_best_fit(vixie_bp.price_return(False)*100, vixie_bp.underlying.price().diff(), fit_intercept=False)
_, vixxo_bp_slope, _ = \
    get_best_fit(vixxo_bp.price_return(False)*100, vixxo_bp.underlying.price().diff(), fit_intercept=False)
_, vixfs_bp_slope, _ = \
    get_best_fit(vixfs_bp.price_return(False)*100, vixfs_bp.underlying.price().diff(), fit_intercept=False)
creditvix_names = ['VIXIG', 'VIXHY', 'VIXIE', 'VIXXO', 'VIXFS',
                   'BP VIXIG', 'BP VIXHY', 'BP VIXIE', 'BP VIXXO', 'BP VIXFS']
creditvix_slopes = [vixig_slope, vixhy_slope, vixie_slope, vixxo_slope, vixfs_slope,
                    vixig_bp_slope, vixhy_bp_slope, vixie_bp_slope, vixxo_bp_slope, vixfs_bp_slope]
creditvix_change_table = \
    pd.DataFrame({'Credit VIX Index': creditvix_names,
                  'Estimated bps spread change of underlying from 10% change in implied vol index':
                      list(map(lambda m: 10*m, creditvix_slopes))}) \
    .set_index('Credit VIX Index')
creditvix_change_table.to_csv('Change Table for Credit VIX.csv')

# [Table] Interest Rate VIX Change per 10% Change in Implied Vol Index Table
_, tyvix_slope, _ = \
    get_best_fit(tyvix.price_return(False)*100, tyvix.underlying.price_return(False)*100, fit_intercept=False)
_, jgbvix_slope, _ = \
    get_best_fit(jgbvix.price_return(False)*100, jgbvix.underlying.price_return(False)*100, fit_intercept=False)
_, tyvix_bp_slope, _ = \
    get_best_fit(tyvix_bp.price_return(False)*100, tyvix_bp.underlying.price().diff()*100, fit_intercept=False)
_, jgbvix_bp_slope, _ = \
    get_best_fit(jgbvix_bp.price_return(False)*100, jgbvix_bp.underlying.price().diff()*100, fit_intercept=False)
_, srvix_slope, _ = \
    get_best_fit(srvix.price_return(False)*100, srvix.underlying.price().diff()*100, fit_intercept=False)
_, vix_slope, _ = \
    get_best_fit(vix.price_return(False)*100, vix.underlying.price_return(False)*100, fit_intercept=False)
irvix_names = ['TYVIX', 'JGB VIX', 'BP TYVIX', 'BP JGB VIX', 'SRVIX', 'VIX']
irvix_slopes = [tyvix_slope, jgbvix_slope, tyvix_bp_slope, jgbvix_bp_slope, srvix_slope, vix_slope]
irvix_change_table = \
    pd.DataFrame({'Interest Rate VIX Index': irvix_names,
                  'Estimated % or bps spread change of underlying from 10% change in implied vol index':
                      list(map(lambda m: 10*m, irvix_slopes))}) \
    .set_index('Interest Rate VIX Index')
irvix_change_table.to_csv('Change Table for Interest Rate VIX.csv')

# [Table] Various Assets Change per 10% Change in Implied Vol Index Table
_, slope_1, _ = \
    get_best_fit(vixig_bp.price_return(False), lqd_tr.price_return(False), fit_intercept=False)
_, slope_2, _ = \
    get_best_fit(vixhy_bp.price_return(False), hyg_tr.price_return(False), fit_intercept=False)
_, slope_3, _ = \
    get_best_fit(vixxo_bp.price_return(False), ihyg_tr.price_return(False), fit_intercept=False)
_, slope_4, _ = \
    get_best_fit(tyvix_bp.price_return(False), ief_tr.price_return(False), fit_intercept=False)
_, slope_5, _ = \
    get_best_fit(srvix.price_return(False), ief_tr.price_return(False), fit_intercept=False)
_, slope_6, _ = \
    get_best_fit(srvix.price_return(False), lqd_tr.price_return(False), fit_intercept=False)
various_names = ['BP VIXIG - LQD', 'BP VIXHY - HYG', 'BP VIXXO - IHYG', 'BP TYVIX - IEF',
                 'SRVIX - IEF', 'SRVIX - LQD']
various_slopes = [slope_1, slope_2, slope_3, slope_4, slope_5, slope_6]
various_change_table = \
    pd.DataFrame({'Various Vol Index - Asset Pairs': various_names,
                  'Estimated % price change of asset from 10% change in implied vol index':
                      list(map(lambda m: 10*m, various_slopes))}) \
    .set_index('Various Vol Index - Asset Pairs')
various_change_table.to_csv('Change Table for Various Vol Index - Asset Pairs.csv')

# [Figure 9] Four Scatter Plots
_, axs = plt.subplots(2, 2)
make_scatterplot(vixhy_bp.price_return(False)*100, hyg_tr.price_return(False)*100,
                 xlabel='% Change', ylabel='% Change',
                 title='BP VIXHY Daily % Change vs. HYG Daily % Change',
                 ax=axs[0, 0])
make_scatterplot(vixxo_bp.price_return(False)*100, ihyg_tr.price_return(False)*100,
                 xlabel='% Change', ylabel='% Change',
                 title='BP VIXXO Daily % Change vs. IHYG Daily % Change',
                 ax=axs[0, 1])
make_scatterplot(tyvix_bp.price_return(False)*100, ief_tr.price_return(False)*100,
                 xlabel='% Change', ylabel='% Change',
                 title='BP TYVIX Daily % Change vs. IEF Daily % Change',
                 ax=axs[1, 0])
make_scatterplot(srvix.price_return(False)*100, ief_tr.price_return(False)*100,
                 xlabel='% Change', ylabel='% Change',
                 title='SRVIX Daily % Change vs. IEF Daily % Change',
                 ax=axs[1, 1])

# [Figure 10, 11, 12, 13, 14] Credit VIX Difference Charts
_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixig.price(), 100*vixig.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXIG', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXIG with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixhy.price(), 100*vixhy.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXHY', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXHY with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixie.price(), 100*vixie.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXIE', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXIE with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixxo.price(), 100*vixxo.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXXO', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXXO with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixfs.price(), 100*vixfs.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['VIXFS', 'Realized Volatility'],
              ylabel='Volatility (% Spread bps)',
              title='VIXFS with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])

# [Figure 15] Credit VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
vix_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(vix_diff, hist=False,
                            label='VIX', xlabel='Implied Vol Premium (%)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixig.price(), 100*vixig.undl_realized_vol(do_shift=True)])
vixig_diff = joined_index - joined_undl_rv
make_histogram(vixig_diff, hist=False,
               label='VIXIG', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C1', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixhy.price(), 100*vixhy.undl_realized_vol(do_shift=True)])
vixhy_diff = joined_index - joined_undl_rv
make_histogram(vixhy_diff, hist=False,
               label='VIXHY', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C2', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixie.price(), 100*vixie.undl_realized_vol(do_shift=True)])
vixie_diff = joined_index - joined_undl_rv
make_histogram(vixie_diff, hist=False,
               label='VIXIE', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C3', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixxo.price(), 100*vixxo.undl_realized_vol(do_shift=True)])
vixxo_diff = joined_index - joined_undl_rv
make_histogram(vixxo_diff, hist=False,
               label='VIXXO', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C4', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixfs.price(), 100*vixfs.undl_realized_vol(do_shift=True)])
vixfs_diff = joined_index - joined_undl_rv
make_histogram(vixfs_diff, hist=False,
               label='VIXFS', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C5', ax=ax_prem)

# [Table] Credit VIX Risk Premium Tails Table
names_list = ['VIX', 'VIXIG', 'VIXHY', 'VIXIE', 'VIXXO', 'VIXFS']
diff_list = [vix_diff, vixig_diff, vixhy_diff, vixie_diff, vixxo_diff, vixfs_diff]
one_percent_mean = list(map(lambda d: d[d.rank(pct=True) < 0.01].mean(), diff_list))
five_percent_mean = list(map(lambda d: d[d.rank(pct=True) < 0.05].mean(), diff_list))
ten_percent_mean = list(map(lambda d: d[d.rank(pct=True) < 0.1].mean(), diff_list))
creditvix_risk_prem_lefttail_table = \
    pd.DataFrame({'Vol Index': names_list,
                  'Mean of 1% Left Tail': one_percent_mean,
                  'Mean of 5% Left Tail': five_percent_mean,
                  'Mean of 10% Left Tail': ten_percent_mean}).set_index('Vol Index')
creditvix_risk_prem_lefttail_table.to_csv('Risk Premium Left Tail Table for Credit VIX.csv')

# [Figure 16, 17, 18, 19] Interest Rate VIX Difference Charts
_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([tyvix.price(), 100*tyvix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['TYVIX', 'Realized Volatility'],
              ylabel='Volatility (% Price)',
              title='TYVIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([jgbvix.price(), 100*jgbvix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['JGB VIX', 'Realized Volatility'],
              ylabel='Volatility (% Price)',
              title='JGB VIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([srvix.price(), srvix.undl_realized_vol(do_shift=True, window=252, bps=True)])
make_lineplot([joined_index, joined_undl_rv], ['SRVIX', 'Realized Volatility'],
              ylabel='Volatility (bps)',
              title='SRVIX with Realized (252 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['S&P500 VIX', 'Realized Volatility'],
              ylabel='Volatility (% Price)',
              title='S&P500 VIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])

# [Figure 20] Interest Rate VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
vix_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(joined_index - joined_undl_rv, hist=False,
                            label='S&P500 VIX',
                            xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
[joined_index, joined_undl_rv] = \
    share_dateindex([tyvix.price(), 100*tyvix.undl_realized_vol(do_shift=True)])
tyvix_diff = joined_index - joined_undl_rv
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='TYVIX', xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C1', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([jgbvix.price(), 100*jgbvix.undl_realized_vol(do_shift=True)])
jgbvix_diff = joined_index - joined_undl_rv
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='JGB VIX', xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C2', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([srvix.price(), srvix.undl_realized_vol(do_shift=True, window=252, bps=True)])
srvix_diff = joined_index - joined_undl_rv
make_histogram(joined_index - joined_undl_rv, hist=False,
               label='SRVIX', xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C3', ax=ax_prem)

# [Table] Interest Rate VIX Risk Premium Tails Table
names_list = ['VIX', 'TYVIX', 'JGB VIX', 'SRVIX']
diff_list = [vix_diff, tyvix_diff, jgbvix_diff, srvix_diff]
# one_percent = list(map(lambda d: d.quantile(0.01), diff_list))
# five_percent = list(map(lambda d: d.quantile(0.05), diff_list))
# ten_percent = list(map(lambda d: d.quantile(0.1), diff_list))
one_percent_mean = list(map(lambda d: d[d.rank(pct=True) < 0.01].mean(), diff_list))
five_percent_mean = list(map(lambda d: d[d.rank(pct=True) < 0.05].mean(), diff_list))
ten_percent_mean = list(map(lambda d: d[d.rank(pct=True) < 0.1].mean(), diff_list))
irvix_risk_prem_lefttail_table = \
    pd.DataFrame({'Vol Index': names_list,
                  'Mean of 1% Left Tail': one_percent_mean,
                  'Mean of 5% Left Tail': five_percent_mean,
                  'Mean of 10% Left Tail': ten_percent_mean}).set_index('Vol Index')
irvix_risk_prem_lefttail_table.to_csv('Risk Premium Left Tail Table for Interest Rate VIX.csv')

# [Figure 21] Credit VIX Scatter Matrix
instr_list = [vix, vstoxx, vixig, vixhy, vixie, vixxo, vixfs]
data_list = list(map(lambda instr: instr.price_return(), instr_list))
color_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
label_list = list(map(lambda instr: instr.name, instr_list))
make_scatter_matrix(data_list, color_list, label_list=label_list, title='Credit VIX Scatter Matrix')

# [Figure 22] Credit VIX Correlation Matrix
make_correlation_matrix(data_list, color_list, label_list=label_list, title='Credit VIX Correlation Matrix')

# [Figure 23] Interest Rate VIX Scatter Matrix
instr_list = [vix, tyvix, jgbvix, srvix]
data_list = list(map(lambda instr: instr.price_return(), instr_list))
color_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
label_list = list(map(lambda instr: instr.name, instr_list))
make_scatter_matrix(data_list, color_list, label_list=label_list, title='Interest Rate VIX Scatter Matrix')

# [Figure 24] Interest Rate VIX Correlation Matrix
make_correlation_matrix(data_list, color_list, label_list=label_list,
                        title='Interest Rate VIX Correlation Matrix')
