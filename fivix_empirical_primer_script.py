import pandas as pd
import matplotlib.pyplot as plt

from model.data_structures import ETF, Index, VolatilityIndex
from utility.graph_utilities import \
    share_dateindex, make_basicstatstable, make_lineplot, make_fillbetween,\
    make_scatterplot, make_histogram


def main():
    # Import data sources with pandas
    bbg_data = pd.read_csv('data/bbg_automated_pull.csv',
                           index_col=0, parse_dates=True, header=[0, 1])
    creditvix_data = pd.read_csv('data/creditvix_pc_bp_missing_4_months.csv',
                                 index_col='Date', parse_dates=True)
    tyvix_bp_data = pd.read_csv('data/tyvix_bp_data.csv',
                                index_col='Trade Date', parse_dates=True)

    # Create data objects
    agg_tr = ETF(bbg_data['AGG Equity']['TOT_RETURN_INDEX_GROSS_DVDS'], 'AGG')
    hyg_tr = ETF(bbg_data['HYG Equity']['TOT_RETURN_INDEX_GROSS_DVDS'], 'HYG')
    ief_tr = ETF(bbg_data['IEF Equity']['TOT_RETURN_INDEX_GROSS_DVDS'], 'IEF')
    lqd_tr = ETF(bbg_data['LQD Equity']['TOT_RETURN_INDEX_GROSS_DVDS'], 'LQD')
    spx_tr = Index(bbg_data['SPX Index']['TOT_RETURN_INDEX_GROSS_DVDS'], 'SPX')
    sx5e_tr = Index(bbg_data['SX5E Index']['TOT_RETURN_INDEX_GROSS_DVDS'], 'Euro Stoxx 50')
    vixig = VolatilityIndex(creditvix_data['VIXIG Percent'], None, 'VIXIG')
    vixig_bp = VolatilityIndex(creditvix_data['VIXIG Basis Point'], None, 'BP VIXIG')
    vixhy = VolatilityIndex(creditvix_data['VIXHY Percent'], None, 'VIXHY')
    vixhy_bp = VolatilityIndex(creditvix_data['VIXHY Basis Point'], None, 'BP VIXHY')
    vixie = VolatilityIndex(creditvix_data['VIXIE Percent'], None, 'VIXIE')
    vixie_bp = VolatilityIndex(creditvix_data['VIXIE Basis Point'], None, 'BP VIXIE')
    vixxo = VolatilityIndex(creditvix_data['VIXXO Percent'], None, 'VIXXO')
    vixxo_bp = VolatilityIndex(creditvix_data['VIXXO Basis Point'], None, 'BP VIXXO')
    vixfs = VolatilityIndex(creditvix_data['VIXFS Percent'], None, 'VIXFS')
    vixfs_bp = VolatilityIndex(creditvix_data['VIXFS Basis Point'], None, 'BP VIXFS')
    jgbvix = VolatilityIndex(bbg_data['SPJGBV Index']['PX_LAST'], None, 'JGB VIX')
    srvix = VolatilityIndex(bbg_data['SRVIX Index']['PX_LAST'], None, 'SRVIX')
    tyvix = VolatilityIndex(bbg_data['TYVIX Index']['PX_LAST'], None, 'TYVIX')
    tyvix_bp = VolatilityIndex(tyvix_bp_data['BP TYVIX'], None, 'BP TYVIX')
    vix = VolatilityIndex(bbg_data['VIX Index']['PX_LAST'], spx_tr, 'VIX')

    # FI VIX Empirical Primer Document Reproduction

    # S&P 500 Index and AGG Total Return
    [truncd_spx, truncd_agg] = share_dateindex([spx_tr.price(), agg_tr.price()])
    make_lineplot([truncd_spx/truncd_spx[0], truncd_agg/truncd_agg[0]],
                  ['SPX total return', 'AGG total return'],
                  ylabel='Normalized Level', title='S&P 500 Index and AGG Total Return')

    # North American Credit VIXs with VIX Index
    make_lineplot([vix.price(), vixig.price(), vixhy.price()],
                  ['S&P 500 VIX', 'VIXIG', 'VIXHY'],
                  ylabel='Volatility Index', title='North American Credit VIXs with VIX Index')

    # European Credit VIXs with VIX Index
    make_lineplot([vix.price(), vixie.price(), vixxo.price()],
                  ['S&P 500 VIX', 'VIXIE', 'VIXXO'],
                  ylabel='Volatility Index', title='European Credit VIXs with VIX Index')

    # VIXs in Rates Group with VIX Index
    make_lineplot([vix.price(), tyvix.price(), jgbvix.price(), srvix.price()],
                  ['S&P 500 VIX', 'TYVIX', 'JGB VIX', 'SRVIX'],
                  ylabel='Volatility Index (% Price)', title='VIXs in Rates Group with VIX Index')

    # Basic Statistics for Credit VIX Indexes
    make_basicstatstable([vix, vixig, vixhy, vixie, vixxo]) \
        .to_csv('Basic Statistics for Credit VIX Indexes.csv')

    # Basic Statistics for Basis Point Credit VIX Indexes
    make_basicstatstable([vix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp]) \
        .to_csv('Basic Statistics for Basis Point Credit VIX Indexes.csv')

    # Basic Statistics for Interest Rate VIX Indexes
    make_basicstatstable([vix, tyvix, jgbvix, srvix]) \
        .to_csv('Basic Statistics for Interest Rate VIX Indexes.csv')

    # Basis Point Version of Credit Group
    make_lineplot([vixig_bp.price(), vixhy_bp.price(), vixie_bp.price(), vixxo_bp.price()],
                  ['BP VIXIG', 'BP VIXHY', 'BP VIXIE', 'BP VIXXO'],
                  ylabel='Volatility Index (bps)', title='Basis Point Version of Credit Group')

    # Basis Point Version of Rates Group
    # Missing basis point JGB VIX
    make_lineplot([tyvix_bp.price(), srvix.price()],
                  ['BP TYVIX', 'SRVIX'],
                  ylabel='Volatility Index (bps)', title='Basis Point Version of Rates Group')

    # VIXIG Daily % Change vs. CDX NAIG Index Daily bps Change
    # Missing CDX NAIG data
    # make_scatterplot(vixig.price_return(), cdx_naig.price().diff(),
    #                  xlabel='% Change', ylabel='bps Change',
    #                  title='VIXIG Daily % Change vs. CDX NAIG Index Daily bps Change')

    # SRVIX Daily Change vs. LQD Daily Change
    make_scatterplot(srvix.price_return(False), lqd_tr.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='SRVIX Daily % Change vs. LQD Daily % Change')

    # Four Scatter Plots
    fig, axs = plt.subplots(2, 2)
    make_scatterplot(vixhy_bp.price_return(False), hyg_tr.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='BP VIXHY Daily % Change vs. HYG Daily % Change',
                     ax=axs[0, 0])
    # make_scatterplot(vixxo_bp.price_return(False), ihyg.price_return(False),
    #                  xlabel='% Change', ylabel='% Change',
    #                  title='BP VIXXO Daily % Change vs. IHYG Daily % Change',
    #                  ax=axs[0, 1])
    make_scatterplot(tyvix_bp.price_return(False), ief_tr.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='BP TYVIX Daily % Change vs. IEF Daily % Change',
                     ax=axs[1, 0])
    make_scatterplot(srvix.price_return(False), ief_tr.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='SRVIX Daily % Change vs. IEF Daily % Change',
                     ax=axs[1, 1])

    # Four Difference Plots
    fig, axs = plt.subplots(2, 2)
    make_lineplot([vixig.price(), 100*vixig.undl_realized_vol(do_shift=True)],
                  ['VIXIG', 'Realized Volatility'],
                  xlabel='% Change', ylabel='% Change',
                  title='BP VIXHY Daily % Change vs. HYG Daily % Change',
                  ax=axs[0, 0])
    # make_scatterplot(vixxo_bp.price_return(False), ihyg.price_return(False),
    #                  xlabel='% Change', ylabel='% Change',
    #                  title='BP VIXXO Daily % Change vs. IHYG Daily % Change',
    #                  ax=axs[0, 1])
    make_scatterplot(tyvix_bp.price_return(False), ief_tr.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='BP TYVIX Daily % Change vs. IEF Daily % Change',
                     ax=axs[1, 0])
    make_scatterplot(srvix.price_return(False), ief_tr.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='SRVIX Daily % Change vs. IEF Daily % Change',
                     ax=axs[1, 1])
