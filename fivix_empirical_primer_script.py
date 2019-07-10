import pandas as pd
import matplotlib.pyplot as plt

from model.data_structures import ETF, Index, VolatilityIndex
from utility.graph_utilities import \
    share_dateindex, make_basicstatstable, make_lineplot, make_fillbetween,\
    make_scatterplot, make_histogram


def main():
    # Import data sources with pandas
    agg_data = pd.read_csv('data/bbg_agg_data.csv', index_col='Date', parse_dates=True)
    hyg_data = pd.read_csv('data/bbg_hyg_data.csv', index_col='Date', parse_dates=True)
    ief_data = pd.read_csv('data/bbg_ief_data.csv', index_col='Date', parse_dates=True)
    lqd_data = pd.read_csv('data/bbg_lqd_data.csv', index_col='Date', parse_dates=True)
    spx_data = pd.read_csv('data/bbg_spx_data.csv', index_col='Date', parse_dates=True)
    sx5e_data = pd.read_csv('data/bbg_sx5e_data.csv', index_col='Date', parse_dates=True)
    credit_vix_data = pd.read_csv('data/creditvix_data.csv',
                                  usecols=['Date', 'IndexSymbol', 'CreditVix_pc', 'CreditVix_bp'],
                                  index_col='Date', parse_dates=True)
    jgbvix_data = pd.read_csv('data/jgbvix_data.csv', index_col='Date', parse_dates=True)
    misc_vol_data = pd.read_csv('data/misc_vol_data.csv', index_col='Date', parse_dates=True)
    srvix_data = pd.read_csv('data/srvix_data.csv', index_col='Date', parse_dates=True)
    tyvix_data = pd.read_csv('data/tyvix_data.csv', index_col='Date', parse_dates=True)
    tyvix_bp_data = pd.read_csv('data/tyvix_bp_data.csv', index_col='Trade Date', parse_dates=True)
    vix_data = pd.read_csv('data/vix_data.csv', index_col='Date', parse_dates=True)

    # Create data objects
    agg = ETF(agg_data['TOT_RETURN_INDEX_GROSS_DVDS'], 'AGG')
    hyg = ETF(hyg_data['TOT_RETURN_INDEX_GROSS_DVDS'], 'HYG')
    ief = ETF(ief_data['TOT_RETURN_INDEX_GROSS_DVDS'], 'IEF')  # 7-10 year treasury bond ETF
    lqd = ETF(lqd_data['TOT_RETURN_INDEX_GROSS_DVDS'], 'LQD')
    spx = Index(spx_data['TOT_RETURN_INDEX_GROSS_DVDS'], 'SPX')
    sx5e = Index(sx5e_data['TOT_RETURN_INDEX_GROSS_DVDS'], 'Euro Stoxx 50')
    vixfs_data = \
        credit_vix_data[credit_vix_data['IndexSymbol'] == 'VIXFS'] \
        .drop('IndexSymbol', axis=1).rename({'CreditVix_pc': 'pc', 'CreditVix_bp': 'bp'}, axis=1)
    vixhy_data = \
        credit_vix_data[credit_vix_data['IndexSymbol'] == 'VIXHY'] \
        .drop('IndexSymbol', axis=1).rename({'CreditVix_pc': 'pc', 'CreditVix_bp': 'bp'}, axis=1)
    vixig_data = \
        credit_vix_data[credit_vix_data['IndexSymbol'] == 'VIXIG'] \
        .drop('IndexSymbol', axis=1).rename({'CreditVix_pc': 'pc', 'CreditVix_bp': 'bp'}, axis=1)
    vixie_data = \
        credit_vix_data[credit_vix_data['IndexSymbol'] == 'VIXIE'] \
        .drop('IndexSymbol', axis=1).rename({'CreditVix_pc': 'pc', 'CreditVix_bp': 'bp'}, axis=1)
    vixxo_data = \
        credit_vix_data[credit_vix_data['IndexSymbol'] == 'VIXXO'] \
        .drop('IndexSymbol', axis=1).rename({'CreditVix_pc': 'pc', 'CreditVix_bp': 'bp'}, axis=1)
    vixfs = VolatilityIndex(vixfs_data['pc'], None, 'VIXFS')
    vixhy = VolatilityIndex(vixhy_data['pc'], None, 'VIXHY')
    vixig = VolatilityIndex(vixig_data['pc'], None, 'VIXIG')
    vixie = VolatilityIndex(vixie_data['pc'], None, 'VIXIE')
    vixxo = VolatilityIndex(vixxo_data['pc'], None, 'VIXXO')
    vixfs_bp = VolatilityIndex(vixfs_data['bp'], None, 'BP VIXFS')
    vixig_bp = VolatilityIndex(vixig_data['bp'], None, 'BP VIXIG')
    vixhy_bp = VolatilityIndex(vixhy_data['bp'], None, 'BP VIXHY')
    vixie_bp = VolatilityIndex(vixie_data['bp'], None, 'BP VIXIE')
    vixxo_bp = VolatilityIndex(vixxo_data['bp'], None, 'BP VIXXO')
    jgbvix = VolatilityIndex(jgbvix_data['SPJGBVIX.Index'], None, 'JGB VIX')  # Need futures data
    vxhyg = VolatilityIndex(misc_vol_data['VXHYG.Index'], hyg, 'VXHYG')
    vxief = VolatilityIndex(misc_vol_data['VXIEF.Index'], ief, 'VXIEF')
    vstoxx = VolatilityIndex(misc_vol_data['V2X.Index'], sx5e, 'VSTOXX')
    srvix = VolatilityIndex(srvix_data, None, 'SRVIX')  # Need swaptions data
    tyvix = VolatilityIndex(tyvix_data['Close'], None, 'TYVIX')  # Need futures data
    tyvix_bp = VolatilityIndex(tyvix_bp_data['BP TYVIX'], None, 'BP TYVIX')  # Need futures data
    vix = VolatilityIndex(vix_data['VIX.Index'], spx, 'VIX')

    # FI VIX Empirical Primer Document Reproduction

    # S&P 500 Index and AGG Total Return
    [truncd_spx, truncd_agg] = share_dateindex([spx.price(), agg.price()])
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
    make_scatterplot(srvix.price_return(False), lqd.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='SRVIX Daily % Change vs. LQD Daily % Change')

    # Four Scatter Plots
    fig, axs = plt.subplots(2, 2)
    make_scatterplot(vixhy_bp.price_return(False), hyg.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='BP VIXHY Daily % Change vs. HYG Daily % Change',
                     ax=axs[0, 0])
    # make_scatterplot(vixxo_bp.price_return(False), ihyg.price_return(False),
    #                  xlabel='% Change', ylabel='% Change',
    #                  title='BP VIXXO Daily % Change vs. IHYG Daily % Change',
    #                  ax=axs[0, 1])
    make_scatterplot(tyvix_bp.price_return(False), ief.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='BP TYVIX Daily % Change vs. IEF Daily % Change',
                     ax=axs[1, 0])
    make_scatterplot(srvix.price_return(False), ief.price_return(False),
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
    make_scatterplot(tyvix_bp.price_return(False), ief.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='BP TYVIX Daily % Change vs. IEF Daily % Change',
                     ax=axs[1, 0])
    make_scatterplot(srvix.price_return(False), ief.price_return(False),
                     xlabel='% Change', ylabel='% Change',
                     title='SRVIX Daily % Change vs. IEF Daily % Change',
                     ax=axs[1, 1])
