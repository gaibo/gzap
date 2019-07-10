import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from model.data_structures import ETF, Index, VolatilityIndex
from utility.graph_utilities import share_dateindex, make_basicstatstable, \
    make_lineplot, make_fillbetween, make_scatterplot, make_histogram


def main():
    # Import data sources with pandas
    full_data = pd.read_csv('data/price_index_data.csv', index_col='Date', parse_dates=True)
    eurostoxx_data = pd.read_csv('data/sx5e_data.csv', index_col='Date', parse_dates=True)
    sptr_data = pd.read_csv('data/sptr_vix_data.csv', index_col='Date', parse_dates=True)
    agg_data = pd.read_csv('data/agg_data.csv', index_col='Date', parse_dates=True)
    credit_vix_data = pd.read_csv('data/creditvix_data.csv',
                                  usecols=['Date', 'IndexSymbol', 'CreditVix_pc', 'CreditVix_bp'],
                                  index_col='Date', parse_dates=True)
    srvix_data = pd.read_csv('data/srvix_data.csv', index_col='Date', parse_dates=True)

    # Create data objects
    spx = Index(sptr_data['SPTR'], 'SPX')
    vix = VolatilityIndex(full_data['VIX.Index'], spx, 'VIX')
    hyg = ETF(full_data['HYG.US.Equity'], 'HYG')
    vxhyg = VolatilityIndex(full_data['VXHYG.Index'], hyg, 'VXHYG')
    ief = ETF(full_data['IEF.US.Equity'], 'IEF')  # 7-10 year treasury bond ETF
    vxief = VolatilityIndex(full_data['VXHYG.Index'], ief, 'VXIEF')
    sx5e = Index(eurostoxx_data['TOT_RETURN_INDEX_GROSS_DVDS'], 'Euro Stoxx 50')
    vstoxx = VolatilityIndex(full_data['V2X.Index'], sx5e, 'VSTOXX')
    agg = ETF(agg_data['TOT_RETURN_INDEX_GROSS_DVDS'], 'AGG')
    lqd = ETF(full_data['LQD.US.Equity'], 'LQD')
    jgbvix = VolatilityIndex(full_data['SPJGBV.Index'], None, 'JGB VIX')  # Need futures data
    tyvix = VolatilityIndex(full_data['TYVIX.Index'], None, 'TYVIX')  # Need futures data
    srvix = VolatilityIndex(srvix_data, None, 'SRVIX')  # Need swaptions data
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
    vixhy_bp = VolatilityIndex(vixhy_data['bp'], None, 'BP VIXHY')
    vixig_bp = VolatilityIndex(vixig_data['bp'], None, 'BP VIXIG')
    vixie_bp = VolatilityIndex(vixie_data['bp'], None, 'BP VIXIE')
    vixxo_bp = VolatilityIndex(vixxo_data['bp'], None, 'BP VIXXO')

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
    make_lineplot([vixig_bp.price(), vixhy_bp.price(), vixie.price(), vixxo.price()],
                  ['BP VIXIG', 'BP VIXHY', 'BP VIXIE', 'BP VIXXO'],
                  ylabel='Volatility Index (bps)', title='Basis Point Version of Credit Group')

    # Basis Point Version of Rates Group
    # Missing basis point JGB VIX

    #