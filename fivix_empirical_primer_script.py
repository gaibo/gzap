import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from model.data_structures import ETF, Index, VolatilityIndex
from utility.utilities import share_dateindex, make_basicstatstable, \
    make_lineplot, make_fillbetween, make_scatterplot, make_histogram


def main():
    # Import data sources with pandas
    full_data = pd.read_csv('data/index_data.csv', index_col='Date', parse_dates=True)
    eurostoxx_data = pd.read_csv('data/STOXX50E.csv', index_col='Date', parse_dates=True)
    sptr_data = pd.read_csv('data/sptr_vix.csv', index_col='Date', parse_dates=True)
    credit_vix_data = pd.read_csv('data/credit_vix.csv',
                                  usecols=['Date', 'IndexSymbol', 'CreditVix_pc', 'CreditVix_bp'],
                                  index_col='Date', parse_dates=True)
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

    # Create data objects
    spx = Index(sptr_data['SPTR'], 'SPX')
    vix = VolatilityIndex(full_data['VIX.Index'], spx, 'VIX')
    hyg = ETF(full_data['HYG.US.Equity'], 'HYG')
    vxhyg = VolatilityIndex(full_data['VXHYG.Index'], hyg, 'VXHYG')
    ief = ETF(full_data['IEF.US.Equity'], 'IEF')  # 7-10 year treasury bond ETF
    vxief = VolatilityIndex(full_data['VXHYG.Index'], ief, 'VXIEF')
    sx5e = Index(eurostoxx_data['Close'], 'Euro Stoxx 50')
    vstoxx = VolatilityIndex(full_data['V2X.Index'], sx5e, 'VSTOXX')
    agg = ETF(full_data['AGG.US.Equity'], 'AGG')
    lqd = ETF(full_data['LQD.US.Equity'], 'LQD')
    jgbvix = VolatilityIndex(full_data['SPJGBV.Index'], None, 'JGB VIX')  # Need futures data
    tyvix = VolatilityIndex(full_data['TYVIX.Index'], None, 'TYVIX')  # Need futures data
    vixfs = Index(vixfs_data['pc'], 'VIXFS')
    vixhy = Index(vixhy_data['pc'], 'VIXHY')
    vixig = Index(vixig_data['pc'], 'VIXIG')
    vixie = Index(vixie_data['pc'], 'VIXIE')
    vixxo = Index(vixxo_data['pc'], 'VIXXO')

    # FI VIX Empirical Primer Document Reproduction

    # SPX total return vs AGG total return
    [truncd_spx, truncd_agg] = share_dateindex([spx, agg])
    make_lineplot([truncd_spx / truncd_spx[0], truncd_agg / truncd_agg[0]]
                  ['SPX total return', 'AGG total return'],
                  ylabel='Normalized Level', title='S&P 500 Index and AGG total return')
