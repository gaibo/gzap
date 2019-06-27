import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest

from data_structures import ETF, Index, VolatilityIndex


def make_lineplot(data_list, label_list=None, xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare labels
    if label_list is None:
        label_list = len(data_list) * ['']
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Plot
    for data, label in zip_longest(data_list, label_list):
        ax.plot(data, label=label, linewidth=3)
    ax.legend()
    ax.grid()
    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def make_basicstatstable(instr_list):
    table_out = pd.DataFrame()
    for instr in instr_list:
        prices = instr.price()
        pct_changes = instr.price_return(logarithmic=False)
        instr_table = \
            pd.concat([pd.DataFrame({instr.name: [prices.first_valid_index(),
                                                  prices.last_valid_index()]},
                                    index=['start date', 'end date']).T,
                       prices.describe().rename(instr.name).iloc[0:1].to_frame().T.astype(int),
                       prices.describe().rename(instr.name).iloc[1:].to_frame().T,
                       pct_changes.describe().rename(instr.name).iloc[1:].to_frame().T
                       .add_suffix(' change (%)') * 100],
                      axis=1)
        table_out = table_out.append(instr_table)
    return table_out


def main():
    # Import data sources with pandas
    full_data = pd.read_csv('index_data.csv', index_col='Date', parse_dates=True)
    eurostoxx_data = pd.read_csv('STOXX50E.csv', index_col='Date', parse_dates=True)
    sptr_data = pd.read_csv('sptr_vix.csv', index_col='Date', parse_dates=True)

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
    jgbvix = VolatilityIndex(full_data['SPJGBV.Index'], None, 'JGB VIX')    # Need futures data
    tyvix = VolatilityIndex(full_data['TYVIX.Index'], None, 'TYVIX')  # Need futures data

    # Execute organized commands to create desired analyses

    # Line Plots
    truncd_spx = spx.price().truncate('2004-01-01', '2018-01-01')
    truncd_agg = agg.price().truncate('2004-01-01', '2018-01-01')
    truncd_lqd = lqd.price().truncate('2004-01-01', '2018-01-01')
    truncd_jgbvix = jgbvix.price().truncate('2004-01-01', '2018-01-01')
    truncd_tyvix = tyvix.price().truncate('2004-01-01', '2018-01-01')
    make_lineplot([truncd_spx/truncd_spx[0], truncd_agg/truncd_agg[0], truncd_jgbvix/truncd_jgbvix[0],
                   truncd_lqd/truncd_lqd[0], truncd_tyvix/truncd_tyvix[0]],
                  ['SPX', 'AGG'],
                  ylabel='Normalized Level', title='SPX and AGG')

    # Tables
    example_table = make_basicstatstable([spx, vix, hyg, vxhyg, ief, vxief, sx5e, vstoxx]).round(1)
    print(example_table)

    return 0


if __name__ == '__main__':
    main()
