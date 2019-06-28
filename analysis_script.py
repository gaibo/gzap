import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest
from sklearn.linear_model import LinearRegression
import seaborn as sns

from data_structures import ETF, Index, VolatilityIndex


def share_dateindex(timeseries_list):
    column_list = range(len(timeseries_list))
    combined_df = pd.DataFrame(dict(zip(column_list, timeseries_list))).dropna()
    return [combined_df[column].rename('value') for column in column_list]


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


def make_lineplot(data_list, label_list=None, xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare labels
    if label_list is None:
        label_list = len(data_list) * ['']
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Create lines
    for data, label in zip_longest(data_list, label_list):
        ax.plot(data, label=label, linewidth=3)
    # Configure
    ax.legend()
    ax.grid(True)
    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def make_fillbetween(x, y1, y2=0, label=None, xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Create filled line plot
    ax.fill_between(x, y1, y2, color='g', label=label)
    ax.legend()
    ax.grid(True)
    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def make_scatterplot(x_data, y_data, do_center=False,
                     xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Create scatter
    [joined_x_data, joined_y_data] = share_dateindex([x_data, y_data])
    ax.scatter(joined_x_data, joined_y_data)
    # Create best fit line
    x = joined_x_data.values.reshape(-1, 1)
    y = joined_y_data.values
    model = LinearRegression(fit_intercept=False).fit(x, y)
    ax.plot(joined_x_data, model.coef_*joined_x_data, 'k')
    # Configure
    ax.grid(True)
    if do_center:
        lim = max(max(joined_x_data.abs()), max(joined_y_data.abs()))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


def make_histogram(data, hist=True, n_bins=10, line=True,
                   label=None, xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Create distribution histogram
    if hist:
        ax.hist(data, bins=n_bins, density=True, label=label)
    # Create density plot line
    if line:
        sns.distplot(data, hist=False, ax=ax, label=label)
    # Configure
    ax.legend()
    ax.grid(True)
    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax


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

    # Name some truncated timeseries
    start_date = '2004-01-01'
    end_date = '2018-01-01'
    truncd_vix = vix.price().truncate(start_date, end_date)
    truncd_spx = spx.price().truncate(start_date, end_date)
    truncd_agg = agg.price().truncate(start_date, end_date)
    truncd_lqd = lqd.price().truncate(start_date, end_date)
    truncd_jgbvix = jgbvix.price().truncate(start_date, end_date)
    truncd_tyvix = tyvix.price().truncate(start_date, end_date)

    # Example data processing that can be done with framework

    # Line Plots
    make_lineplot([truncd_spx/truncd_spx[0], truncd_agg/truncd_agg[0], truncd_jgbvix/truncd_jgbvix[0],
                   truncd_lqd/truncd_lqd[0], truncd_tyvix/truncd_tyvix[0]],
                  ['SPX', 'AGG'],
                  ylabel='Normalized Level', title='SPX and AGG')

    # Tables
    example_table = make_basicstatstable([spx, vix, hyg, vxhyg, ief, vxief, sx5e, vstoxx]).round(1)
    print(example_table)

    # Scatterplots
    make_scatterplot(spx.price_return(logarithmic=False), vix.price_return(logarithmic=False), False,
                     '% Change in SPX', '% Change in VIX', '% Change: SPX vs. VIX')

    # Line Plots with Difference
    truncd_realvol = 100 * vix.undl_realized_vol(do_shift=True).truncate(start_date, end_date)
    [joined_vix, joined_realvol] = share_dateindex([truncd_vix, truncd_realvol])
    fig, axs = plt.subplots(2, 1, sharex='all')
    make_lineplot([joined_vix, joined_realvol],
                  ['VIX Level', 'SPX Realized Vol (21 Days Shifted)'], ax=axs[0])
    difference = joined_vix - joined_realvol
    make_fillbetween(difference.index, joined_vix, joined_realvol,
                     label='Difference', ax=axs[0])
    make_fillbetween(difference.index, difference, label='Difference', ax=axs[1])

    # Histograms
    [joined_vxhyg, joined_realhygvol] = share_dateindex([vxhyg.price(),
                                                         vxhyg.undl_realized_vol(do_shift=True) * 100])
    hyg_voldiff = joined_vxhyg - joined_realhygvol
    fig, ax = make_histogram([difference, hyg_voldiff], hist=True, n_bins=100, line=False,
                             label=['VIX', 'VXHYG'], xlabel='Vol Points', title='Risk Premium')
    make_histogram(difference, hist=False, line=True, label='VIX', ax=ax)
    make_histogram(hyg_voldiff, hist=False, line=True, label='VXHYG', ax=ax)

    return 0


if __name__ == '__main__':
    main()
