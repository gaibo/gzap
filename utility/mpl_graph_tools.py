import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest, combinations
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from model.data_structures import ETF, Index, VolatilityIndex
from utility.universal_tools import share_dateindex, get_best_fit

register_matplotlib_converters()


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


def make_lineplot(data_list, label_list=None, color_list=None,
                  xlabel=None, ylabel=None, title=None, ax=None):
    # If non-list single elements are passed in, convert
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(label_list, list) and label_list is not None:
        label_list = [label_list]
    if not isinstance(color_list, list) and color_list is not None:
        color_list = [color_list]
    # Prepare labels and colors
    none_list = len(data_list) * [None]
    if label_list is None:
        label_list = none_list
    if color_list is None:
        color_list = none_list
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Create lines
    for data, label, color in zip_longest(data_list, label_list, color_list):
        ax.plot(data, label=label, color=color, linewidth=1.5)
    # Configure
    if label_list != none_list:
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


def make_fillbetween(x, y1, y2=0, label=None, color=None,
                     xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Create filled line plot
    ax.fill_between(x, y1, y2, color=color, label=label)
    # Configure
    if label:
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


def make_regime(interval_list, label=None, color=None,
                xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Create regimes from list of intervals
    span_handle = None
    for interval in interval_list:
        span_handle = ax.axvspan(*interval, color=color, alpha=0.5)
    # Configure
    if label:
        span_handle.set_label(label)    # Set label using last span's handle to avoid duplicates
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


def make_scatterplot(x_data, y_data, do_center=False, color=None,
                     xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Create scatter
    [joined_x_data, joined_y_data] = share_dateindex([x_data, y_data])  # Scatter must have matching x-y
    ax.scatter(joined_x_data, joined_y_data, c=color, zorder=3)
    ax.axhline(y=0, color='k', linewidth=1, zorder=2)
    ax.axvline(x=0, color='k', linewidth=1, zorder=2)
    # Create best fit line
    _, slope, _ = get_best_fit(x_data, y_data, fit_intercept=False)
    ax.plot(joined_x_data, slope*joined_x_data, 'k', zorder=4)
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


def make_histogram(data, hist=True, n_bins=10, line=True, label=None, color=None, color_line=None,
                   xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Create distribution histogram
    if hist:
        ax.hist(data, bins=n_bins, density=True, label=label, color=color)
    # Create density plot line
    if line:
        if isinstance(data, list):
            print("ERROR: can't make multiple density plot lines at same time.")
        else:
            sns.distplot(data, hist=False, ax=ax, label=label, color=color_line)
    # Configure
    if label:
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


def make_scatter_matrix(data_list, color_list, label_list=None, title=None):
    n_data = len(data_list)
    n_colors = len(color_list)
    # Prepare labels
    if label_list is None:
        label_list = n_data * [None]
    # Prepare coloring pattern
    color_dict = {}
    i = 0
    for combo in combinations(label_list, 2):
        color_dict[frozenset(combo)] = color_list[i]
        i = (i + 1) % n_colors
    # Create figure and subplots
    fig, axs = plt.subplots(n_data, n_data)
    for x_pos, data_x in enumerate(data_list):
        for y_pos, data_y in enumerate(data_list):
            # Use labels if subplot is on an axis
            xlabel = None
            ylabel = None
            if x_pos == 0:
                ylabel = label_list[y_pos]
            if y_pos == 0:
                xlabel = label_list[x_pos]
            # Convert x and y to row and col (y bottom to top -> matrix row top to bottom)
            row = n_data-1 - y_pos
            col = x_pos
            if x_pos > y_pos:
                # Bottom triangle - scatter
                make_scatterplot(data_x, data_y, ax=axs[row, col],
                                 color=color_dict[frozenset({label_list[x_pos], label_list[y_pos]})],
                                 xlabel=xlabel, ylabel=ylabel)
            elif y_pos > x_pos:
                # Top triangle - text of best fit line slope and R^2
                # NOTE: reverse x-y to match scatter slope
                r_sq, slope, _ = get_best_fit(data_y, data_x, fit_intercept=False)
                text_str = "$Slope$: {}\n$R^2$: {}".format(round(slope, 2), round(r_sq, 2))
                axs[row, col].text(0.3, 0.4, text_str, fontsize=12,
                                   color=color_dict[frozenset({label_list[x_pos], label_list[y_pos]})])
                axs[row, col].xaxis.set_ticks([])
                axs[row, col].yaxis.set_ticks([])
                if xlabel:
                    axs[row, col].set_xlabel(xlabel)
                if ylabel:
                    axs[row, col].set_ylabel(ylabel)
            else:
                # Diagonal - distribution
                make_histogram(data_x, hist=True, n_bins=100, line=False,
                               ax=axs[row, col], xlabel=xlabel, ylabel=ylabel)
    fig.suptitle(title)
    return fig, axs


def make_correlation_matrix(data_list, color_list, label_list=None, title=None):
    n_data = len(data_list)
    n_colors = len(color_list)
    # Prepare labels
    if label_list is None:
        label_list = n_data * [None]
    # Prepare coloring pattern
    color_dict = {}
    i = 0
    for combo in combinations(label_list, 2):
        color_dict[frozenset(combo)] = color_list[i]
        i = (i + 1) % n_colors
    # Create figure and subplots
    fig, axs = plt.subplots(n_data, n_data)
    for x_pos, data_x in enumerate(data_list):
        for y_pos, data_y in enumerate(data_list):
            # Use labels if subplot is on an axis
            xlabel = None
            ylabel = None
            if x_pos == 0:
                ylabel = label_list[y_pos]
            if y_pos == 0:
                xlabel = label_list[x_pos]
            # Convert x and y to row and col (y bottom to top -> matrix row top to bottom)
            row = n_data-1 - y_pos
            col = x_pos
            # Calculate 6-month rolling correlation
            [joined_data_x, joined_data_y] = share_dateindex([data_x, data_y])
            six_month_rolling_corr = joined_data_x.rolling(6 * 21).corr(joined_data_y).dropna()
            if x_pos > y_pos:
                # Bottom triangle - rolling correlation plot
                make_lineplot([six_month_rolling_corr],
                              color_list=[color_dict[frozenset({label_list[x_pos], label_list[y_pos]})]],
                              ax=axs[row, col], xlabel=xlabel, ylabel=ylabel)
                axs[row, col].axhline(y=0, color='k')
                axs[row, col].xaxis.set_major_locator(plt.MaxNLocator(3))
            elif y_pos > x_pos:
                # Top triangle - text of mean rolling correlation
                mean_rolling_corr = six_month_rolling_corr.mean()
                text_str = "$Mean$ $Corr$: {}%".format(round(mean_rolling_corr*100))
                axs[row, col].text(0.2, 0.4, text_str, fontsize=12,
                                   color=color_dict[frozenset({label_list[x_pos], label_list[y_pos]})])
                axs[row, col].xaxis.set_ticks([])
                axs[row, col].yaxis.set_ticks([])
                if xlabel:
                    axs[row, col].set_xlabel(xlabel)
                if ylabel:
                    axs[row, col].set_ylabel(ylabel)
            else:
                # Diagonal - do nothing except show labels
                axs[row, col].xaxis.set_ticks([])
                axs[row, col].yaxis.set_ticks([])
                if xlabel:
                    axs[row, col].set_xlabel(xlabel)
                if ylabel:
                    axs[row, col].set_ylabel(ylabel)
    fig.suptitle(title)
    return fig, axs


def main():
    # Import data sources with pandas
    bbg_data = pd.read_csv('../data/bbg_automated_pull.csv',
                           index_col=0, parse_dates=True, header=[0, 1])

    # Create data objects
    spx = Index(bbg_data[('SPX Index', 'PX_LAST')], 'SPX')
    vix = VolatilityIndex(bbg_data[('VIX Index', 'PX_LAST')], spx, 'VIX')
    ty1 = Index(bbg_data[('TY1 Comdty', 'PX_LAST')], 'TY1')
    tyvix = VolatilityIndex(bbg_data[('TYVIX Index', 'PX_LAST')], ty1, 'TYVIX')
    jb1 = Index(bbg_data[('JB1 Comdty', 'PX_LAST')], 'JB1')
    jgbvix = VolatilityIndex(bbg_data[('SPJGBV Index', 'PX_LAST')], jb1, 'JGB VIX')
    sx5e_tr = Index(bbg_data[('SX5E Index', 'TOT_RETURN_INDEX_GROSS_DVDS')], 'Euro Stoxx 50')
    agg_tr = ETF(bbg_data[('AGG Equity', 'TOT_RETURN_INDEX_GROSS_DVDS')], 'AGG')
    hyg_tr = ETF(bbg_data[('HYG Equity', 'TOT_RETURN_INDEX_GROSS_DVDS')], 'HYG')
    ief_tr = ETF(bbg_data[('IEF Equity', 'TOT_RETURN_INDEX_GROSS_DVDS')], 'IEF')
    lqd_tr = ETF(bbg_data[('LQD Equity', 'TOT_RETURN_INDEX_GROSS_DVDS')], 'LQD')

    # Name some truncated timeseries
    start_date = '2004-01-01'
    end_date = '2018-01-01'
    truncd_vix = vix.price().truncate(start_date, end_date)
    truncd_spx = spx.price().truncate(start_date, end_date)
    truncd_agg = agg_tr.price().truncate(start_date, end_date)
    truncd_lqd = lqd_tr.price().truncate(start_date, end_date)
    truncd_jgbvix = jgbvix.price().truncate(start_date, end_date)
    truncd_tyvix = tyvix.price().truncate(start_date, end_date)

    # Example data processing that can be done with framework

    # Line Plots
    make_lineplot([truncd_spx/truncd_spx[0], truncd_agg/truncd_agg[0], truncd_jgbvix/truncd_jgbvix[0],
                   truncd_lqd/truncd_lqd[0], truncd_tyvix/truncd_tyvix[0]],
                  ['SPX', 'AGG'],
                  ylabel='Normalized Level', title='SPX and AGG')

    # Tables
    example_table = make_basicstatstable([spx, vix, hyg_tr, ief_tr, sx5e_tr]).round(1)
    print(example_table)

    # Scatterplots
    make_scatterplot(spx.price_return(logarithmic=False), vix.price_return(logarithmic=False), False, None,
                     '% Change in SPX', '% Change in VIX', '% Change: SPX vs. VIX')

    # Line Plots with Difference
    truncd_realvol = 100 * vix.undl_realized_vol(do_shift=True).truncate(start_date, end_date)
    [joined_vix, joined_realvol] = share_dateindex([truncd_vix, truncd_realvol])
    fig, axs = plt.subplots(2, 1, sharex='all')
    make_lineplot([joined_vix, joined_realvol],
                  ['VIX Level', 'SPX Realized Vol (21 Days Shifted)'], ax=axs[0])
    difference = joined_vix - joined_realvol
    make_fillbetween(difference.index, joined_vix, joined_realvol, label='Difference', color='g', ax=axs[0])
    make_fillbetween(difference.index, difference, label='Difference', color='g', ax=axs[1])

    # Vol Regimes
    fig, ax = plt.subplots()
    make_lineplot(vix.price(), 'VIX', ax=ax)
    make_regime(vix.vol_regime()[2], 'High Vol Regime', 'r', 'Date', 'Index Level', 'VIX Vol Regimes', ax=ax)
    make_regime(vix.vol_regime()[1], 'Low Vol Regime', 'g', 'Date', 'Index Level', 'VIX Vol Regimes', ax=ax)
    ax.autoscale(enable=True, axis='x', tight=True)

    # Histograms
    [joined_jgbvix, joined_jb1_rv] = \
        share_dateindex([jgbvix.price(), jgbvix.undl_realized_vol(do_shift=True) * 100])
    jgbvix_jb1_rv_diff = joined_jgbvix - joined_jb1_rv
    fig, ax = make_histogram([difference, jgbvix_jb1_rv_diff], hist=True, n_bins=100, line=False,
                             label=['VIX', 'JGB VIX'], xlabel='Vol Points', title='Risk Premium')
    make_histogram(difference, hist=False, line=True, label='VIX', ax=ax)
    make_histogram(jgbvix_jb1_rv_diff, hist=False, line=True, label='JGB VIX', ax=ax)

    # Scatter and Correlation Matrix
    instr_list = [spx, vix, hyg_tr, ief_tr, sx5e_tr, agg_tr]
    data_list = list(map(lambda instr: instr.price_return(), instr_list))
    color_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    label_list = list(map(lambda instr: instr.name, instr_list))
    make_scatter_matrix(data_list, color_list, label_list=label_list)
    make_correlation_matrix(data_list, color_list, label_list=label_list)

    return 0


if __name__ == '__main__':
    main()
