import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest, combinations
from sklearn.linear_model import LinearRegression
import seaborn as sns

from model.data_structures import ETF, Index, VolatilityIndex


def share_dateindex(timeseries_list):
    column_list = range(len(timeseries_list))
    combined_df = pd.DataFrame(dict(zip(column_list, timeseries_list))).dropna()
    return [combined_df[column].rename('value') for column in column_list]


def get_best_fit(x_data, y_data, fit_intercept=True):
    [joined_x_data, joined_y_data] = share_dateindex([x_data, y_data])
    x = joined_x_data.values.reshape(-1, 1)
    y = joined_y_data.values
    model = LinearRegression(fit_intercept=fit_intercept).fit(x, y)
    r_sq = model.score(x, y)
    slope = model.coef_[0]
    return r_sq, slope, model


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
        ax.plot(data, label=label, color=color, linewidth=2)
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
    if label is not None:
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
    ax.scatter(joined_x_data, joined_y_data, c=color)
    # Create best fit line
    _, slope, _ = get_best_fit(x_data, y_data, fit_intercept=False)
    ax.plot(joined_x_data, slope*joined_x_data, 'k')
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


def make_histogram(data, hist=True, n_bins=10, line=True, label=None, color=None,
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
            sns.distplot(data, hist=False, ax=ax, label=label, color=color)
    # Configure
    if label is not None:
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


def make_correlation_matrix(data_list, color_list, label_list=None):
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
                # Top triangle - text
                # NOTE: reverse x-y to match scatter slope
                r_sq, slope, _ = get_best_fit(data_y, data_x, fit_intercept=False)
                text_str = "$Slope$: {}\n$R^2$: {}".format(round(slope, 2), round(r_sq, 2))
                axs[row, col].text(0.3, 0.4, text_str,
                                   color=color_dict[frozenset({label_list[x_pos], label_list[y_pos]})])
                if xlabel:
                    axs[row, col].set_xlabel(xlabel)
                if ylabel:
                    axs[row, col].set_ylabel(ylabel)
            else:
                # Diagonal - distribution
                make_histogram(data_x, hist=True, n_bins=100, line=False,
                               ax=axs[row, col], xlabel=xlabel, ylabel=ylabel)
    return fig, axs


def main():
    # Import data sources with pandas
    full_data = pd.read_csv('../data/price_index_data.csv', index_col='Date', parse_dates=True)
    eurostoxx_data = pd.read_csv('../data/sx5e_data.csv', index_col='Date', parse_dates=True)
    sptr_data = pd.read_csv('../data/sptr_vix_data.csv', index_col='Date', parse_dates=True)
    agg_data = pd.read_csv('../data/agg_data.csv', index_col='Date', parse_dates=True)

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

    # Histograms
    [joined_vxhyg, joined_realhygvol] = share_dateindex([vxhyg.price(),
                                                         vxhyg.undl_realized_vol(do_shift=True) * 100])
    hyg_voldiff = joined_vxhyg - joined_realhygvol
    fig, ax = make_histogram([difference, hyg_voldiff], hist=True, n_bins=100, line=False,
                             label=['VIX', 'VXHYG'], xlabel='Vol Points', title='Risk Premium')
    make_histogram(difference, hist=False, line=True, label='VIX', ax=ax)
    make_histogram(hyg_voldiff, hist=False, line=True, label='VXHYG', ax=ax)

    # Correlation Matrix
    instr_list = [spx, vix, hyg, ief, sx5e, agg]
    data_list = list(map(lambda instr: instr.price_return(), instr_list))
    color_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    label_list = list(map(lambda instr: instr.name, instr_list))
    make_correlation_matrix(data_list, color_list, label_list=label_list)

    return 0


if __name__ == '__main__':
    main()
