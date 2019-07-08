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
    make_lineplot([truncd_spx / truncd_spx[0], truncd_agg / truncd_agg[0], truncd_jgbvix / truncd_jgbvix[0],
                   truncd_lqd / truncd_lqd[0], truncd_tyvix / truncd_tyvix[0]],
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

    # Matrix Analysis
    fig, axs = plt.subplots(6, 6)
    instr_list = [spx, vix, hyg, ief, sx5e, agg]
    color_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    for x_pos, instr_x in zip(range(6), instr_list):
        for y_pos, instr_y in zip(range(6), instr_list):
            row = 5 - y_pos
            col = x_pos
            if x_pos > y_pos:
                # Bottom triangle - scatter
                make_scatterplot(instr_x.price_return(), instr_y.price_return(), ax=axs[row, col])
            elif y_pos > x_pos:
                # Top triangle - text
                [joined_x_data, joined_y_data] = share_dateindex([instr_x.price_return(),
                                                                  instr_y.price_return()])
                x = joined_x_data.values.reshape(-1, 1)
                y = joined_y_data.values
                model = LinearRegression(fit_intercept=False).fit(x, y)
                r_sq = round(model.score(x, y), 2)
                slope = round(model.coef_[0], 2)
                axs[row, col].text(0.30, 0.4, "$Slope$: {}\n$R^2$: {}".format(slope, r_sq))
            else:
                # Diagonal - distribution
                make_histogram(instr_x.price_return(), hist=True, n_bins=100, line=False,
                               ax=axs[row, col])


if __name__ == '__main__':
    main()
