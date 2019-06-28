calc_log_returns = lambda arr: np.log(arr[-1]/arr[0]) if len(arr)>1 else np.NaN

elif granularity == 'intraday':
if time_start != time_end:
    print("WARNING: only time_start parameter is used for intraday.")
intraday_day = truncd_levels.loc[time_start, time_start + ONE_DAY]
return intraday_day.resample(intraday_interval, label='right', closed='right').pad()
elif granularity == 'multiday':
# NOTE: still uncertain about how to do this
return truncd_levels.resample(multiday_interval).pad()


def log_returns(self, time_start=None, time_end=None,
                do_resample=False, resample_interval=pd.Timedelta(minutes=5)):
    """ Calculate log returns
    :param time_start: start of time-series to use
    :param time_end: end of time-series to use
    :param do_resample: set True to resample to the granularity of <resample_interval>
    :param resample_interval: only relevant if resampling
    :return: pd.Series with 'time' and 'value'
    """
    if time_start is None:
        time_start = self.levels.first_valid_index()
    if time_end is None:
        time_end = self.levels.last_valid_index()
    truncd_levels = self.levels.truncate(time_start, time_end)
    if not self.is_intraday or not do_resample:
        # Data is already regular (i.e. daily or fixed-interval intraday) -
        # avoid messing with business days, etc.
        return np.log(truncd_levels).diff()
    else:
        # Data is intraday and irregular - resample to get fixed intervals
        resampled_truncd_levels = \
            truncd_levels.resample(resample_interval,
                                   label='right', closed='right').pad()
        return np.log(resampled_truncd_levels).diff()

table = pd.concat([vix_prices.describe().iloc[0:1].astype(int),
                       vix_prices.describe().iloc[1:],
                       vix_pct_changes.describe().iloc[1:]*100])

def make_scatterplot(x_data, y_data, xlabel=None, ylabel=None, title=None, ax=None):
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Plot
    x_data.to_frame(x_data.name).join(y_data.to_frame(y_data.name))
    ax.scatter(x_data, y_data)
    ax.grid()
    ax.plot([0, 1], [0, x_data.corr(y_data)], 'k')
    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax

def make_scatterplot(x_instr, y_instr, ax=None):
    # Prepare Figure and Axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    # Plot
    x_returns = x_instr.price_return(logarithmic=False)
    y_returns = y_instr.price_return(logarithmic=False)
    joined_data = x_returns.to_frame(x_instr.name).join(y_returns.to_frame(y_instr.name), how='inner')
    ax.scatter(joined_data[x_instr.name], joined_data[y_instr.name])
    ax.grid()
    corr_x_y = x_returns.corr(y_returns)
    x_min = x_returns.min()
    x_max = x_returns.max()
    ax.plot([x_min, x_max], [x_min*corr_x_y, x_max*corr_x_y], 'k')
    # Set labels
    ax.set_xlabel("{} % Change".format(x_instr.name))
    ax.set_ylabel("{} % Change".format(y_instr.name))
    ax.set_title("Daily Percent Change: {} vs {}".format(x_instr.name, y_instr.name))
    return fig, ax

ax.set_xlabel("{} % Change".format(x_instr.name))
    ax.set_ylabel("{} % Change".format(y_instr.name))
    ax.set_title("Daily Percent Change: {} vs {}".format(x_instr.name, y_instr.name))

joined_data = x_data.to_frame('x').join(y_data.to_frame('y'), how='inner').dropna()

# Execute organized commands to create desired analyses