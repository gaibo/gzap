import pandas as pd
from sklearn.linear_model import LinearRegression

BUS_DAYS_IN_MONTH = 21
BUS_DAYS_IN_YEAR = 252
BUS_DAYS_IN_SIX_MONTHS = 126
ONE_DAY = pd.Timedelta(days=1)
ONE_NANOSECOND = pd.Timedelta(nanoseconds=1)


def construct_timeseries(ts_df, time_col=None, value_col=None):
    """ Construct pseudo-time-series object, i.e. a pandas Series with 'time' and 'value'
    :param ts_df: DataFrame or Series object with at least a time column and a value column
    :param time_col: DataFrame column that represents time
    :param value_col: DataFrame column that represents value
    :return: pd.Series object with 'time' as index and 'value' as name
    """
    ts = ts_df.copy()   # Avoid modifying original DataFrame
    if isinstance(ts, pd.Series):
        # Convert Series to 2-column DataFrame, since we strive to be overly robust
        ts = ts.reset_index()
    n_cols = ts.shape[1]
    if n_cols == 0:
        print("ERROR construct_timeseries: 0 columns in DataFrame.")
        return None
    elif n_cols == 1:
        # Try restoring index to column
        ts = ts.reset_index()
    elif n_cols > 2:
        # Get columns based on parameters if there are more than 2
        if time_col is None or value_col is None:
            print("ERROR construct_timeseries: more than 2 columns in DataFrame, "
                  "please provide time and value column names.")
            return None
        ts = ts[[time_col, value_col]]
    ts.columns = ['time', 'value']
    return ts.set_index('time')['value'].sort_index().dropna()


def share_dateindex(timeseries_list):
    """Align a list of time-series by their shared date-times
    :param timeseries_list: list of time-series
    :return: list of aligned/truncated time-series
    """
    column_list = range(len(timeseries_list))
    combined_df = pd.DataFrame(dict(zip(column_list, timeseries_list))).dropna()
    return [combined_df[column].rename('value') for column in column_list]


def get_best_fit(x_data, y_data, fit_intercept=True):
    """Find line of best fit for x and y data using linear regression
    :param x_data: first set of data (does not need to match y_data in date index)
    :param y_data: second set of data
    :param fit_intercept: whether to fit an intercept (False to force 0)
    :return: tuple of R^2, slope of line, and the best fit model (from which
             prior two items and more can be obtained)
    """
    [joined_x_data, joined_y_data] = share_dateindex([x_data, y_data])
    x = joined_x_data.values.reshape(-1, 1)
    y = joined_y_data.values
    model = LinearRegression(fit_intercept=fit_intercept).fit(x, y)
    r_sq = model.score(x, y)
    slope = model.coef_[0]
    return r_sq, slope, model
