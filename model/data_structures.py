import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BUS_DAYS_IN_MONTH = 21
BUS_DAYS_IN_YEAR = 252
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


class Instrument(object):
    """
    Financial instrument base class
    """
    def __init__(self, ts_df, name='', tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param name: name of the data
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        self.levels = construct_timeseries(ts_df)
        self.name = name
        self.tradestats = tradestats

    def price(self, granularity='daily', time_start=None, time_end=None,
              intraday_interval='5T', multiday_interval='M'):
        """ Output price levels with custom granularity
        :param granularity: 'daily', 'intraday', 'multiday', or None
        :param time_start: start of time-series to use
        :param time_end: end of time-series to use
        :param intraday_interval: interval to use with 'intraday'
        :param multiday_interval: interval to use with 'multiday'
        :return: pd.Series with 'time' and 'value'
        """
        # Get specified date/time range
        if time_start is None:
            time_start = self.levels.first_valid_index()
        if time_end is None:
            time_end = self.levels.last_valid_index()
        truncd_levels = self.levels.truncate(time_start, time_end)
        # Return prices based on granularity choice
        if granularity == 'daily':
            days = truncd_levels.index.normalize().unique()
            if len(days) == len(truncd_levels.index):
                # We already have daily data
                return truncd_levels
            else:
                # Create daily data
                return construct_timeseries(
                           pd.DataFrame([truncd_levels[day:day+ONE_DAY-ONE_NANOSECOND].iloc[-1]
                                         for day in days], index=days))
        elif granularity == 'intraday':
            # NOTE: currently unsure if we need to upsample (pad()) or downsample (mean()) or both
            if time_start != time_end:
                print("WARNING: only time_start parameter is used for intraday.")
            intraday_day = truncd_levels[time_start:time_start+ONE_DAY-ONE_NANOSECOND]
            day_upsample_pad = intraday_day.resample('S').pad()
            day_downsample_mean = day_upsample_pad.resample(intraday_interval,
                                                            label='right', closed='right').mean()
            return day_downsample_mean
        elif granularity == 'multiday':
            # NOTE: perhaps this should be combined with 'intraday'
            return truncd_levels.resample(multiday_interval, label='right', closed='right').mean()
        else:
            return truncd_levels

    def price_return(self, logarithmic=True,
                     granularity='daily', time_start=None, time_end=None,
                     intraday_interval='5T', multiday_interval='M'):
        """ Calculate percent or logarithmic returns with custom granularity
        :param logarithmic: set False for percent returns
        :param granularity: 'daily', 'intraday', 'multiday', or None
        :param time_start: start of time-series to use
        :param time_end: end of time-series to use
        :param intraday_interval: interval to use with 'intraday'
        :param multiday_interval: interval to use with 'multiday'
        :return: pd.Series with 'time' and 'value'
        """
        prices = self.price(granularity, time_start, time_end,
                            intraday_interval, multiday_interval)
        if logarithmic:
            return np.log(prices).diff().iloc[1:]
        else:
            return prices.pct_change().iloc[1:]

    def realized_vol(self, do_shift=False):
        """ Calculate annualized realized vol from past month
        :param do_shift: set True to shift data back one month, to compare to implied vol
        :return: pd.Series with 'time' and 'value', with ~20 NaNs at beginning
        """
        result = np.sqrt(self.price_return().rolling(BUS_DAYS_IN_MONTH).var(ddof=0)
                         * BUS_DAYS_IN_YEAR)
        if do_shift:
            return result.shift(-BUS_DAYS_IN_MONTH)
        else:
            return result


class CashInstr(Instrument):
    """
    Cash instrument, derived from financial instrument
    """
    def __init__(self, ts_df, name='', tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param name: name of the data
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, name, tradestats)


class Derivative(Instrument):
    """
    Derivative instrument, derived from financial instrument
    """
    def __init__(self, ts_df, underlying, name='', tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: underlying Instrument
        :param name: name of the data
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, name, tradestats)
        self.underlying = underlying

    def undl_realized_vol(self, do_shift=False):
        """ Calculate annualized realized vol from past month for underlying asset
        :param do_shift: set True to shift data back one month, to compare to implied vol
        :return: pd.Series with 'time' and 'value', with ~20 NaNs at beginning
        """
        return self.underlying.realized_vol(do_shift)


class Index(CashInstr):
    def __init__(self, ts_df, name='', tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param name: name of the data
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, name, tradestats)


class Stock(CashInstr):
    def __init__(self, ts_df, name='', tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param name: name of the data
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, name, tradestats)


class ETF(CashInstr):
    def __init__(self, ts_df, name='', tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param name: name of the data
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, name, tradestats)


class Futures(Derivative):
    def __init__(self, ts_df, underlying, maturity_date, name='', tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: underlying Instrument
        :param maturity_date: maturity date of futures contract
        :param name: name of the data
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, underlying, name, tradestats)
        self.maturity_date = maturity_date


class Options(Derivative):
    def __init__(self, ts_df, underlying, expiry_date, pc, strike, name='', tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: underlying Instrument
        :param expiry_date: expiration date of options contract
        :param pc: whether options contract is put or call
        :param strike: strike price of options contract
        :param name: name of the data
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, underlying, name, tradestats)
        self.expiry_date = expiry_date
        self.pc = pc
        self.strike = strike
        # TODO: may need more efficient way to store options in bulk


class VolatilityIndex(Index):
    def __init__(self, ts_df, underlying, name='', tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: the underlying instrument whose volatility is being gauged
        :param name: name of the data
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, name, tradestats)
        self.underlying = underlying

    def undl_realized_vol(self, do_shift=False):
        """ Calculate annualized realized vol from past month for underlying asset
        :param do_shift: set True to shift data back one month, to compare to implied vol
        :return: pd.Series with 'time' and 'value', with ~20 NaNs at beginning
        """
        return self.underlying.realized_vol(do_shift)


def main():
    # Load example data
    sptr_vix_df = pd.read_csv('../data/sptr_vix.csv')
    sptr_vix_df['Date'] = pd.to_datetime(sptr_vix_df['Date'])
    sptr_vix_df = sptr_vix_df.set_index('Date')

    # Create objects
    sptr = Index(sptr_vix_df['SPTR'])
    vix = VolatilityIndex(sptr_vix_df['VIX'], sptr)

    # Look at implied volatility vs realized volatility
    start = pd.Timestamp('2015-01-01')
    end = pd.Timestamp('2018-01-01')
    truncd_vix = vix.price().truncate(start, end)
    truncd_realized_vol = vix.undl_realized_vol().truncate(start, end)
    truncd_shifted_realized_vol = vix.undl_realized_vol(do_shift=True).truncate(start, end)

    fig, ax = plt.subplots()
    ax.plot(truncd_vix, label='VIX', linewidth=3)
    ax.plot(100 * truncd_realized_vol, label='SPTR Realized Vol')
    ax.plot(100 * truncd_shifted_realized_vol, label='SPTR Realized Vol from 21 Days Ahead', linewidth=3)
    ax.legend()

    return 0


if __name__ == '__main__':
    main()
