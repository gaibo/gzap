import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BUS_DAYS_IN_MONTH = 21
BUS_DAYS_IN_YEAR = 252
ONE_DAY = pd.Timedelta(days=1)
ONE_NANOSECOND = pd.Timedelta(nanoseconds=1)


def construct_timeseries(ts_df):
    """ Construct pseudo-time-series object, i.e. a pandas Series with 'time' and 'value'
    :param ts_df: DataFrame object with 1 index (time) and 1 column (value)
    :return: pd.Series object with 'time' as index and 'value' as name
    """
    ts_df = ts_df.reset_index()
    assert ts_df.shape[1] == 2   # 2 columns - time and value
    ts = pd.Series(ts_df.iloc[:, 1].values, index=ts_df.iloc[:, 0], name='value')
    ts.index.name = 'time'
    ts = ts.sort_index()
    return ts.truncate(ts.first_valid_index(), ts.last_valid_index())


class Instrument(object):
    """
    Financial instrument base class
    """
    def __init__(self, ts_df, is_intraday=False, tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param is_intraday: set True if data is intraday to prepare additional handling
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        self.levels = construct_timeseries(ts_df)
        self.is_intraday = is_intraday
        self.tradestats = tradestats

    def prices(self, time_start=None, time_end=None, granularity='daily',
               intraday_interval='5T', multiday_interval='M'):
        """ Output price levels with custom granularity
        :param time_start: start of time-series to use
        :param time_end: end of time-series to use
        :param granularity: 'daily', 'intraday', 'multiday', or None
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
            return construct_timeseries(
                       pd.DataFrame([truncd_levels[day:day+ONE_DAY-ONE_NANOSECOND].iloc[-1]
                                     for day in days], index=days))
        elif granularity == 'intraday':
            if time_start != time_end:
                print("WARNING: only time_start parameter is used for intraday.")
            intraday_day = truncd_levels.loc[time_start, time_start+ONE_DAY]
            return intraday_day.resample(intraday_interval, label='right', closed='right').pad()
        elif granularity == 'multiday':
            # NOTE: perhaps this should be combined with 'intraday'
            return truncd_levels.resample(multiday_interval).pad()
        else:
            return truncd_levels

    def log_returns(self, time_start=None, time_end=None, granularity='daily',
                    intraday_interval='5T', multiday_interval='M'):
        """ Calculate logarithmic returns with custom granularity
        :param time_start: start of time-series to use
        :param time_end: end of time-series to use
        :param granularity: 'daily', 'intraday', 'multiday', or None
        :param intraday_interval: interval to use with 'intraday'
        :param multiday_interval: interval to use with 'multiday'
        :return: pd.Series with 'time' and 'value'
        """
        prices = self.prices(time_start, time_end, granularity,
                             intraday_interval, multiday_interval)
        return np.log(prices).diff()


class CashInstr(Instrument):
    """
    Cash instrument, derived from financial instrument
    """
    def __init__(self, ts_df, is_intraday=False, tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param is_intraday: set True if data is intraday to prepare additional handling
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, is_intraday, tradestats)


class Derivative(Instrument):
    """
    Derivative instrument, derived from financial instrument
    """
    def __init__(self, ts_df, underlying, is_intraday=False, tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: underlying Instrument
        :param is_intraday: set True if data is intraday to prepare additional handling
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, is_intraday, tradestats)
        self.underlying = underlying


class Index(CashInstr):
    def __init__(self, ts_df, is_intraday=False):
        """
        :param ts_df: time-series DataFrame with time and value
        :param is_intraday: set True if data is intraday to prepare additional handling
        """
        super().__init__(ts_df, is_intraday)


class Stock(CashInstr):
    def __init__(self, ts_df, is_intraday=False, tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param is_intraday: set True if data is intraday to prepare additional handling
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, is_intraday, tradestats)


class ETF(CashInstr):
    def __init__(self, ts_df, is_intraday=False, tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param is_intraday: set True if data is intraday to prepare additional handling
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, is_intraday, tradestats)


class Futures(Derivative):
    def __init__(self, ts_df, underlying, maturity, is_intraday=False, tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: underlying Instrument
        :param maturity: maturity date of futures
        :param is_intraday: set True if data is intraday to prepare additional handling
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, underlying, is_intraday, tradestats)
        self.maturity = maturity


class Options(Derivative):
    def __init__(self, ts_df, underlying, expiry, pc, strike, is_intraday=False, tradestats=pd.DataFrame(None)):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: underlying Instrument
        :param expiry: expiration date of options contract
        :param pc: whether options contract is put or call
        :param strike: strike price of options contract
        :param is_intraday: set True if data is intraday to prepare additional handling
        :param tradestats: (optional) trade statistics to be included, as a DataFrame
        """
        super().__init__(ts_df, underlying, is_intraday, tradestats)
        self.expiry = expiry
        self.pc = pc
        self.strike = strike


class VolatilityIndex(Index):
    def __init__(self, ts_df, underlying, is_intraday=False):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: the underlying instrument whose volatility is being gauged
        :param is_intraday: set True if data is intraday to prepare additional handling
        """
        super().__init__(ts_df, is_intraday)
        self.underlying = underlying

    def realized_vol(self, do_shift=False):
        """ Calculate annualized realized vol from past month
        :return: pd.Series with 'time' and 'value', with ~20 NaNs at beginning
        """
        result = np.sqrt(self.underlying.log_returns()
                         .rolling(BUS_DAYS_IN_MONTH).var(ddof=0)
                         * BUS_DAYS_IN_YEAR)
        if do_shift:
            return result.shift(-BUS_DAYS_IN_MONTH)
        else:
            return result


def main():
    # Load example data
    sptr_vix_df = pd.read_csv('sptr_vix.csv')
    sptr_vix_df['Date'] = pd.to_datetime(sptr_vix_df['Date'])
    sptr_vix_df = sptr_vix_df.set_index('Date')

    # Create objects
    sptr = Index(sptr_vix_df['SPTR'])
    vix = VolatilityIndex(sptr_vix_df['VIX'], sptr)

    # Look at implied volatility vs realized volatility
    start = pd.Timestamp('2015-01-01')
    end = pd.Timestamp('2018-01-01')
    truncd_vix = vix.levels.truncate(start, end)
    truncd_realized_vol = vix.realized_vol().truncate(start, end)
    truncd_shifted_realized_vol = vix.realized_vol(do_shift=True).truncate(start, end)

    fig, ax = plt.subplots()
    ax.plot(truncd_vix, label='VIX', linewidth=3)
    ax.plot(100 * truncd_realized_vol, label='SPTR Realized Vol')
    ax.plot(100 * truncd_shifted_realized_vol, label='SPTR Realized Vol from 21 Days Ahead', linewidth=3)
    ax.legend()

    return 0


if __name__ == '__main__':
    main()
