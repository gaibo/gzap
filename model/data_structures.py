import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from utility.universal_tools import construct_timeseries, share_dateindex, \
    BUS_DAYS_IN_MONTH, BUS_DAYS_IN_YEAR, BUS_DAYS_IN_SIX_MONTHS, ONE_DAY, ONE_NANOSECOND
from utility.gaibo_modules.cboe_exchange_holidays_v3 import datelike_to_timestamp

register_matplotlib_converters()


class Instrument(object):
    """
    Financial instrument base class
    """
    def __init__(self, ts_df, name='', tradestats=None):
        """
        :param ts_df: time-series DataFrame/Series with time and value
        :param name: name of the data
        :param tradestats: (optional) relevant trade statistics, etc. to be included; default empty DataFrame
        """
        self.levels = construct_timeseries(ts_df)   # Drops NaNs and sorts
        self.loc = self.levels.loc  # Allow for native use of DataFrame/Series .loc[] indexing
        self.name = name
        if isinstance(tradestats, pd.DataFrame) or isinstance(tradestats, pd.Series):
            self.tradestats = tradestats.copy()
        else:
            self.tradestats = tradestats    # Including default of None

    def __getitem__(self, index):
        """ Allow for native use of DataFrame/Series indexing
            e.g. vix_instrument[4:] works instead of vix_instrument.price()[4:]
        """
        return self.levels[index]

    def price(self, granularity='daily', time_start=None, time_end=None,
              intraday_interval='5T', multiday_interval='M'):
        """ Output price levels with custom granularity
        :param granularity: 'daily', 'intraday', 'multiday', or None
        :param time_start: start of time-series to use
        :param time_end: end of time-series to use
        :param intraday_interval: interval to use with 'intraday'; '5T' is 5 minutes, etc.
        :param multiday_interval: interval to use with 'multiday'; 'M' is 1 month, etc.
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
                # Create daily data - last price on each day - from more granular data
                return construct_timeseries(pd.Series([truncd_levels[day:day+ONE_DAY-ONE_NANOSECOND].iloc[-1]
                                                       for day in days], index=days))
        elif granularity == 'intraday':
            # Warning: day-by-day processing for intraday granularity change is pretty inefficient
            # but needed to prevent padding from going beyond end of trade hours
            days = truncd_levels.index.normalize().unique()
            def change_day_granularity_intraday(day):
                # NOTE: currently performing upsample (pad()) then downsample (mean()); arbitrary methodology
                intraday_day = truncd_levels[day:day+ONE_DAY-ONE_NANOSECOND]
                day_upsample_pad = intraday_day.resample('L').pad()     # Propogate to finest granularity (milliseconds)
                day_downsample_mean = \
                    day_upsample_pad.resample(intraday_interval, label='right', closed='right').mean()
                return day_downsample_mean
            return pd.concat([change_day_granularity_intraday(day) for day in days])
        elif granularity == 'multiday':
            # NOTE: very similar to 'intraday', but exclusively to downsample
            return truncd_levels.resample(multiday_interval, label='right', closed='right').mean()
        else:
            raise ValueError(f"expected 'daily', 'intraday', or 'multiday' for granularity; not '{granularity}'")

    def price_return(self, logarithmic=True, **price_kwargs):
        """ Calculate logarithmic or percent returns (with custom granularity)
        :param logarithmic: set True for log returns, False for percent returns
        :param price_kwargs: kwargs for passing to price() method
        :return: pd.Series with 'time' and 'value'
        """
        prices = self.price(**price_kwargs)
        if logarithmic:
            return np.log(prices).diff().iloc[1:]
        else:
            return prices.pct_change().iloc[1:]

    def realized_vol(self, shift_backwards=False, window=BUS_DAYS_IN_MONTH, bps=False, price_in_bps=False):
        """ Calculate (annualized) realized vol with rolling window
        :param shift_backwards: set True to shift data back to beginning of window, to compare to implied vol
        :param window: rolling window, also used as days to shift; default a business month
        :param bps: set True to calculate basis point return vol instead of percent return vol
                    (e.g. if calculating on time-series of annual percent yields, as opposed to prices)
                    NOTE: by default, this mode assumes price time-series holds percent yields
        :param price_in_bps: set True when bps=True and price time-series is in basis point (spreads)
                             instead of percent (yields) (e.g. if calculating on time-series of
                             spreads over Treasury rates, as opposed to the rates themselves)
        :return: pd.Series with 'time' and 'value', with window of NaNs at beginning (~20 if monthly)
        """
        if not bps:
            # Price return vol
            result = np.sqrt(self.price_return().rolling(window).var(ddof=0)
                             * BUS_DAYS_IN_YEAR)
        else:
            # Basis point return vol
            if price_in_bps:
                to_bps_multiplier = 1
            else:
                to_bps_multiplier = 100
            result = (self.price().rolling(window).apply(
                          lambda yields: (np.mean(np.diff(yields)**2) * BUS_DAYS_IN_YEAR)**0.5,
                          raw=True)) * to_bps_multiplier
        if shift_backwards:
            return result.shift(-window)
        else:
            return result


class CashInstr(Instrument):
    """
    Cash instrument, derived from financial instrument
    """
    def __init__(self, ts_df, **super_kwargs):
        """
        :param ts_df: time-series DataFrame with time and value
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(ts_df, **super_kwargs)


class Derivative(Instrument):
    """
    Derivative instrument, derived from financial instrument
    """
    def __init__(self, ts_df, underlying, **super_kwargs):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: underlying Instrument
        :param super_kwargs: kwargs for passing to superclass init()
        """
        super().__init__(ts_df, **super_kwargs)
        self.underlying = underlying

    def undl_realized_vol(self, **realized_vol_kwargs):
        """ Calculate realized vol for underlying asset (by calling underlying's realized_vol)
        :param realized_vol_kwargs: kwargs for passing to realized_vol() method
        :return: pd.Series with 'time' and 'value', with ~20 NaNs at beginning
        """
        return self.underlying.realized_vol(**realized_vol_kwargs)


class Index(CashInstr):
    def __init__(self, ts_df, **super_kwargs):
        """
        :param ts_df: time-series DataFrame with time and value
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(ts_df, **super_kwargs)


class Stock(CashInstr):
    def __init__(self, ts_df, **super_kwargs):
        """
        :param ts_df: time-series DataFrame with time and value
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(ts_df, **super_kwargs)


class ETF(CashInstr):
    def __init__(self, ts_df, **super_kwargs):
        """
        :param ts_df: time-series DataFrame with time and value
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(ts_df, **super_kwargs)


class Futures(Derivative):
    def __init__(self, ts_df, underlying, maturity_date, **super_kwargs):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: underlying Instrument
        :param maturity_date: maturity date of futures contract
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(ts_df, underlying, **super_kwargs)
        self.maturity_date = datelike_to_timestamp(maturity_date)


class Options(Derivative):
    def __init__(self, ts_df, underlying, expiry_date, pc, strike, **super_kwargs):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: underlying Instrument
        :param expiry_date: expiration date of options contract
        :param pc: whether options contract is put or call
        :param strike: strike price of options contract
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(ts_df, underlying, **super_kwargs)
        self.expiry_date = expiry_date
        self.pc = pc
        self.strike = strike
        # TODO: need more efficient way to store options in bulk... after 1 year, solution: DataFrame with named cols


class VolatilityIndex(Index):
    def __init__(self, ts_df, underlying, **super_kwargs):
        """
        :param ts_df: time-series DataFrame with time and value
        :param underlying: the underlying instrument whose volatility is being gauged
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(ts_df, **super_kwargs)
        self.underlying = underlying

    def undl_realized_vol(self, **kwargs):
        """ Calculate realized vol for underlying asset (by calling underlying's realized_vol)
        :return: pd.Series with 'time' and 'value', with ~20 NaNs at beginning
        """
        return self.underlying.realized_vol(**kwargs)

    def vol_regime(self, low_threshold=0.1, high_threshold=0.9, window=BUS_DAYS_IN_SIX_MONTHS,
                   time_start=None, time_end=None):
        """ Produce a label of 'high' or 'low' volatility regime for each day
        :param low_threshold: low quantile - a break would switch high regime to low
        :param high_threshold: high quantile - a break would switch low regime to high
        :param window: number of data points in rolling window
        :param time_start: start of time-series to use
        :param time_end: end of time-series to use
        :return: pd.Series with 'time' and 'value', with 'high' and 'low' labels as value
                 list of tuples of start and end dates of low regimes
                 list of tuples of start and end dates of high regimes
        """
        # Get prices, find corresponding rolling high and low thresholds
        prices = self.price('daily', time_start, time_end)  # Currently only daily vol regimes
        rolling_low = prices.rolling(window).quantile(low_threshold).dropna()
        rolling_high = prices.rolling(window).quantile(high_threshold).dropna()
        [joined_prices, joined_rolling_low, joined_rolling_high] = \
            share_dateindex([prices, rolling_low, rolling_high])
        # Find where prices break the high and low thresholds (to change regimes)
        high_breakthrough = joined_prices[joined_prices > joined_rolling_high].index
        low_breakthrough = joined_prices[joined_prices < joined_rolling_low].index
        # Track chronologically to identify regime changes
        regime = joined_prices.copy()   # Copy joined_prices as template with dates
        curr_state = 'low'
        low_start = regime.index[0]
        high_start = None   # Not needed to begin with since we start on "low"
        final_day = regime.index[-1]
        low_intervals_list = []
        high_intervals_list = []
        for day in regime.index:
            if curr_state == 'low' and day in high_breakthrough:
                # Regime change: low to high
                low_intervals_list.append((low_start, day))     # Record completed low interval
                high_start = day    # Set start of high interval
                curr_state = 'high'     # Change state low to high
            elif curr_state == 'high' and day in low_breakthrough:
                # Regime change: high to low
                high_intervals_list.append((high_start, day))  # Record completed high interval
                low_start = day     # Set start of low interval
                curr_state = 'low'      # Change state high to low
            regime[day] = curr_state    # Record state regardless of state change
        if curr_state == 'low':
            low_intervals_list.append((low_start, final_day))     # Complete last interval
        else:
            high_intervals_list.append((high_start, final_day))
        return regime, low_intervals_list, high_intervals_list


def main():
    # Load example data
    vix_df = pd.read_csv('../data/vix_ohlc.csv', index_col='Date', parse_dates=True)
    bbg_data = pd.read_csv('../data/bbg_automated_pull.csv',
                           index_col=0, parse_dates=True, header=[0, 1])

    # Create objects
    spx = Index(bbg_data[('SPX Index', 'PX_LAST')], name='SPX')
    vix = VolatilityIndex(vix_df['VIX Close'], spx, name='VIX')

    # Look at implied volatility vs realized volatility
    start = pd.Timestamp('2015-01-01')
    end = pd.Timestamp('2018-01-01')
    truncd_vix = vix.price().truncate(start, end)
    truncd_realized_vol = vix.undl_realized_vol().truncate(start, end) * 100
    truncd_shifted_realized_vol = vix.undl_realized_vol(do_shift=True).truncate(start, end) * 100

    fig, ax = plt.subplots()
    ax.plot(truncd_vix, label='VIX', linewidth=3)
    ax.plot(truncd_realized_vol, label='SPX Realized Vol')
    ax.plot(truncd_shifted_realized_vol, label='SPX Realized Vol from 21 Days Ahead', linewidth=3)
    ax.legend()

    return 0


if __name__ == '__main__':
    main()
