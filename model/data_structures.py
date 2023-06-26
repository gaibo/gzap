import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

from universal_tools import construct_timeseries, share_dateindex, \
    BUS_DAYS_IN_MONTH, BUS_DAYS_IN_YEAR, BUS_DAYS_IN_SIX_MONTHS, ONE_DAY, ONE_NANOSECOND
from utility.gaibo_modules.cboe_exchange_holidays_v3 import datelike_to_timestamp

pd.plotting.register_matplotlib_converters()


class ManagedPandas:
    """ Descriptor class to manage pandas DataFrame/Series type attributes. Features:
          - setter prints warning if input is not pandas DataFrame/Series
          - setter auto-updates associated object's .loc[] functionality to match that of recently set DF/Series
            NOTE: I'm doing this here rather than in object's methods because pandas .loc is an object with
                  __getitem__ indexing rather than a method, so I can't just forward the arguments
        The thinking here is we want GZAP objects to be "resilient" even after re-setting data.
    """
    def __set_name__(self, owner, name):
        self._name = name   # Make small effort to distinguish managed attribute's text name

    def __get__(self, instance, owner=None):
        return instance.__dict__[self._name]    # Note we store and retrieve from associated object's __dict__

    def __set__(self, instance, value):
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            instance.__dict__[self._name] = value.copy()    # Deep copy to avoid DataFrame state confusion
        else:
            print(f"WARNING: {type(instance).__name__} input data is not DataFrame/Series!\n"
                  f"Things may not work until you re-set the '{self._name}' attribute:\n"
                  f"{value}")
            instance.__dict__[self._name] = value   # None passes here; leaving in flexibility to change input later
        # Allow native use of DataFrame/Series .loc[] indexing if available
        try:
            instance.__dict__['loc'] = instance.__dict__[self._name].loc    # Passes in most cases
            instance.__dict__['iloc'] = instance.__dict__[self._name].iloc
        except AttributeError:
            if hasattr(instance, 'loc'):
                delattr(instance, 'loc')    # Preserve AttributeError if new input doesn't have .loc
                delattr(instance, 'iloc')


class Instrument:
    """
    Financial instrument base class
    Defined by: 1) a pandas DataFrame/Series including all relevant stats (e.g. dates and times, prices)
                2) a name
    """
    raw_data_df = ManagedPandas()   # Descriptor-based managed attribute

    def __init__(self, raw_data_df, name=None):
        """
        :param raw_data_df: DataFrame/Series containing reasonably formatted financial data
        :param name: name of the financial instrument
        """
        self.raw_data_df = raw_data_df
        self.name = name

    def __getitem__(self, index):
        """ Allow for native use of DataFrame/Series indexing
            e.g. vix_instrument[4:] works instead of vix_instrument.price()[4:]
        """
        return self.raw_data_df[index]

    def __str__(self):
        """ Print helpful summary of the Instrument object - small peek at raw data wrapped by class """
        if hasattr(self.raw_data_df, 'shape') and self.raw_data_df.shape[0] > 10:
            data_peek = (
                f"{self.raw_data_df.head()}\n"
                f"...\n"
                f"{self.raw_data_df.tail()}"
            )
        else:
            data_peek = f"{self.raw_data_df}"
        return (f"'{self.name}' raw data:\n"
                f"{data_peek}")

    def __repr__(self):
        """ Return a string that can reconstruct this object """
        return (
            f"{type(self).__name__}("
            f"raw_data_df=<{self.name}_df>, "
            f"name='{self.name}')"
        )


class CashInstr(Instrument):
    """
    Cash instrument, derived from financial instrument; think of as archetype of index, stock, ETF, etc.
    Defined by: 1) extracting a price time-series from Instrument through column names
    """
    levels = ManagedPandas()    # Descriptor-based managed attribute; priority - .loc will link to levels

    def __init__(self, raw_data_df, time_col=None, price_col=None, index_is_time=False, **super_kwargs):
        """
        :param raw_data_df: DataFrame/Series containing reasonably formatted financial data
        :param time_col: name of DataFrame column that represents time (in a time-series)
        :param price_col: name of DataFrame column that represents price (value in a time-series)
        :param index_is_time: set True if index of raw_data_df is time
        :param super_kwargs: kwargs for passing to superclass init()
        """
        super().__init__(raw_data_df, **super_kwargs)
        self.time_col = time_col    # I'm still on the fence about whether to keep these in state
        self.price_col = price_col
        self.index_is_time = index_is_time
        self.levels = construct_timeseries(self.raw_data_df, time_col, price_col, index_is_time)  # Drop NaNs and sort

    def refresh_levels(self, persist_cols=True, time_col=None, price_col=None, index_is_time=False):
        """ Manually refresh self.levels in edge case of self.raw_data_df being modified.
            Yes, at the moment this is duplicate code from init. But want extensibility and ways to "correct" state.
        """
        print(f"Refreshing...\n\n"
              f"'raw_data_df' (head):\n{self.raw_data_df.head()}\n")
        if persist_cols:
            print(f"PRESERVING column info:\n"
                  f"'time_col': {self.time_col}\n"
                  f"'price_col': {self.price_col}\n"
                  f"'index_is_time': {self.index_is_time}\n")
        else:
            print(f"CHANGING column info:\n"
                  f"'time_col': {self.time_col} -> {time_col}\n"
                  f"'price_col': {self.price_col} -> {price_col}\n"
                  f"'index_is_time': {self.index_is_time} -> {index_is_time}\n")
            self.time_col = time_col
            self.price_col = price_col
            self.index_is_time = index_is_time
        self.levels = construct_timeseries(self.raw_data_df, self.time_col,
                                           self.price_col, self.index_is_time)
        print(f" ... Done! Levels refreshed (head):\n"
              f"{self.levels.head()}")

    def __getitem__(self, index):
        """ Allow for native use of DataFrame/Series indexing
            e.g. vix_instrument[4:] works instead of vix_instrument.price()[4:]
            NOTE: more refined than base Instrument's __getitem__() - uses levels, not raw_data_df
        """
        return self.levels[index]

    def price(self, granularity='daily', time_start=None, time_end=None,
              intraday_interval='5T', multiday_interval='M'):
        """ Output price levels with custom granularity
        :param granularity: 'daily', 'intraday', or 'multiday'
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
        """ Calculate (annualized) realized vol with rolling window;
            formula is like rolling population standard deviation but asserting 0 as population mean
        :param shift_backwards: set True to shift data back to beginning of window, to compare to implied vol
        :param window: rolling window, also used as days to shift; default one business month
        :param bps: set True to calculate basis point return vol instead of percent return vol
                    (e.g. if calculating on time-series of annual percent yields, as opposed to prices)
                    NOTE: by default, this mode assumes price time-series holds percent yields
        :param price_in_bps: set True when bps=True and price time-series is in basis point (spreads)
                             instead of percent (yields) (e.g. if calculating on time-series of
                             spreads over Treasury rates, as opposed to the rates themselves)
        :return: pd.Series with 'time' and 'value', with window (~20 if monthly) of NaNs at beginning (end if shifting)
        """
        if not bps:
            # Price return vol - done on log returns (% of change) of prices
            result = self.price_return().rolling(window).apply(
                         lambda returns: (np.mean(returns**2) * BUS_DAYS_IN_YEAR)**0.5, raw=True) * 100
        else:
            # Basis point return vol - done on differences (amount of change) of rates (which happen to be percents)
            if not price_in_bps:
                to_bps_multiplier = 100     # Scale up to basis points - looks silly small otherwise
            else:
                to_bps_multiplier = 1
            # Note the following .iloc to ensure np.mean() always receives constant # of non-NaN elements
            result = self.price().diff().iloc[1:].rolling(window).apply(
                         lambda changes: (np.mean(changes**2) * BUS_DAYS_IN_YEAR)**0.5, raw=True) * to_bps_multiplier
        if shift_backwards:
            return result.shift(-window)
        else:
            return result


class Derivative(Instrument):
    """
    Derivative instrument, derived from financial instrument
    Defined by: 1) an underlying instrument
    """
    data_df = ManagedPandas()   # Descriptor-based managed attribute

    def __init__(self, raw_data_df, underlying, **super_kwargs):
        """
        :param raw_data_df: DataFrame/Series containing reasonably formatted financial data
        :param underlying: underlying Instrument
        :param super_kwargs: kwargs for passing to superclass (Instrument) init()
        """
        super().__init__(raw_data_df, **super_kwargs)   # Note we don't get Levels functionality, but we do get .loc
        self.underlying = underlying
        self.data_df = None     # Declare for subclasses; "clean" version of raw_data_df
        self.data_cols = None   # Declare for subclasses; mapping of standardized financial attributes to data_df's cols

    def __getitem__(self, index):
        """ Allow for native use of DataFrame/Series indexing
            e.g. vix_instrument[4:] works instead of vix_instrument.price()[4:]
            NOTE: more refined than base Instrument's __getitem__() - uses levels, not raw_data_df
        """
        return self.data_df[index]

    def undl_realized_vol(self, **realized_vol_kwargs):
        """ Calculate realized vol for underlying asset (by calling underlying's realized_vol)
        :param realized_vol_kwargs: kwargs for passing to realized_vol() method
        :return: pd.Series with 'time' and 'value', with ~20 NaNs at beginning
        """
        return self.underlying.realized_vol(**realized_vol_kwargs)

    def merge(self, objs):
        """ Merge other class instance(s) into this one
        :param objs: single class object or iterable of class objects
        :return: merged data_df
        """
        def _single_merge(other):
            # Rename the object's relevant columns to match those of self
            obj_to_self_rename_map = {other.data_cols[k]: self.data_cols[k]
                                      for k in (self.data_cols.keys() & other.data_cols.keys())}
            obj_renamed_data_df = other.data_df.rename(obj_to_self_rename_map, axis=1)
            # Merge object's relevant columns into self's, but don't overwrite anything non-NaN
            self.data_df = self.data_df.combine_first(obj_renamed_data_df).sort_index()
            # Replace self's "None cols" if merged object had them filled
            # NOTE: "None cols" is the concept of an expected data_cols key that doesn't have a mapping yet
            self.data_cols = {**other.data_cols, **self.data_cols}  # Covering other with self
        if isinstance(objs, Iterable):
            # TODO: this type of logic could be a decorator
            for obj in objs:
                _single_merge(obj)
        else:
            _single_merge(objs)
        return self.data_df


class Index(CashInstr):
    def __init__(self, raw_data_df, **super_kwargs):
        """ NOTE: Index, Stock, and ETF don't have distinguishing features yet!
        :param raw_data_df: DataFrame/Series containing reasonably formatted financial data
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(raw_data_df, **super_kwargs)


class Stock(CashInstr):
    def __init__(self, raw_data_df, **super_kwargs):
        """ NOTE: Index, Stock, and ETF don't have distinguishing features yet!
        :param raw_data_df: DataFrame/Series containing reasonably formatted financial data
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(raw_data_df, **super_kwargs)


class ETF(CashInstr):
    def __init__(self, raw_data_df, **super_kwargs):
        """ NOTE: Index, Stock, and ETF don't have distinguishing features yet!
        :param raw_data_df: DataFrame/Series containing reasonably formatted financial data
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(raw_data_df, **super_kwargs)


FUTURES_COLS = {'maturity_date', 'price',
                'trade_date', 'volume', 'open_interest',
                't_to_mat', 'rate'}


class Futures(Derivative):
    def __init__(self, raw_data_df, underlying, maturity_date_col=None, price_col=None,
                 trade_date_col=None, single_trade_date=None,
                 volume_col=None, open_interest_col=None,
                 t_to_mat_col=None, rate_col=None, **super_kwargs):
        """ Explanation on organization:
            Futures data is always given as prices associated with a maturity (either date or month/year).
            Therefore, the underlying "data model" for the Futures class will be a DataFrame,
            and the class will simply track the names of the columns containing these labels.
            Yes, some columns such as maturity will contain the same value hundreds of times.
        :param raw_data_df: DataFrame/Series containing reasonably formatted financial data
        :param underlying: underlying Instrument
        :param maturity_date_col: name of DataFrame column that represents futures maturity date
        :param price_col: name of DataFrame column that represents price
        :param trade_date_col: name of DataFrame column that represents time (in a time-series);
                               set None and use single_trade_date if no such column, i.e. data is only for single day
        :param single_trade_date: datelike representing trade date of data; only used if trade_date_col is None
        :param volume_col: name of DataFrame column that represents trading volume
        :param open_interest_col: name of DataFrame column that represents trading open interest
        :param t_to_mat_col: name of DataFrame column that represents time to maturity (if it already exists)
        :param rate_col: name of DataFrame column that represents risk-free rate (if it already exists)
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(raw_data_df, underlying, **super_kwargs)
        self.data_df = self.raw_data_df.copy()  # Make a point to never modify raw original
        # Create dictionary mapping data_df column names to standard set
        self.data_cols = {
            'maturity_date': maturity_date_col,
            'price': price_col,
            'volume': volume_col,
            'open_interest': open_interest_col,
            't_to_mat': t_to_mat_col,
            'rate': rate_col
        }
        if trade_date_col is None and single_trade_date is not None:
            if 'trade_date' in self.raw_data_df.columns:
                print("WARNING: 'trade_date' column exists in raw_data_df but has not been set as trade_date_col;\n"
                      "         object init() will overwrite this column, so please ensure that's intentional.")
            self.data_df['trade_date'] = datelike_to_timestamp(single_trade_date)
            self.data_cols['trade_date'] = 'trade_date'
        else:
            self.data_cols['trade_date'] = trade_date_col   # Includes dangerous case of None
        self.data_cols = {k: v for k, v in self.data_cols.items() if v is not None}     # Filter out None
        # Drop NaNs, index, and sort raw_data_df
        self.data_df = (self.data_df.dropna(how='all')
                        .set_index([self.data_cols['trade_date'], self.data_cols['maturity_date']])
                        .sort_index())


OPTIONS_COLS = {'expiry_date', 'call_put', 'strike', 'price',
                'trade_date', 'volume', 'open_interest',
                't_to_exp', 'rate', 'forward', 'implied_vol', 'iv_price',
                'delta', 'vega'}


class Options(Derivative):
    def __init__(self, raw_data_df, underlying, expiry_date_col=None, call_put_col=None, strike_col=None,
                 price_col=None, trade_date_col=None, single_trade_date=None,
                 volume_col=None, open_interest_col=None,
                 t_to_exp_col=None, rate_col=None, forward_col=None,
                 implied_vol_col=None, iv_price_col=None, delta_col=None, vega_col=None, **super_kwargs):
        """ Explanation on organization:
            Options data is always given as prices associated with a tuple of characteristics
            (expiry, put or call, strike price). Therefore, the underlying "data model"
            for the Options class will be a DataFrame, and the class will simply track
            the names of the columns containing these labels.
            Yes, some columns such as expiry will contain the same value hundreds of times.
        :param raw_data_df: DataFrame/Series containing reasonably formatted financial data
        :param underlying: underlying Instrument
        :param expiry_date_col: name of DataFrame column that represents options expiry date
        :param call_put_col: name of DataFrame column that represents options call/put identification
        :param strike_col: name of DataFrame column that represents options strike (exercise) price
        :param price_col: name of DataFrame column that represents price
        :param trade_date_col: name of DataFrame column that represents time (in a time-series);
                               set None and use single_trade_date if no such column, i.e. data is only for single day
        :param single_trade_date: datelike representing trade date of data; only used if trade_date_col is None
        :param volume_col: name of DataFrame column that represents trading volume
        :param open_interest_col: name of DataFrame column that represents trading open interest
        :param t_to_exp_col: name of DataFrame column that represents time to expiry (if it already exists)
        :param rate_col: name of DataFrame column that represents risk-free rate (if it already exists)
        :param forward_col: name of DataFrame column that represents options-implied forward price
                            (if it already exists)
        :param implied_vol_col: name of DataFrame column that represents Black-Scholes/Black-76 implied volatility
                                (if it already exists)
        :param iv_price_col: name of DataFrame column that represents Black-Scholes/Black-76 implied volatility-implied
                             premium (if it already exists)
        :param delta_col: name of DataFrame column that represents Black-Scholes/Black-76 delta (if it already exists)
        :param vega_col: name of DataFrame column that represents Black-Scholes/Black-76 vega (if it already exists)
        :param super_kwargs: kwargs for passing to superclass
        """
        super().__init__(raw_data_df, underlying, **super_kwargs)
        self.data_df = self.raw_data_df.copy()  # Make a point to never modify raw original
        # Create dictionary mapping data_df column names to standard set
        self.data_cols = {
            'expiry_date': expiry_date_col,
            'call_put': call_put_col,
            'strike': strike_col,
            'price': price_col,
            'volume': volume_col,
            'open_interest': open_interest_col,
            't_to_exp': t_to_exp_col,
            'rate': rate_col,
            'forward': forward_col,
            'implied_vol': implied_vol_col,
            'iv_price': iv_price_col,
            'delta': delta_col,
            'vega': vega_col
        }
        if trade_date_col is None and single_trade_date is not None:
            if 'trade_date' in self.raw_data_df.columns:
                print("WARNING: 'trade_date' column exists in raw_data_df but has not been set as trade_date_col;\n"
                      "         object init() will overwrite this column, so please ensure that's intentional.")
            self.data_df['trade_date'] = datelike_to_timestamp(single_trade_date)
            self.data_cols['trade_date'] = 'trade_date'
        else:
            self.data_cols['trade_date'] = trade_date_col   # Includes dangerous case of None
        self.data_cols = {k: v for k, v in self.data_cols.items() if v is not None}     # Filter out None
        # Format call-put indicator column
        cp_ser = self.data_df[self.data_cols['call_put']].copy()
        if cp_ser.dtypes == bool:
            # Convert True to 'C' and False to 'P' for clarity
            self.data_df.loc[cp_ser, self.data_cols['call_put']] = 'C'
            self.data_df.loc[~cp_ser, self.data_cols['call_put']] = 'P'
        # Drop NaNs, index, and sort raw_data_df
        self.data_df = (self.data_df.dropna(how='all')
                        .set_index([self.data_cols['trade_date'], self.data_cols['expiry_date'],
                                    self.data_cols['call_put'], self.data_cols['strike']])
                        .sort_index())


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
