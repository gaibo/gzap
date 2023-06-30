import pandas as pd
import matplotlib as mpl

import yfinance as yf

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.async_support.timeseries import TimeSeries as AsyncTimeSeries
import asyncio
from alpha_vantage.fundamentaldata import FundamentalData

import pandas_datareader as pdr
from pandas_datareader import wb
import pandas_datareader.data as web

mpl.use('Qt5Agg')   # matplotlib 3.5 changed default backend and PyCharm freezes; revert to avoid problem
pd.plotting.register_matplotlib_converters()    # Register pandas plotting with matplotlib


# yfinance #####################################################################

# "Download" method
tickers = ['AAPL', 'GOOGL']
stock_data = yf.download(tickers, start='2023-01-01', end='2023-06-26', group_by='ticker')
# Pre/post market hours data filter and repair of obvious price errors (100x) are very nice
alt_data = yf.download(tickers='SPY AAPL', period='1y', interval='1d', prepost=False, repair=True)

# "Ticker" method
msft = yf.Ticker('MSFT')
info_ret = msft.info    # Frustrating property-based managed attribute used for querying/caching all info
hist = msft.history(period='1mo')   # Historical OHLC, volume, dividends, stock splits
metadata_dict = msft.history_metadata   # Meta info about history after it's been called
shares_hist = msft.get_shares_full(start='2023-01-01', end=None)  # A couple updates a month, it looks like
yearly_cashflows = msft.cashflow   # More financials are available
print(msft.major_holders)
print(msft.institutional_holders)
print(msft.mutualfund_holders)
print(msft.earnings_dates)
news_list = msft.news   # That's wild
# Okay finally something cool - options chain!
# Okay deadass I think you have to manually cycle through expiration dates
options_expirations = msft.options
options_df_list = []
for exp in options_expirations:
    opt = msft.option_chain(exp)
    exp_calls = opt.calls.copy()
    exp_puts = opt.puts.copy()
    exp_calls['expirationDate'] = exp_puts['expirationDate'] = pd.Timestamp(exp)
    exp_calls['callPut'], exp_puts['callPut'] = 'C', 'P'
    options_df_list.extend([exp_calls, exp_puts])
options_df = pd.concat(options_df_list)
options_df = options_df.set_index(['expirationDate', 'callPut', 'strike']).sort_index(ascending=[True, False, True])
# Time comes UTC, so manually convert to Chicago (automatically does daylight savings, not sure if that's good...)
options_df['lastTradeDate'] = options_df['lastTradeDate'].dt.tz_convert('America/Chicago')
otm_df = options_df[~options_df['inTheMoney']].loc[options_expirations[1]]  # OTMs are more liquid

# yf.pdr_override()     # Hijack pandas_datareader.data.get_data_yahoo() to speed it up!


# Alpha Vantage ################################################################
# My API key is 16XCF7K429HGVIGJ; can be stored in environment variable ALPHAVANTAGE_API_KEY
# rapidAPI can also be used for the key variable; use that key and set rapidapi=True

# Set up TimeSeries handler
ts = TimeSeries(key='16XCF7K429HGVIGJ', output_format='pandas', indexing_type='date')   # 'integer' indexing also
# Get object with intraday (default 15 min intervals) data and another with call's metadata
data, meta_data = ts.get_intraday('MSFT', interval='1min', outputsize='full')  # Pulls about a month; 'compact' 100
msft_dailyadj, msft_dailyadj_meta = ts.get_daily_adjusted('MSFT')

# Set up TechIndicators handler
ti = TechIndicators(key='16XCF7K429HGVIGJ', output_format='pandas')
msft_ti, msft_ti_meta = ti.get_bbands(symbol='MSFT', interval='60min', time_period=60)  # 60 data points window

# Set up SectorPerformances handler
sp = SectorPerformances(key='16XCF7K429HGVIGJ', output_format='pandas')
data_sp, data_sp_meta = sp.get_sector()     # As of 2023-06-28 this gives "error getting data from api"

# Set up CryptoCurrencies handler
cc = CryptoCurrencies(key='16XCF7K429HGVIGJ', output_format='pandas')
eth_data, eth_data_meta = cc.get_digital_currency_daily(symbol='ETH', market='CNY')
# Okay LMAO this is useless as of 2023-06-28 - they just multiply the USD column by TODAY's constant exchange rate
# to get the CNY values. That is unusual levels of laziness.

# Set up ForeignExchange handler
fe = ForeignExchange(key='16XCF7K429HGVIGJ', output_format='pandas')
fe_data, no_meta = fe.get_currency_exchange_rate(from_currency='BTC', to_currency='USD')    # No metadata for this


# Asyncio support
# NOTE: to try this code, must import the async version of TimeSeries, which returns awaitable coroutine objects!
async def get_data_av(symbol):
    ts_async = AsyncTimeSeries(key='16XCF7K429HGVIGJ', output_format='pandas')
    async_data, _ = await ts_async.get_quote_endpoint(symbol)
    await ts_async.close()
    return async_data


symbols = ['AAPL', 'GOOG', 'TSLA', 'MSFT']
loop = asyncio.get_event_loop()
tasks = [get_data_av(symbol) for symbol in symbols]
group1 = asyncio.gather(*tasks)
results = loop.run_until_complete(group1)
loop.close()
print(results)

# Set up FundamentalData handler
fd = FundamentalData(key='16XCF7K429HGVIGJ', output_format='pandas')
cash_flow = fd.get_cash_flow_annual('TSLA')


# Pandas-DataReader ############################################################

fed_data = pdr.get_data_fred('GS10')    # 10-year CMT yields
# World Bank GDP per capita in constant dollars in North America
wb_data = wb.download(indicator='NY.GDP.PCAP.KD', country=['US', 'CA'], start=2016, end=2020)

# Tiingo - historical EOD prices on equities, mutual funds, ETFs
# IEX - historical stock prices for 15 years
# os.environ["IEX_API_KEY"] = "pk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" or export key before starting IPython session

# Alpha Vantage - fun fact, "daily adjusted" data is free but "daily" is PREMIUM lmao
# As of 2023-06-29, pandas_datareader's type hint for api_key is wrong - they did tuple instead of set
# LMAO, Alpha Vantage doesn't do start and end dates, so pandas_datareader is deciding between 'compact'
# and 'full' and then filtering by itself
f = web.DataReader('MSFT', 'av-daily-adjusted', start='2023-02-09', end='2023-05-24', api_key='16XCF7K429HGVIGJ')
f_compare = msft_dailyadj.loc['2023-02-09':'2023-05-24'].sort_index()
assert (f.values == f_compare.values).all()     # Different column names but identical output

# Stooq index data
stooq_spx = web.DataReader('^SPX', 'stooq')     # 5 years OHLC and "volume"?

# Yahoo
# LMAOOO as of 2023-06 and I think since December of last year, entirety of Yahoo is bricked
# because Yahoo introduced some encryption. There are active pull requests but pandas_datareader is not active.
# Looks like commits have resumed within the past 2 weeks though after 1.5 years of nothing, so there's hope.
msft_yahoo = web.DataReader('MSFT', 'yahoo', start='2023-05-30', end='2023-06-29')
