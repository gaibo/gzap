import pandas as pd

CVIX_UPDATED_LOC = 'data/cvix_spotindex_all.csv'
CVIX_HISTORICAL_LOC = 'data/cvix_historical.csv'
INDEX_SYMBOLS = ['VIXIG', 'VIXHY', 'VIXIE', 'VIXXO', 'VIXFS']
REFLOC_DICT = {'VIXIG': 'N1730', 'VIXHY': 'N1730',
               'VIXIE': 'L1730', 'VIXXO': 'L1730', 'VIXFS': 'L1730'}    # North America: NYC, Europe: London


def read_credit_vix(new_loc=CVIX_UPDATED_LOC, hist_loc=CVIX_HISTORICAL_LOC):
    """ Load DataFrame containing all Credit VIX data
        NOTE: Credit VIX has "historical" values going up until 2018-04-25,
              at which point there is a break until 2018-08-23, at which point an
              automated daily calculation was implemented in SAS. This function was
              written to streamline consolidating from the two data sources.
    :param new_loc: location of CSV file containing actively updated values
    :param hist_loc: location of CSV file containing inactive historical values
    :return: pandas DataFrame with multiindexed columns
    """
    # Read actively updated CSV
    df = pd.read_csv(new_loc, parse_dates=['Date'],
                     usecols=['Date', 'IndexSymbol', 'IndexSeries', 'IndexVersion', 'refloc',
                              'CreditVix_pc', 'CreditVix_bp'])
    # Get the latest series and version index values for each ticker (at correct time)
    index_df_list = []
    for symbol in INDEX_SYMBOLS:
        symbol_df = df[(df['IndexSymbol'] == symbol) &
                       (df['refloc'] == REFLOC_DICT[symbol])].drop(['IndexSymbol', 'refloc'], axis=1)
        sorted_symbol_df = symbol_df.fillna(0).sort_values(['Date', 'IndexSeries', 'IndexVersion'])
        symbol_latest_df = sorted_symbol_df.groupby('Date').last()[['CreditVix_pc', 'CreditVix_bp']]
        multiindex = pd.MultiIndex.from_product([[symbol], symbol_latest_df.columns],
                                                names=['IndexSymbol', 'IndexType'])
        symbol_latest_df.columns = multiindex   # To facilitate joining and exporting
        index_df_list.append(symbol_latest_df)
    # Join all five Credit VIX tickers into DataFrame
    new_df = index_df_list[0].join(index_df_list[1:], how='outer')

    # Read inactive "historical" CSV
    hist_df = pd.read_csv(hist_loc, index_col=0, parse_dates=True, header=[0, 1])

    # Join old and new data sources
    return pd.concat([hist_df, new_df])
