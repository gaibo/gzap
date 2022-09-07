import futures_reader

DOWNLOADS_DIR = 'C:/Users/gzhang/Downloads/'

good_ticker_list = \
    ['UXF12 Index', 'UXG12 Index', 'UXH12 Index', 'UXJ12 Index', 'UXK12 Index', 'UXM12 Index',
     'UXN12 Index', 'UXQ12 Index', 'UXU12 Index', 'UXV12 Index', 'UXX12 Index', 'UXZ12 Index',
     'UXF13 Index', 'UXG13 Index', 'UXH13 Index', 'UXJ13 Index', 'UXK13 Index', 'UXM13 Index',
     'UXN13 Index', 'UXQ13 Index', 'UXU13 Index', 'UXV13 Index', 'UXX13 Index', 'UXZ13 Index',
     'UXF14 Index', 'UXG14 Index', 'UXH14 Index', 'UXJ14 Index', 'UXK14 Index', 'UXM14 Index',
     'UXN14 Index', 'UXQ14 Index', 'UXU14 Index', 'UXV14 Index', 'UXX14 Index', 'UXZ14 Index',
     'UXF15 Index', 'UXG15 Index', 'UXH15 Index', 'UXJ15 Index', 'UXK15 Index', 'UXM15 Index',
     'UXN15 Index', 'UXQ15 Index', 'UXU15 Index', 'UXV15 Index', 'UXX15 Index', 'UXZ15 Index',
     'UXF16 Index', 'UXG16 Index', 'UXH16 Index', 'UXJ16 Index', 'UXK16 Index', 'UXM16 Index',
     'UXN16 Index', 'UXQ16 Index', 'UXU16 Index', 'UXV16 Index', 'UXX16 Index', 'UXZ16 Index',
     'UXF17 Index', 'UXG17 Index', 'UXH17 Index', 'UXJ17 Index', 'UXK17 Index', 'UXM17 Index',
     'UXN17 Index', 'UXQ17 Index', 'UXU17 Index', 'UXV17 Index', 'UXX17 Index', 'UXZ17 Index',
     'UXF18 Index', 'UXG18 Index', 'UXH18 Index', 'UXJ18 Index', 'UXK18 Index', 'UXM18 Index',
     'UXN18 Index', 'UXQ18 Index', 'UXU18 Index', 'UXV18 Index', 'UXX18 Index', 'UXZ18 Index',
     'UXF19 Index', 'UXG19 Index', 'UXH19 Index', 'UXJ19 Index', 'UXK19 Index', 'UXM19 Index',
     'UXN19 Index', 'UXQ19 Index', 'UXU19 Index', 'UXV19 Index', 'UXX19 Index', 'UXZ19 Index',
     'UXF20 Index', 'UXG20 Index', 'UXH20 Index', 'UXJ20 Index', 'UXK20 Index', 'UXM20 Index',
     'UXN20 Index', 'UXQ20 Index', 'UXU20 Index', 'UXV20 Index', 'UXX20 Index', 'UXZ20 Index',
     'UXF1 Index', 'UXG1 Index', 'UXH1 Index', 'UXJ1 Index', 'UXK1 Index', 'UXM1 Index',
     'UXN1 Index', 'UXQ1 Index', 'UXU1 Index', 'UXV1 Index']

vix_futures = futures_reader.pull_fut_prices(
               fut_codes='UX', start_datelike='2012-01-03', end_datelike='2021-02-11',
               end_year_current=True, n_maturities_past_end=3,
               contract_cycle='monthly', product_type='Index', ticker_list=good_ticker_list,
               file_dir=DOWNLOADS_DIR, file_name='vix_futures_pull_2021-02-11.csv',
               bloomberg_con=None, verbose=True)

#### Manual control version

con = futures_reader.create_bloomberg_connection()
vix_futures_alt = con.bdh(good_ticker_list, ['PX_LAST', 'PX_VOLUME', 'OPEN_INT'],
                          start_date='20120103', end_date='20210211')
vix_futures_alt = vix_futures_alt[good_ticker_list]     # Enforce column order!
vix_futures_alt.to_csv(DOWNLOADS_DIR + 'vix_futures_pull_2021-02-11.csv')
