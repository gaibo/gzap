import pandas as pd
from model.data_structures import ETF, Index, VolatilityIndex
from utility.read_credit_vix import read_credit_vix


# Miscellaneous Time-Series Pulled from Bloomberg
bbg_data = pd.read_csv('data/bbg_automated_pull.csv',
                       index_col=0, parse_dates=True, header=[0, 1])
# Create objects
# Mature VIXs
# VIX
spx = Index(bbg_data['SPX Index', 'PX_LAST'], 'SPX')
vix = VolatilityIndex(bbg_data['VIX Index', 'PX_LAST'], spx, 'VIX')
# TYVIX
ty1 = Index(bbg_data['TY1 Comdty', 'PX_LAST'], 'TY1')
tyvix = VolatilityIndex(bbg_data['TYVIX Index', 'PX_LAST'], ty1, 'TYVIX')
# JGB VIX
jb1 = Index(bbg_data['JB1 Comdty', 'PX_LAST'], 'JB1')
jgbvix = VolatilityIndex(bbg_data['SPJGBV Index', 'PX_LAST'], jb1, 'JGB VIX')
# SRVIX
usfs0110 = Index(bbg_data['USFS0110 CMPN Curncy', 'PX_LAST'], '1Y-10Y Forward Swap Rate')
srvix = VolatilityIndex(bbg_data['SRVIX Index', 'PX_LAST'], usfs0110, 'SRVIX')
# VSTOXX
sx5e = Index(bbg_data['SX5E Index', 'PX_LAST'], 'Euro Stoxx 50')
vstoxx = VolatilityIndex(bbg_data['V2X Index', 'PX_LAST'], sx5e, 'VSTOXX')
# ETFs
# Total Return
spx_tr = Index(bbg_data['SPX Index', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'SPX')
agg_tr = ETF(bbg_data['AGG Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'AGG')
hyg_tr = ETF(bbg_data['HYG Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'HYG')
ihyg_tr = ETF(bbg_data['IHYG EU Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'IHYG')
ief_tr = ETF(bbg_data['IEF Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'IEF')
lqd_tr = ETF(bbg_data['LQD Equity', 'TOT_RETURN_INDEX_GROSS_DVDS'], 'LQD')

##############################################################################

# Unofficial In-House Basis Point Versions
# Load additional data
tyvix_bp_data = pd.read_csv('data/tyvix_bp.csv',
                            index_col='Trade Date', parse_dates=True)
jgbvix_bp_data = pd.read_csv('data/jgbvix_bp.csv',
                             index_col='Trade Date', parse_dates=True)  # Rough estimate
# Create objects
# NOTE: yields from Bloomberg are used as the underlying to allow realized vol calculations
# BP TYVIX
ty1_yield = Index(bbg_data['TY1 Comdty', 'YLD_CNV_LAST'], 'TY1 Yield')
tyvix_bp = VolatilityIndex(tyvix_bp_data['BP TYVIX'], ty1_yield, 'BP TYVIX')
# BP JGB VIX
jb1_yield = Index(bbg_data['JB1 Comdty', 'YLD_CNV_LAST'], 'JB1 Yield')
jgbvix_bp = VolatilityIndex(jgbvix_bp_data['BP JGBVIX'], jb1_yield, 'BP JGB VIX')

#############################################################################

# Credit VIX
# Load additional data
creditvix_data = read_credit_vix()
scaled_cds_index_data = pd.read_csv('data/scaled_cds_indexes.csv',
                                    index_col='date', parse_dates=True)
# Create objects
# NOTE: "normalized" CDS indexes have past series scaled up to magnitude of the present;
#       they are used as the underlying for the percent Credit VIXes, for which they can be
#       used to calculate realized vol in a much more meaningful manner; non-scaled
#       indexes are used for the basis point Credit VIXes, since the magnitudes matter
# CDX NA IG (Investment Grade)
cdx_ig_normalized = Index(scaled_cds_index_data['CDX NA IG'], 'CDX NA IG (Normalized)')
cdx_ig = Index(bbg_data['IBOXUMAE CBBT Curncy', 'PX_LAST'], 'CDX NA IG')
vixig = VolatilityIndex(creditvix_data['VIXIG', 'CreditVix_pc'], cdx_ig_normalized, 'VIXIG')
vixig_bp = VolatilityIndex(creditvix_data['VIXIG', 'CreditVix_bp'], cdx_ig, 'BP VIXIG')
# CDX NA HY (High Yield)
cdx_hy_normalized = Index(scaled_cds_index_data['CDX NA HY'], 'CDX NA HY (Normalized)')
cdx_hy = Index(bbg_data['IBOXHYSE CBBT Curncy', 'PX_LAST'], 'CDX NA HY')
vixhy = VolatilityIndex(creditvix_data['VIXHY', 'CreditVix_pc'], cdx_hy_normalized, 'VIXHY')
vixhy_bp = VolatilityIndex(creditvix_data['VIXHY', 'CreditVix_bp'], cdx_hy, 'BP VIXHY')
# iTraxx EU Main (Investment Grade)
itraxx_ie_normalized = Index(scaled_cds_index_data['iTraxx EU Main'], 'iTraxx EU Main (Normalized)')
itraxx_ie = Index(bbg_data['ITRXEBE CBBT Curncy', 'PX_LAST'], 'iTraxx EU Main')
vixie = VolatilityIndex(creditvix_data['VIXIE', 'CreditVix_pc'], itraxx_ie_normalized, 'VIXIE')
vixie_bp = VolatilityIndex(creditvix_data['VIXIE', 'CreditVix_bp'], itraxx_ie, 'BP VIXIE')
# iTraxx EU Crossover (High Yield)
itraxx_xo_normalized = Index(scaled_cds_index_data['iTraxx EU Xover'], 'iTraxx EU Xover (Normalized)')
itraxx_xo = Index(bbg_data['ITRXEXE CBBT Curncy', 'PX_LAST'], 'iTraxx EU Xover')
vixxo = VolatilityIndex(creditvix_data['VIXXO', 'CreditVix_pc'], itraxx_xo_normalized, 'VIXXO')
vixxo_bp = VolatilityIndex(creditvix_data['VIXXO', 'CreditVix_bp'], itraxx_xo, 'BP VIXXO')
# iTraxx EU Senior Financials
itraxx_fs_normalized = Index(scaled_cds_index_data['iTraxx EU SenFin'], 'iTraxx EU SenFin (Normalized)')
itraxx_fs = Index(bbg_data['ITRXESE CBBT Curncy', 'PX_LAST'], 'iTraxx EU SenFin')
vixfs = VolatilityIndex(creditvix_data['VIXFS', 'CreditVix_pc'], itraxx_fs_normalized, 'VIXFS')
vixfs_bp = VolatilityIndex(creditvix_data['VIXFS', 'CreditVix_bp'], itraxx_fs, 'BP VIXFS')
