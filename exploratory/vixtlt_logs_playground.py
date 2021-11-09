import pandas as pd
from pathlib import Path

LOGS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Gaibo/VIXTLT/VIXTLTP Log Pulls/')


def time_fix_helper(time_string):
    time_string[-4] = '.'
    return time_string


# Load pre-cleaned data
nncp = pd.read_excel(LOGS_DIR/'VXTLTPE_2021_09_20_10am_11am_split.xlsx',
                     sheet_name='Near Next Calls Puts', parse_dates=[['Calc Date', 'Calc Time']])
nncp = nncp.rename({'Calc Date_Calc Time': 'Calc Datetime'}, axis=1)
# nncp_raw = pd.read_excel(LOGS_DIR/'VXTLTPE_2021_09_20_10am_11am_split.xlsx',
#                          sheet_name='Near Next Calls Puts')
# nncp = nncp_raw.copy()
# nncp['Calc Time'] = \
#     pd.to_timedelta(nncp['Calc Time'].apply(lambda s: s[:-4]+'.'+s[-3:]))   # Original data has too many colons
# nncp['Calc Datetime'] = nncp['Calc Date'] + nncp['Calc Time']
# nncp = nncp.drop(['Calc Date', 'Calc Time'], axis=1).set_index('Calc Datetime')

# Grab approximately OTM
nncp_otm = nncp[((nncp['Calls or Puts'] == 'Puts') & (nncp['Strike'] <= 160))
                | ((nncp['Calls or Puts'] == 'Calls') & (nncp['Strike'] >= 140))]

# See when data refreshes and by how much
nncp_indexed = (nncp_otm.set_index(['Term', 'Maturity', 'Calls or Puts', 'Strike', 'Calc Datetime'])
                .sort_index(ascending=[True, True, False, True, True]))

# Group calculation inputs by 15-second interval
