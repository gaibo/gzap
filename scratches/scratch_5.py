import pandas as pd

IBOXX_DIR = 'P:/ProductDevelopment/Database/sftp_transfers/Iboxx/'

components = pd.read_csv(IBOXX_DIR+'iboxx_B180402_usd_hy_eom_components_202104.csv')
