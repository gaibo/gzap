from impala.dbapi import connect
from impala.util import as_pandas
import pandas as pd

# Establish connection to Data Platform with Impala (distributed SQL database that runs on Hadoop cluster)
conn = connect(host='dpc.us.cboe.net', port=21050)

# Test query: BZX equities order_summary table
# NOTE: as of 2021-01-08, equities_us.order_summary does not have column "exchange";
#       this query no longer works
# TEST_SQL_QUERY = ("SELECT exchange, order_id, modify_sequence, side, price, size "
#                   "FROM equities_us.order_summary "
#                   "WHERE exchange='00' "
#                   "LIMIT 10")
TEST_SQL_QUERY = ("SELECT firm, order_id, modify_sequence, side, price, size "
                  "FROM equities_us.order_summary "
                  "WHERE firm='GSCO' "
                  "LIMIT 10")

#%%#### Method 1: Cursor

# Get a connection cursor and use its execute() function
cursor = conn.cursor()
cursor.execute(TEST_SQL_QUERY)
print(f"cursor.fetchall():\n{cursor.fetchall()}")   # List
cursor.execute(TEST_SQL_QUERY)
print(f"cursor.fetchone():\n{cursor.fetchone()}")   # Iterable
cursor.execute(TEST_SQL_QUERY)
print("for-loop on cursor:")  # Can also just treat cursor as iterable
for row in cursor:
    print(row)

# Cursor can directly be converted into DataFrame
cursor.execute(TEST_SQL_QUERY)  # The .fetch methods on cursor expend it, so reload cursor
cursor_df = as_pandas(cursor)
print(f"as_pandas(cursor):\n{cursor_df}")

#%%#### Method 2: pandas.read_sql()

# Directly pass connection object into pandas function
alt_df = pd.read_sql(con=conn, sql=TEST_SQL_QUERY)
print(f"pd.read_sql(conn, TEST_SQL_QUERY):\n{alt_df}")

#%%#### Cleanup

# Close the established connection
conn.close()    # Empirically we find this doesn't work lol - pd.read_sql() still works
