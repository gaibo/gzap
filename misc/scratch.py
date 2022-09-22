# Simplified way to get email attachment
# Specify mail server
conn = imaplib.IMAP4_SSL("outlook.office365.com")
# Log in
conn.login("gzhang@cboe.com", "better with you up")
# Select mailbox (default Inbox)
conn.select("\"Bulk/Dennis BlackRock iShares Daily Values\"")
# Search messages
check_search, msg_nums_list = conn.search(None, 'ALL')
# Get a message
latest_msg_num = msg_nums_list[0].split()[-1]   # Note this is byte encoded
check_fetch, data = conn.fetch(latest_msg_num, '(RFC822)')
msg = email.message_from_string(data[0][1].decode())    # Decode from byte
# Export attachment to disk
download_folder = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/')
att_path = "No attachment found."
for part in msg.walk():
    if part.get_content_maintype() == 'multipart':
        continue
    if part.get('Content-Disposition') is None:
        continue
    filename = part.get_filename()
    att_path = os.path.join(download_folder, filename)
    if not os.path.isfile(att_path):
        fp = open(att_path, 'wb')
        fp.write(part.get_payload(decode=True))
        fp.close()
print(att_path)


# TODO: future Gaibo, please read all about classes and their built-ins, like __init__, etc. Then thoroughly explore
# TODO: the imaplib/email options for how messages are structured - make class able to deal with text and attachments,
# TODO: login and logout, close and stuff

import email
import imaplib
import os

class FetchEmail:
    connection = None
    error = None
    mail_server = "host_name"
    username = "outlook_username"
    password = "password"

    def __init__(self, mail_server, username, password):
        self.connection = imaplib.IMAP4_SSL(mail_server)
        self.connection.login(username, password)
        self.connection.select(readonly=False) # so we can mark mails as read

    def close_connection(self):
        """
        Close the connection to the IMAP server
        """
        self.connection.close()

    def save_attachment(self, msg, download_folder="/tmp"):
        """
        Given a message, save its attachments to the specified
        download folder (default is /tmp)

        return: file path to attachment
        """
        att_path = "No attachment found."
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue

            filename = part.get_filename()
            att_path = os.path.join(download_folder, filename)

            if not os.path.isfile(att_path):
                fp = open(att_path, 'wb')
                fp.write(part.get_payload(decode=True))
                fp.close()
        return att_path

    def fetch_unread_messages(self):
        """
        Retrieve unread messages
        """
        emails = []
        (result, messages) = self.connection.search(None, 'UnSeen')
        if result == "OK":
            for message in messages[0].split(' '):
                try:
                    ret, data = self.connection.fetch(message,'(RFC822)')
                except:
                    print "No new emails to read."
                    self.close_connection()
                    exit()

                msg = email.message_from_string(data[0][1])
                if isinstance(msg, str) == False:
                    emails.append(msg)
                response, data = self.connection.store(message, '+FLAGS','\\Seen')

            return emails

        self.error = "Failed to retrieve emails."
        return emails


from pathlib import Path
DOWNLOADS_DIR = Path('C:/Users/gzhang/OneDrive - CBOE/Downloads/')
# SPX index options price grab
import pandas as pd
from futures_reader import create_bloomberg_connection
# # Generate ticker list of format "SPX US 12/16/22 C7000 Index"
# # For expiries 12/16/22 and 12/15/23 | C and P | 1700-7300 by 25
# ticker_list = [f"SPX US {exp_str} {cp}{strike_str} Index"
#                for exp_str in ['12/16/22', '12/15/23']
#                for cp in ['C', 'P']
#                for strike_str in range(1700, 7325, 25)]
# Read Danny's options series for exact ticker list
danny_df = pd.read_csv(DOWNLOADS_DIR/'all_baml_spx.csv')


def merge_danny_cols(danny_row):
    exp_str = pd.Timestamp(danny_row['expiry']).strftime('%m/%d/%y')
    cp = danny_row['putCall']
    strike_str = str(round(danny_row['strike']))
    return f"SPX US {exp_str} {cp}{strike_str} Index"


ticker_list = danny_df.apply(merge_danny_cols, axis=1).tolist()
# Establish connection to Bloomberg API
con = create_bloomberg_connection()
# Bloomberg configs
START_DATE = END_DATE = pd.Timestamp('2022-05-16')  # MODIFY THIS
# Pull if ticker exists (ValueError otherwise)
res_list = []
good_ticker_list = []
bad_ticker_list = []    # Using Danny's exact series, this should stay empty
counter = 0     # Print sanity check
for ticker in ticker_list:
    try:
        res = con.bdh(ticker, ['PX_BID', 'PX_ASK', 'PX_MID', 'PX_LAST'],
                      f"{START_DATE.strftime('%Y%m%d')}", f"{END_DATE.strftime('%Y%m%d')}")
        if res.empty:
            # Update 2022-05-12: when options have no HP prices, con.bdh return empty DF instead of a row of NaNs
            bad_ticker_list.append(ticker)
            continue
        res = res.droplevel(0, axis=1)  # Drop column index level containing ticker
        res_list.append(res)
        good_ticker_list.append(ticker)
        counter += 1
        print(f"Prices pulled: {counter}")
    except ValueError:
        bad_ticker_list.append(ticker)
        continue


def split_ticker_into_exp_cp_strike(ticker_in):
    ticker_split = ticker_in.split()
    exp_out = pd.Timestamp(ticker_split[2])
    cp_out = ticker_split[3][0]
    strike_out = int(ticker_split[3][1:])
    return exp_out, cp_out, strike_out


# Concatenate
res_df = pd.concat(res_list).reset_index(drop=True)     # Index is useless single date
context_df = pd.DataFrame(map(split_ticker_into_exp_cp_strike, good_ticker_list),
                          columns=['Expiry', 'PC', 'Strike'])   # Tuple list into columns
assert res_df.shape[0] == context_df.shape[0]
bad_context_df = pd.DataFrame(map(split_ticker_into_exp_cp_strike, bad_ticker_list),
                              columns=['Expiry', 'PC', 'Strike'])
good_df = pd.concat([context_df, res_df], axis=1)
good_bad_df = pd.concat([good_df, bad_context_df])
spx_df = good_bad_df.set_index(['Expiry', 'PC', 'Strike']).sort_index()
spx_df = spx_df[['PX_BID', 'PX_ASK', 'PX_MID', 'PX_LAST']]  # Enforce column order
# Export
spx_df.to_csv(DOWNLOADS_DIR/f'{END_DATE.strftime("%Y-%m-%d")}_SPX_options_BBG_for_Danny.csv')

####

# Fix 2022-05-12 bug without pulling data again
# Back up the buggy lists
false_res_list = res_list
false_good_ticker_list = good_ticker_list
false_bad_ticker_list = bad_ticker_list
false_res_df = res_df.copy()
false_context_df = context_df.copy()
false_spx_df = spx_df.copy()
# Fix them
true_res_list = [res for res in false_res_list if not res.empty]
res_list = true_res_list
empty_bool_col = [res.empty for res in false_res_list]
empty_bool_df = pd.DataFrame({'Empty_Bool': empty_bool_col, 'Ticker': false_good_ticker_list})
true_good_ticker_list = empty_bool_df.loc[~empty_bool_df['Empty_Bool'], 'Ticker'].tolist()
good_ticker_list = true_good_ticker_list
true_bad_ticker_list = empty_bool_df.loc[empty_bool_df['Empty_Bool'], 'Ticker'].tolist()
bad_ticker_list = true_bad_ticker_list

# Redo the output
res_df = pd.concat(res_list).reset_index(drop=True)     # Index is useless single date
context_df = pd.DataFrame(map(split_ticker_into_exp_cp_strike, good_ticker_list),
                          columns=['Expiry', 'PC', 'Strike'])   # Tuple list into columns
assert res_df.shape[0] == context_df.shape[0]
bad_context_df = pd.DataFrame(map(split_ticker_into_exp_cp_strike, bad_ticker_list),
                              columns=['Expiry', 'PC', 'Strike'])
good_df = pd.concat([context_df, res_df], axis=1)
good_bad_df = pd.concat([good_df, bad_context_df])
spx_df = good_bad_df.set_index(['Expiry', 'PC', 'Strike']).sort_index()
spx_df = spx_df[['PX_BID', 'PX_ASK', 'PX_MID', 'PX_LAST']]  # Enforce column order
# Export
spx_df.to_csv(DOWNLOADS_DIR/f'{END_DATE.strftime("%Y-%m-%d")}_SPX_options_BBG_for_Danny.csv')

####

# Unique new accounts each day for IBHY
# Step 1: Aggregate size by date, CTI, firm, account; each row will be unique
ibhy_new = ibhy.groupby(['Trade Date', 'CTI', 'Name', 'Account '])['Size'].sum()

# Step 2: Generate "known" set of accounts for each day
ibhy_days = ibhy_new.index.get_level_values('Trade Date').unique()
known_set_dict = {ibhy_days[0]: set()}  # No known accounts on first day
for day, prev_day in zip(ibhy_days[1:], ibhy_days[:-1]):
    known_set_dict[day] = \
        (known_set_dict[prev_day]
         | set(ibhy_new.loc[prev_day].index.get_level_values('Account ')))

# Step 3: Mark accounts each day that were not known at the beginning of the day
ibhy_new_reset = ibhy_new.reset_index('Account ')
for day in ibhy_days:
    # .values is great for doing stuff with no unique indexes
    ibhy_new_reset.loc[day, 'New Account'] = \
        (~ibhy_new_reset.loc[day]['Account '].isin(known_set_dict[day])).values

# Step 4: Aggregate for final results
ibhy_new_accounts = ibhy_new_reset.groupby(['Trade Date', 'CTI'])['New Account'].sum().unstack()
# Use .fillna(method='ffill') on cumsum to fill NaNs, but NaNs actually give you info on what days the CTI had volume
ibhy_new_accounts_cumsum = ibhy_new_accounts.cumsum()
# Exportable
ibhy_new_accounts_export = ibhy_new_reset


git submodule update --remote --merge   # If you make changes to submodules I think
git submodule update    # Updates to superproject's version I think
git submodule update --remote   # Updates to absolute latest from web I think

# Basis Point Index vs. Realized and Difference
bp_list = [tyvix_bp, jgbvix_bp, srvix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]
for obj, color in zip(bp_list, color_list):
    if obj.name == 'SRVIX':
        _, axs = plt.subplots(2, 1, sharex='all')
        [joined_index, joined_undl_rv] = \
            share_dateindex([obj.price(), obj.undl_realized_vol(do_shift=True, window=252, bps=True)])
        make_lineplot([joined_index, joined_undl_rv], [obj.name, 'Realized Volatility'],
                      ylabel='Volatility (bps)',
                      title='{} with Realized ({} Days Shifted)'.format(obj.name, 252), ax=axs[0])
        difference = joined_index - joined_undl_rv
        # make_fillbetween(difference.index, joined_index, joined_undl_rv,
        #                  label='Difference', color='mediumseagreen', ax=axs[0])
        make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
        make_lineplot([difference], color_list=['g'], ax=axs[1])
    else:
        if obj.name in [vixig_bp.name, vixhy_bp.name, vixie_bp.name, vixxo_bp.name, vixfs_bp.name]:
            in_bps = True
        else:
            in_bps = False
        _, axs = plt.subplots(2, 1, sharex='all')
        [joined_index, joined_undl_rv] = \
            share_dateindex([obj.price(), obj.undl_realized_vol(do_shift=True, bps=True, price_in_bps=in_bps)])
        make_lineplot([joined_index, joined_undl_rv], [obj.name, 'Realized Volatility'],
                      ylabel='Volatility (bps)',
                      title='{} with Realized ({} Days Shifted)'.format(obj.name, 21), ax=axs[0])
        difference = joined_index - joined_undl_rv
        # make_fillbetween(difference.index, joined_index, joined_undl_rv,
        #                  label='Difference', color='mediumseagreen', ax=axs[0])
        make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
        make_lineplot([difference], color_list=['g'], ax=axs[1])

####

bp_list = [tyvix_bp, jgbvix_bp, srvix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]
color_list = ['C2', 'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

# Basis Point Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([bp_list[0].price(), bp_list[0].undl_realized_vol(do_shift=True, bps=True)])
bp_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(bp_diff, hist=False,
               label=bp_list[0].name, xlabel='Implied Vol Premium (bps)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line=color_list[0])
for o, color in zip(bp_list[1:], color_list[1:]):
    if o.name == 'SRVIX':
        [joined_index, joined_undl_rv] = \
            share_dateindex([o.price(), o.undl_realized_vol(do_shift=True, window=252, bps=True)])
        bp_diff = joined_index - joined_undl_rv
        make_histogram(bp_diff, hist=False,
                       label=o.name, xlabel='Implied Vol Premium (bps)', ylabel='Probability',
                       title='Risk Premium Distribution',
                       color_line=color, ax=ax_prem)
    else:
        if o.name in [vixig_bp.name, vixhy_bp.name, vixie_bp.name, vixxo_bp.name, vixfs_bp.name]:
            in_bps = True
        else:
            in_bps = False
        [joined_index, joined_undl_rv] = \
            share_dateindex([o.price(), o.undl_realized_vol(do_shift=True, bps=True, price_in_bps=in_bps)])
        bp_diff = joined_index - joined_undl_rv
        make_histogram(bp_diff, hist=False,
                       label=o.name, xlabel='Implied Vol Premium (bps)', ylabel='Probability',
                       title='Risk Premium Distribution',
                       color_line=color, ax=ax_prem)

cvix_list = [vixig, vixhy, vixie, vixxo, vixfs]
cvix_color_list = ['C4', 'C5', 'C6', 'C7', 'C8']

# Re-colored Credit VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
pc_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(pc_diff, hist=False,
                            label='VIX', xlabel='Implied Vol Premium (%)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
for o, color in zip(cvix_list[1:], cvix_color_list[1:]):
    [joined_index, joined_undl_rv] = \
        share_dateindex([o.price(), 100*o.undl_realized_vol(do_shift=True)])
    pc_diff = joined_index - joined_undl_rv
    make_histogram(pc_diff, hist=False,
                   label=o.name, xlabel='Implied Vol Premium (%)', ylabel='Probability',
                   title='Risk Premium Distribution',
                   color_line=color, ax=ax_prem)

irvix_list = [vix, tyvix, jgbvix, srvix]
irvix_color_list = ['C0', 'C2', 'C1', 'C3']

# Re-colored Interest Rate VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
pc_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(pc_diff, hist=False,
                            label='VIX', xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
for o, color in zip(irvix_list[1:], irvix_color_list[1:]):
    if o.name == 'SRVIX':
        [joined_index, joined_undl_rv] = \
            share_dateindex([o.price(), o.undl_realized_vol(do_shift=True, window=252, bps=True)])
        pc_diff = joined_index - joined_undl_rv
        make_histogram(pc_diff, hist=False,
                       label=o.name, xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
                       title='Risk Premium Distribution',
                       color_line=color, ax=ax_prem)
    else:
        [joined_index, joined_undl_rv] = \
            share_dateindex([o.price(), 100*o.undl_realized_vol(do_shift=True)])
        pc_diff = joined_index - joined_undl_rv
        make_histogram(pc_diff, hist=False,
                       label=o.name, xlabel='Implied Vol Premium (bps for SRVIX, % for other)', ylabel='Probability',
                       title='Risk Premium Distribution',
                       color_line=color, ax=ax_prem)

####

# Medians (with dates aligned)
bp_list = [tyvix_bp, jgbvix_bp, srvix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]
bp_dict = {name: median for name, median in zip(map(lambda o: o.name, bp_list), map(lambda o: o.price().loc['2012-06-18':].median(), bp_list))}
bp_median_table = pd.DataFrame(bp_dict, index=['Median Level (Since 2012-06-18)']).T

pc_list = [vix, tyvix, jgbvix, vixig, vixhy, vixie, vixxo, vixfs]
pc_dict = {name: median for name, median in zip(map(lambda o: o.name, pc_list), map(lambda o: o.price().loc['2012-06-18':].median(), pc_list))}
pc_median_table = pd.DataFrame(pc_dict, index=['Median Level (Since 2012-06-18)']).T

####

bp_list = [tyvix_bp, jgbvix_bp, srvix, vixig_bp, vixhy_bp, vixie_bp, vixxo_bp, vixfs_bp]
bp_dict = {name: median for name, median in zip(map(lambda o: o.name, bp_list), map(lambda o: o.price().median(), bp_list))}
bp_median_table = pd.DataFrame(bp_dict, index=['Median Level']).T

fig, ax = plt.subplots()
ax.plot(tyvix_bp.price(), label='BP TYVIX', color='C2')
ax.plot(jgbvix_bp.price(), label='BP JGB VIX', color='C1')
ax.plot(srvix.price(), label='SRVIX', color='C3')
ax.plot(vixig_bp.price().loc[:'2018-04-25'], label='BP VIXIG', color='C4')
ax.plot(vixhy_bp.price().loc[:'2018-04-25'], label='BP VIXHY', color='C5')
ax.plot(vixie_bp.price().loc[:'2018-04-25'], label='BP VIXIE', color='C6')
ax.plot(vixxo_bp.price().loc[:'2018-04-25'], label='BP VIXXO', color='C7')
ax.plot(vixfs_bp.price().loc[:'2018-04-25'], label='BP VIXFS', color='C8')
ax.legend(fontsize=13)
ax.plot(vixig_bp.price().loc['2018-08-23':], label='BP VIXIG', color='C4')
ax.plot(vixhy_bp.price().loc['2018-08-23':], label='BP VIXHY', color='C5')
ax.plot(vixie_bp.price().loc['2018-08-23':], label='BP VIXIE', color='C6')
ax.plot(vixxo_bp.price().loc['2018-08-23':], label='BP VIXXO', color='C7')
ax.plot(vixfs_bp.price().loc['2018-08-23':], label='BP VIXFS', color='C8')
ax.set_ylabel('Volatility Index (bps)', fontsize=16)
ax.set_title('All Basis Point Volatility Indexes', fontsize=16)

fig, ax = plt.subplots()
ax.plot(tyvix_bp.price(), label='BP TYVIX', color='C2')
ax.plot(jgbvix_bp.price(), label='BP JGB VIX', color='C1')
ax.plot(srvix.price(), label='SRVIX', color='C3')
ax.legend()
ax.set_ylabel('Volatility Index (bps)')
ax.set_title('Interest Rate Volatility Indexes (Basis Point Versions)')

fig, ax = plt.subplots()
ax.plot(vixig_bp.price().loc[:'2018-04-25'], label='BP VIXIG', color='C0')
ax.plot(vixhy_bp.price().loc[:'2018-04-25'], label='BP VIXHY', color='C1')
ax.plot(vixie_bp.price().loc[:'2018-04-25'], label='BP VIXIE', color='C2')
ax.plot(vixxo_bp.price().loc[:'2018-04-25'], label='BP VIXXO', color='C3')
ax.plot(vixfs_bp.price().loc[:'2018-04-25'], label='BP VIXFS', color='C4')
ax.legend(fontsize=13)
ax.plot(vixig_bp.price().loc['2018-08-23':], label='BP VIXIG', color='C0')
ax.plot(vixhy_bp.price().loc['2018-08-23':], label='BP VIXHY', color='C1')
ax.plot(vixie_bp.price().loc['2018-08-23':], label='BP VIXIE', color='C2')
ax.plot(vixxo_bp.price().loc['2018-08-23':], label='BP VIXXO', color='C3')
ax.plot(vixfs_bp.price().loc['2018-08-23':], label='BP VIXFS', color='C4')
ax.set_ylabel('Volatility Index (bps)', fontsize=16)
ax.set_title('Credit Volatility Indexes (Basis Point Versions)', fontsize=16)

# [Figure 4] VIXs in Rates Group with VIX Index
start_date = tyvix.price().index[0]
[truncd_tyvix, truncd_jgbvix, truncd_srvix, truncd_vixig, truncd_vixhy, truncd_vixie, truncd_vixxo, truncd_vixfs] = \
    map(lambda p: p.truncate(start_date), [tyvix_bp.price(), jgbvix_bp.price(), srvix.price(),
                                           vixig_bp.price(), vixhy_bp.price(), vixie_bp.price(), vixxo_bp.price(), vixfs_bp.price()])
_, axleft = plt.subplots()
axleft.plot(truncd_vix, label='S&P500 VIX')
axleft.plot(truncd_tyvix, label='TYVIX')
axleft.plot(truncd_jgbvix, label='JGB VIX')
axleft.legend(loc=2)
axleft.set_ylabel('Volatility Index (% Price)')
axleft.set_title('VIXs in Rates Group with VIX Index')
axright = axleft.twinx()
axright.plot(truncd_srvix, label='SRVIX', color='C3')
axright.legend(loc=1)
axright.set_ylabel('Volatility Index (SRVIX) (bps)')
axright.set_ylim(20, 115)

# [Figure 16, 17, 18, 19] Interest Rate VIX Difference Charts
_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['S&P500 VIX', 'Realized Volatility'],
              ylabel='Volatility (% Price)',
              title='S&P500 VIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])
axs[0].set_ylabel('Volatility (% Price)', fontsize=16)
axs[0].set_title('S&P500 VIX with Realized (21 Days Shifted)', fontsize=16)
axs[0].legend(fontsize=14)
axs[1].legend(fontsize=14)

_, axs = plt.subplots(2, 1, sharex='all')
[joined_index, joined_undl_rv] = \
    share_dateindex([tyvix.price(), 100*tyvix.undl_realized_vol(do_shift=True)])
make_lineplot([joined_index, joined_undl_rv], ['TYVIX', 'Realized Volatility'],
              ylabel='Volatility (% Price)',
              title='TYVIX with Realized (21 Days Shifted)', ax=axs[0])
difference = joined_index - joined_undl_rv
# make_fillbetween(difference.index, joined_index, joined_undl_rv,
#                  label='Difference', color='mediumseagreen', ax=axs[0])
make_fillbetween(difference.index, difference, label='Difference', color='mediumseagreen', ax=axs[1])
make_lineplot([difference], color_list=['g'], ax=axs[1])
axs[0].set_ylabel('Volatility (% Price)', fontsize=16)
axs[0].set_title('TYVIX with Realized (21 Days Shifted)', fontsize=16)
axs[0].legend(fontsize=14)
axs[1].legend(fontsize=14)

# [Figure 15] Credit VIX Risk Premium Distribution
[joined_index, joined_undl_rv] = \
    share_dateindex([vix.price(), 100*vix.undl_realized_vol(do_shift=True)])
vix_diff = joined_index - joined_undl_rv
_, ax_prem = make_histogram(vix_diff, hist=False,
                            label='VIX', xlabel='Implied Vol Premium (%)', ylabel='Probability',
                            title='Risk Premium Distribution',
                            color_line='C0')
[joined_index, joined_undl_rv] = \
    share_dateindex([vixig.price(), 100*vixig.undl_realized_vol(do_shift=True)])
vixig_diff = joined_index - joined_undl_rv
make_histogram(vixig_diff, hist=False,
               label='VIXIG', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C1', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixhy.price(), 100*vixhy.undl_realized_vol(do_shift=True)])
vixhy_diff = joined_index - joined_undl_rv
make_histogram(vixhy_diff, hist=False,
               label='VIXHY', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C2', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixie.price(), 100*vixie.undl_realized_vol(do_shift=True)])
vixie_diff = joined_index - joined_undl_rv
make_histogram(vixie_diff, hist=False,
               label='VIXIE', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C3', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixxo.price(), 100*vixxo.undl_realized_vol(do_shift=True)])
vixxo_diff = joined_index - joined_undl_rv
make_histogram(vixxo_diff, hist=False,
               label='VIXXO', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C4', ax=ax_prem)
[joined_index, joined_undl_rv] = \
    share_dateindex([vixfs.price(), 100*vixfs.undl_realized_vol(do_shift=True)])
vixfs_diff = joined_index - joined_undl_rv
make_histogram(vixfs_diff, hist=False,
               label='VIXFS', xlabel='Implied Vol Premium (%)', ylabel='Probability',
               title='Risk Premium Distribution',
               color_line='C5', ax=ax_prem)

####

treasury_vix_data = pd.read_csv('data/cme_eod_treasury_vix.csv', index_col='Date', parse_dates=True)

####

def get_regime_data_list(intervals, data):
    return [data.loc[start:end] for start, end in intervals]

def combine_data_list(data_list):
    acc = pd.Series()
    for data in data_list:
        acc = acc.append(data.dropna())
    return acc.sort_index()

tyvix_hl, tyvix_lows, tyvix_highs = tyvix.vol_regime()
vix_hl, vix_lows, vix_highs = vix.vol_regime()

vix_tyvix_hl = pd.DataFrame({'vix': vix_hl, 'tyvix': tyvix_hl}).dropna()
high_high_days = vix_tyvix_hl[(vix_tyvix_hl['vix']=='high') & (vix_tyvix_hl['tyvix']=='high')].index
low_low_days = vix_tyvix_hl[(vix_tyvix_hl['vix']=='low') & (vix_tyvix_hl['tyvix']=='low')].index
diffs_high = test['10YR - 2YR Yield'].diff().loc[high_high_days]
diffs_low = test['10YR - 2YR Yield'].diff().loc[low_low_days]

ten_two_list_low = get_regime_data_list(tyvix_lows, test['10YR - 2YR Yield'])
combined_ten_two_low = combine_data_list(map(lambda df: df.diff(), ten_two_list_low))
ten_two_list_high = get_regime_data_list(tyvix_highs, test['10YR - 2YR Yield'])
combined_ten_two_high = combine_data_list(map(lambda df: df.diff(), ten_two_list_high))
ttest_ind(combined_ten_two_low.loc['2014':], combined_ten_two_high.loc['2014':])

ten_two_list_low = get_regime_data_list(tyvix_lows, tenyr)
combined_ten_two_low = combine_data_list(map(lambda df: df.diff(), ten_two_list_low))
ten_two_list_high = get_regime_data_list(tyvix_highs, tenyr)
combined_ten_two_high = combine_data_list(map(lambda df: df.diff(), ten_two_list_high))
ttest_ind(combined_ten_two_low.loc['2014':], combined_ten_two_high.loc['2014':])

#### TYVIX regimes stats attempt

from scipy.stats import ttest_ind

# Recession-predictors
ten_minus_two = pd.read_csv('Y:/Research/Research1/Gaibo/S&P Webinar Figures/T10Y2Y.csv', index_col='DATE', parse_dates=True)['T10Y2Y'].dropna()
ten_minus_threemonth = pd.read_csv('Y:/Research/Research1/Gaibo/S&P Webinar Figures/T10Y3M.csv', index_col='DATE', parse_dates=True)['T10Y3M'].dropna()

# t-tests on in-regime daily differences
tyvix_hl, tyvix_lows, tyvix_highs = tyvix.vol_regime()
high_days = tyvix_hl[tyvix_hl == 'high'].index
low_days = tyvix_hl[tyvix_hl == 'low'].index
diffs_high_10_2 = ten_minus_two.diff().loc[high_days].dropna()
diffs_low_10_2 = ten_minus_two.diff().loc[low_days].dropna()
diffs_high_10_3m = ten_minus_threemonth.diff().loc[high_days].dropna()
diffs_low_10_3m = ten_minus_threemonth.diff().loc[low_days].dropna()
ttest_ind(diffs_high_10_2.loc['2018':], diffs_low_10_2.loc['2018':])
ttest_ind(diffs_high_10_3m.loc['2018':], diffs_low_10_3m.loc['2018':])

# Plot
# TYVIX with regimes
_, axleft = plt.subplots()
make_lineplot(tyvix.price(), 'TYVIX', ax=axleft)
make_regime(tyvix.vol_regime()[2], 'High Vol Regime', 'grey', 'Date', 'Index Level', 'TYVIX Vol Regimes', ax=axleft)
make_regime(tyvix.vol_regime()[1], 'Low Vol Regime', 'white', 'Date', 'Index Level', 'TYVIX Vol Regimes', ax=axleft)
axleft.set_xlabel('Date', fontsize=16)
axleft.set_ylabel('Volatility Index (%)', fontsize=16)
axleft.set_title('TYVIX Vol Regimes', fontsize=16)
# 10-2 and 10-3m
axright = axleft.twinx()
axright.plot(ten_minus_two, label='10yr - 2yr Yield', color='C1')
axright.plot(ten_minus_threemonth, label='10yr - 3month Yield', color='C2')
axright.set_ylabel('% (Annualized)', fontsize=16)
# Adjustments (zoom, framing, etc.)
axleft.set_xlim('2018-01-01', '2019-09-03')
axleft.set_ylim(3, 6.5)
axright.set_ylim(-0.75, 1.5)
axright.axhline(0, color='k', linestyle='--', label='Yield Curve Inversion')
axleft.legend(loc=2, fontsize=13)
axright.legend(loc=1, fontsize=13)

####

cdx_ig_old = bbg_data['IBOXUMAE CBBT Curncy', 'PX_LAST'].dropna()
itraxx_ie_old = bbg_data['ITRXEBE CBBT Curncy', 'PX_LAST'].dropna()
itraxx_xo_old = bbg_data['ITRXEXE CBBT Curncy', 'PX_LAST'].dropna()
itraxx_fs_old = bbg_data['ITRXESE CBBT Curncy', 'PX_LAST'].dropna()
cdx_ig_new = scaled_cds_index_data['CDX NA IG'].dropna()
itraxx_ie_new = scaled_cds_index_data['iTraxx EU Main'].dropna()
itraxx_xo_new = scaled_cds_index_data['iTraxx EU Xover'].dropna()
itraxx_fs_new = scaled_cds_index_data['iTraxx EU SenFin'].dropna()

plt.subplots()
cdx_ig_old.plot()
cdx_ig_new.plot()
plt.subplots()
itraxx_ie_old.plot()
itraxx_ie_new.plot()
plt.subplots()
itraxx_xo_old.plot()
itraxx_xo_new.plot()
plt.subplots()
itraxx_fs_old.plot()
itraxx_fs_new.plot()

##############################################################################

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(vega_volumes_df['Vega Volume'], label='Vega Volume using Given IVs', linewidth=4)
ax.plot(vega_volumes_df['Vega Volume using Backed-Out IV'], label='Vega Volume using Backed-Out IVs')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Vega Volume (MM)')
ax.set_title('{} Options Daily Vega Volume'.format(NOTE_CODE))

####

day = pd.to_datetime('2016-05-10')
cleaned_f = read_eod_file(day.strftime('%Y-%m-%d'), 'f', EOD_LOCATION)
indexed_f = (cleaned_f[cleaned_f['Total Volume'] > 0]
             .set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])
             .sort_index())  # Easier to look at
unindexed_f = indexed_f.reset_index()  # Easier to work with but still sorted

# Get corresponding underlying futures prices
opt_exps = pd.to_datetime(unindexed_f['Last Trade Date'].unique())  # Need to re-create Timestamp
undl_fut_prices = [get_fut_price(futures_data, day, exp) for exp in opt_exps]
opt_exps_undl_prices_df = \
    pd.DataFrame({'Last Trade Date': opt_exps, 'Underlying Price': undl_fut_prices})

# Calculate time to expiration
days_to_exp = (opt_exps - day).map(lambda td: td.days)
days_to_exp = days_to_exp.where(days_to_exp != 0, 0.5)  # Avoid 0 days to exp by giving it a half day
opt_exps_t_to_exps_df = \
    pd.DataFrame({'Last Trade Date': opt_exps, 'Time to Expiration': days_to_exp/365})

# Get risk-free rates
# Refresh rates or use persisted rates
if day in one_mo_rates.index:
    curr_1_mo_rate = one_mo_rates.loc[day]
if day in three_mo_rates.index:
    curr_3_mo_rate = three_mo_rates.loc[day]
use_1_mo = abs(days_to_exp - 28) < abs(days_to_exp - 91)    # Expiration closer to 1 month than 3
rates = np.zeros_like(days_to_exp, dtype=float)
rates[use_1_mo] = curr_1_mo_rate
rates[~use_1_mo] = curr_3_mo_rate
rates /= 100
opt_exps_rates_df = pd.DataFrame({'Last Trade Date': opt_exps, 'Risk-Free Rate': rates})

# Combine
combined_df = \
    (unindexed_f.merge(opt_exps_t_to_exps_df, how='left', on='Last Trade Date')
     .merge(opt_exps_undl_prices_df, how='left', on='Last Trade Date')
     .merge(opt_exps_rates_df, how='left', on='Last Trade Date'))

# Calculate our own IV to compare and to use when no IV is given
our_iv = implied_vol_b76(combined_df['Put/Call'] == 'C',
                         combined_df['Time to Expiration'], combined_df['Strike Price'],
                         combined_df['Underlying Price'], combined_df['Risk-Free Rate'],
                         combined_df['Settlement'])
combined_df['Backed-Out Implied Volatility'] = our_iv

# Run vega volume (in MM) calculation
if not combined_df['Implied Volatility'].dropna().empty:
    # There are some days where given IV inexplicably contains 0s
    no0IV_combined_df = combined_df[combined_df['Implied Volatility'] > 0]
    vega = vega_b76(no0IV_combined_df['Time to Expiration'], no0IV_combined_df['Strike Price'],
                    no0IV_combined_df['Underlying Price'], no0IV_combined_df['Risk-Free Rate'],
                    no0IV_combined_df['Implied Volatility'])
    vega_volume = (vega * no0IV_combined_df['Total Volume']).sum() * FUTURES_MULTIPLIER / ONE_MILLION
    vega_volumes_list.append(vega_volume)   # Record
else:
    vega_volumes_list.append(np.NaN)
# Alternate calculation with homemade IVs
# Ensure backed-out IV does not contain negative numbers
realIV_combined_df = combined_df[combined_df['Backed-Out Implied Volatility'] > 0]
vega_alt = vega_b76(realIV_combined_df['Time to Expiration'], realIV_combined_df['Strike Price'],
                    realIV_combined_df['Underlying Price'], realIV_combined_df['Risk-Free Rate'],
                    realIV_combined_df['Backed-Out Implied Volatility'])
vega_volume_alt = (vega_alt * realIV_combined_df['Total Volume']).sum() * FUTURES_MULTIPLIER / ONE_MILLION
vega_volumes_alt_list.append(vega_volume_alt)     # Record
# Record day
good_days_list.append(day)  # Record

###################

test = combined_df.set_index(['Last Trade Date', 'Put/Call', 'Strike Price'])
for exp in opt_exps:
    fig, ax = plt.subplots()
    test.xs(exp, level='Last Trade Date').xs('C', level='Put/Call')['Settlement'].plot(marker='o', linewidth=4)
    # test.xs(exp, level='Last Trade Date').xs('C', level='Put/Call')['Given'].plot(marker='o', linewidth=4)
    test.xs(exp, level='Last Trade Date').xs('P', level='Put/Call')['Settlement'].plot(marker='o')
    # test.xs(exp, level='Last Trade Date').xs('P', level='Put/Call')['Given'].plot(marker='o')
    ax.legend(['Call Price', 'Put Price'])
    ax.set_title('Trade Date: {}; Expiration Date: {}'.format(day.strftime('%Y-%m-%d'), exp.strftime('%Y-%m-%d')))
for exp in opt_exps:
    fig, ax = plt.subplots()
    test.xs(exp, level='Last Trade Date').xs('C', level='Put/Call')['Backed-Out Implied Volatility'].plot(marker='o', linewidth=4)
    test.xs(exp, level='Last Trade Date').xs('C', level='Put/Call')['Implied Volatility'].plot(marker='o', linewidth=4)
    test.xs(exp, level='Last Trade Date').xs('P', level='Put/Call')['Backed-Out Implied Volatility'].plot(marker='o')
    test.xs(exp, level='Last Trade Date').xs('P', level='Put/Call')['Implied Volatility'].plot(marker='o')
    ax.legend(['Call Calculated IV', 'Call Official IV', 'Put Calculated IV', 'Put Official IV'])
    ax.set_title('Trade Date: {}; Expiration Date: {}'.format(day.strftime('%Y-%m-%d'), exp.strftime('%Y-%m-%d')))

today = '2016-03-21'
exps = pd.to_datetime(['2016-03-24', '2016-04-22', '2016-05-20', '2016-06-24'])
for exp in exps:
    fig, ax = plt.subplots()
    test.xs(exp, level='Last Trade Date').xs('C', level='Put/Call')['Calculated'].plot(marker='o', linewidth=4)
    test.xs(exp, level='Last Trade Date').xs('C', level='Put/Call')['Given'].plot(marker='o', linewidth=4)
    test.xs(exp, level='Last Trade Date').xs('P', level='Put/Call')['Calculated'].plot(marker='o')
    test.xs(exp, level='Last Trade Date').xs('P', level='Put/Call')['Given'].plot(marker='o')
    ax.legend(['Call Calculated', 'Call Given', 'Put Calculated', 'Put Given'])
    ax.set_title('Trade Date: {}; Expiration Date: {}'.format(today, exp.strftime('%Y-%m-%d')))

fig, ax = plt.subplots()
make_lineplot(vix.price(), 'VIX', ax=ax)
make_regime(vix.vol_regime()[2], 'High Vol Regime', 'r', 'Date', 'Index Level', 'VIX Vol Regimes', ax=ax)
make_regime(vix.vol_regime()[1], 'Low Vol Regime', 'g', 'Date', 'Index Level', 'VIX Vol Regimes', ax=ax)

plt.figure()
prices = vix.price()
prices.plot()
window = 126
low_threshold = 0.1
high_threshold = 0.9
rolling_low = prices.rolling(window).quantile(low_threshold).dropna()
rolling_high = prices.rolling(window).quantile(high_threshold).dropna()
rolling_low.plot()
rolling_high.plot()
regime, low_list, high_list = vix.vol_regime()
for interval in low_list:
    plt.axvspan(*interval, color='C1', alpha=0.5)
for interval in high_list:
    plt.axvspan(*interval, color='C2', alpha=0.5)

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

#%% Shape up the data for use

###############################################################################
### [DAILY] Premiums and volumes
## Obtain sorted index by option ticker and no extra columns
# Transpose
prem_vol_df_clean_T = prem_vol_df_clean.T
# Reset index
prem_vol_df_clean_T_noindex = prem_vol_df_clean_T.reset_index()
# Remove column name to reorganize
prem_vol_df_clean_T_noindex.columns = prem_vol_df_clean_T_noindex.columns.rename(None)
# Rename, reindex, and sort
prem_vol_df_clean_T_reindex = \
    prem_vol_df_clean_T_noindex.rename({'ticker':'opt_ticker'}, axis=1) \
                               .set_index('opt_ticker') \
                               .sort_index()

## Option ticker-indexed daily volumes
volume_df_indexed = \
    prem_vol_df_clean_T_reindex[prem_vol_df_clean_T_reindex['field']=='PX_VOLUME'] \
    .drop(columns='field')
volume_df_indexed = volume_df_indexed.fillna(value=0)   # Replace NaN with 0
## Option ticker-indexed daily premiums
prem_df_indexed = \
    prem_vol_df_clean_T_reindex[prem_vol_df_clean_T_reindex['field']=='PX_LAST'] \
    .drop(columns='field')
## Option ticker-index that returned usable information
chosen_index = volume_df_indexed.index
################################################################################

################################################################################
### [DAILY] Underlying prices
## Clean underlying ticker mapping by removing and renaming columns
undl_ticker_df_clean = \
    undl_ticker_df.drop(columns='field') \
                  .rename({'ticker':'opt_ticker', 'value':'undl_ticker'}, axis=1)
## Obtain price DF to merge into undl_ticker_df_clean
## NOTE: ensure reset_index() is always used as intended
# Make copy to retain clean original BBG pull
undl_price_df_copy = undl_price_df.copy()
# Remove unnecessary multiindex
undl_price_df_copy.columns = undl_price_df.columns.get_level_values(0).rename(None)
# Transpose
undl_price_df_T = undl_price_df_copy.T
# Remove column name to reorganize
undl_price_df_T.columns = undl_price_df_T.columns.rename(None)
# Rename index
undl_price_df_T.index = undl_price_df_T.index.rename('undl_ticker')
# Match formatting of undl_ticker_df_clean by resetting index
undl_price_df_clean = undl_price_df_T.reset_index()

## Option ticker-indexed daily underlying prices (from merging)
undl_price_df_indexed = \
    undl_ticker_df_clean.merge(undl_price_df_clean) \
    .drop(columns='undl_ticker') \
    .set_index('opt_ticker') \
    .sort_index()
################################################################################

################################################################################
### Expiration date, financing rate, and strike
## Clean by indexing by option ticker and isolating the two types of data
# Rename and index
exp_strike_rate_df_indexed = exp_rate_df.rename({'ticker':'opt_ticker'}, axis=1) \
                                               .set_index('opt_ticker')
# Isolate expiration date and financing rate
exp_df_indexed = \
    exp_strike_rate_df_indexed.loc[exp_strike_rate_df_indexed['field']=='OPT_EXPIRE_DT',
                                   'value']
rate_df_indexed = \
    exp_strike_rate_df_indexed.loc[exp_strike_rate_df_indexed['field']=='OPT_FINANCE_RT',
                                   'value'].astype(float) * 0.01    # Scale
# Create strike in same format
strike_df_indexed = \
    pd.Series(exp_df_indexed.index.map(lambda s: float(s.split(' ')[1])),
              index=exp_df_indexed.index)
# Create put or call indication in same format
pc_df_indexed = \
    pd.Series(exp_df_indexed.index.map(lambda s: s.split(' ')[0][-1]),
              index=exp_df_indexed.index)

## Option ticker-indexed expiration date, strike, and financing rate
constant_df_indexed = pd.DataFrame(dict(expiration=exp_df_indexed,
                                        strike=strike_df_indexed,
                                        putcall=pc_df_indexed,
                                        rate=rate_df_indexed)).sort_index()
################################################################################
