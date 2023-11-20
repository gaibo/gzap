from requests_html import HTMLSession
import pandas as pd
import random
from threading import Event
from signal import SIGTERM, SIGINT, signal

if __name__ == '__main__':
    # Prompt for user input
    raw_input = \
        input("Enter the Nike product link to monitor for stock...\n"
              "     - Press <Enter> to start with a default example\n"
              "...>> ")
    if raw_input == '':
        NIKE_LINK = "https://www.nike.com/t/air-trainer-1-mens-shoes-0vx2ft/DM0521-100"     # Air Trainer "Chlorophyll"
        # NIKE_LINK = "https://www.nike.com/t/air-max-90-se-mens-shoes-sXJXK0/DV2614-100"     # Air Max 90 "Moving Co"
        print(f"Querying default (Air Trainer 1 \"Chlorophyll\" - {NIKE_LINK}), please wait...")
    else:
        NIKE_LINK = raw_input
        print("Querying above link, please wait...")

    # Create requests_html session
    # TODO: I prefer the "going down, going up, going sideways" recursive tree flexibility of BeautifulSoup;
    #       I know requests_html uses BS and probably has the functionality, I just can't find it; so I
    #       may rewrite this at some point with Selenium+BS instead
    session = HTMLSession()
    COLLECTION_DATAFRAME = pd.DataFrame()   # Initialize storage DF
    pd.options.display.expand_frame_repr = False

    # Repeatedly scrape Nike for sizes
    FINISH_SCRAPING = Event()     # Set up interrupting event to end infinite loop

    def finish_handler(signum, _frame):
        print(f"Interrupted by {signum}; finishing scraping...")
        FINISH_SCRAPING.set()

    signal(SIGTERM, finish_handler)
    signal(SIGINT, finish_handler)
    while not FINISH_SCRAPING.is_set():
        # Send request, get response
        nike_response = session.get(NIKE_LINK)
        # Render Nike's extremely complicated JavaScript (this is the time-intensive part)
        # NOTE: stock cannot be ascertained without rendering JavaScript
        nike_response.html.render(retries=8, wait=3, scrolldown=3, timeout=30)  # Very finicky, maybe anti-bot
        curr_time = pd.Timestamp('now')     # Time of rendering, i.e. when site "loads"

        # Find product's sizes and check which sizes are disabled/enabled
        # NOTE: these CSS selectors are tailored from inspecting Nike in browser;
        #       initially I tried to find the sizes section using class="mt2-sm css hzulvp", but found that changes;
        #       currently find input elements (each corresponding to a size) containing "skuAndSize" attribute;
        #       CSS selectors ":disabled" and ":enabled" can be used to determine stock, but I choose to grab
        #       all sizes in original web page order to serve as collection DataFrame columns
        # TODO: you would have to click the enabled (in-stock) sizes to trigger the JavaScript that tells
        #       you whether something is low-stock; that could be a fun future feature, with Selenium
        sizes_input_elements = nike_response.html.find("div input[name=\"skuAndSize\"]")    # List of ~17 for shoes
        all_sizes_raw = [element.attrs['value'].split(':')[-1] for element in sizes_input_elements]
        try:
            all_sizes = list(map(float, all_sizes_raw))     # If numerical sizing, convert '9.5' to 9.5
        except ValueError:
            all_sizes = all_sizes_raw   # Keep non-numerical sizing in web page order, i.e. keep 'M' as string
        out_of_stock = [s for s, e in zip(all_sizes, sizes_input_elements) if 'disabled' in e.attrs]    # ":disabled"
        in_stock = [s for s, e in zip(all_sizes, sizes_input_elements) if 'disabled' not in e.attrs]    # ":enabled"
        if COLLECTION_DATAFRAME.empty:
            # (Re-)Initialize DF with column order; works for both numeric and non-numeric sizing
            COLLECTION_DATAFRAME = pd.DataFrame(columns=all_sizes)

        # Record in DataFrame
        COLLECTION_DATAFRAME.loc[curr_time, out_of_stock] = 0
        COLLECTION_DATAFRAME.loc[curr_time, in_stock] = "In-Stock"  # Visual difference between in and out of stock
        # Alert user to stock change
        print("---- ---- ---- ----")
        print(COLLECTION_DATAFRAME)
        if len(COLLECTION_DATAFRAME) > 1:
            diff = COLLECTION_DATAFRAME.loc[curr_time, (COLLECTION_DATAFRAME.iloc[-1] != COLLECTION_DATAFRAME.iloc[-2])]
            if not diff.empty:
                print("\nALERT! Sizing stock change:")
                if (diff == 0).any():
                    print(f"\tOut of stock: {list(diff[diff == 0].index)}")
                if (diff == "In-Stock").any():
                    print(f'\tIn stock: {list(diff[diff == "In-Stock"].index)}\n')
        # TODO: ask for email, ask for desired size(s), send email?

        # Wait for approximately 2 hours (+- 15 minute "human error" range)
        FINISH_SCRAPING.wait(2*60*60 + random.uniform(-15*60, 15*60))

    # Cleanup
    print("Closing HTMLSession...")
    session.close()
