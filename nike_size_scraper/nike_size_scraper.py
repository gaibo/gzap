from requests_html import HTMLSession
import pandas as pd
import random
from threading import Event
import signal
import ctypes
from types import FrameType

DEFAULT_PRODUCT_LINK = "https://www.nike.com/t/air-trainer-1-mens-shoes-0vx2ft/DM0521-100"  # Air Trainer "Chlorophyll"
# DEFAULT_PRODUCT_LINK = "https://www.nike.com/t/air-max-90-se-mens-shoes-sXJXK0/DV2614-100"    # Air Max 90 "Moving Co"
DEFAULT_REFRESH_MINUTES = 120
REFRESH_DANGER_BOUND = 10   # Don't want to get banned querying too often


def demand_input() -> tuple[str, float]:
    """ Helper function to handle user input cases:
        1) > 2 arguments: loop and demand inputs again
        2) 2 arguments: parse as Nike link and refresh rate
        3) 1 argument: parse as Nike link, use default refresh
        4) 0 arguments: use defaults
        NOTE: refresh rate inputs below REFRESH_DANGER_BOUND will be rejected
    :return: tuple of (Nike website link, refresh rate in minutes)
    """
    input_raw = \
        input("Enter the Nike product link to monitor for stock...\n"
              "     - Optional: follow it with whitespace and a number to serve as refresh rate in minutes\n"
              "     - e.g.: https://www.nike.com/t/air-trainer-1-mens-shoes-0vx2ft/DM0521-100 120\n"
              "     - Press <Enter> to start with above default example\n"
              "...>> ")
    input_list = input_raw.split()
    if len(input_list) > 2:
        print(f"ERROR: {len(input_list)} inputs detected instead of 1 (link) or 2 (link + refresh)", end='\n\n')
        return demand_input()   # Restart until appropriate inputs received... it's called "demand_input"
    elif len(input_list) == 2:
        nike_link = input_list[0]
        refresh_mins = float(input_list[1])     # Chance to raise ValueError
    elif len(input_list) == 1:
        nike_link = input_list[0]
        refresh_mins = DEFAULT_REFRESH_MINUTES
    else:
        # Case of just pressing <Enter>, i.e. '' empty string
        nike_link, refresh_mins = DEFAULT_PRODUCT_LINK, DEFAULT_REFRESH_MINUTES
        print(f"Default product: Air Trainer 1 \"Chlorophyll\" - {DEFAULT_PRODUCT_LINK}; "
              f"Default refresh: {DEFAULT_REFRESH_MINUTES} minutes")
    if refresh_mins < REFRESH_DANGER_BOUND:
        print(f"Refresh rate of {refresh_mins} minutes will get both of us in trouble. Try {DEFAULT_REFRESH_MINUTES}.")
        refresh_mins = DEFAULT_REFRESH_MINUTES
    return nike_link, refresh_mins


def finish_handler(signum: signal.Signals | int, _frame: FrameType) -> None:
    # Handler function used by signal.signal() to "set" threading.Event and exit cleanly
    print(f"Interrupted by {signum}; finishing scrape...")
    FINISH_SCRAPING.set()


if __name__ == '__main__':
    input_nike_link, input_refresh_mins = demand_input()
    print(f"Querying above link every {input_refresh_mins} minutes, please wait...")

    # Create requests_html session
    # TODO: I prefer the "going down, going up, going sideways" recursive tree flexibility of BeautifulSoup;
    #       I know requests_html uses BS and probably has the functionality, I just can't find it; so I
    #       may rewrite this at some point with Selenium+BS instead
    SESSION = HTMLSession()
    # Set up collection variables
    COLLECTION_DATAFRAME = pd.DataFrame()   # Initialize storage DF
    pd.options.display.expand_frame_repr = False    # Make DataFrames print fully
    FIRST_TIME_THROUGH = True   # Get human-readable shoe info, only once

    # Repeatedly scrape Nike for sizes
    FINISH_SCRAPING = Event()   # Set up interrupting event to end scraping loop; NOTE: doesn't work on Windows...
    signal.signal(signal.SIGTERM, finish_handler)
    signal.signal(signal.SIGINT, finish_handler)  # In theory (not on Windows), program handles Ctrl+c to exit cleanly
    while not FINISH_SCRAPING.is_set():
        # Send request, get response
        nike_response = SESSION.get(input_nike_link)
        # Render Nike's extremely complicated JavaScript (this is the time-intensive part)
        # NOTE: stock cannot be ascertained without rendering JavaScript
        nike_response.html.render(retries=8, wait=3, scrolldown=3, timeout=30)  # Very finicky, maybe anti-bot
        curr_time = pd.Timestamp('now')     # Time of rendering, i.e. when site "loads
        if FIRST_TIME_THROUGH:
            # Print shoe info and add it to window title
            title = nike_response.html.find("h1#pdp_product_title")[0].text
            color_desc = nike_response.html.find("div ul li.description-preview__color-description.ncss-li")[0].text
            product = f"{title}, \"{color_desc.replace('Shown: ', '')}\""   # Nike Air Trainer 1, "White/Medium Grey"
            print(f"Product found: {product}")
            ctypes.windll.kernel32.SetConsoleTitleW(f"{product} - nike_size_scraper")   # Set window title
            FIRST_TIME_THROUGH = False

        # Find product's sizes and check which sizes are disabled/enabled
        # NOTE: these CSS selectors are tailored from inspecting Nike in browser;
        #       initially I tried to find the sizes section using class="mt2-sm css hzulvp", but found that changes;
        #       currently find input elements (each corresponding to a size) containing "skuAndSize" attribute;
        #       CSS selectors ":disabled" and ":enabled" can be used to determine stock-ness, but I just grab
        #       all sizes in original web page order to serve as collection DataFrame columns
        # TODO: you would have to click the enabled (in-stock) sizes to trigger the JavaScript that generates
        #       text revealing something is low-stock; that could be a fun future feature, with Selenium
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
                    print(f'\tIn stock: {list(diff[diff == "In-Stock"].index)}')
                print(end='\n')
        # TODO: ask for email, ask for desired size(s), send email?

        # Wait (interruptible blocking, in theory) for approximately the refresh time (+- random "human error" range)
        FINISH_SCRAPING.wait(input_refresh_mins*60
                             + random.uniform(-input_refresh_mins/8*60, input_refresh_mins/8*60))

    # Cleanup, upon signal handler "setting" threading.Event to finish above loop
    print("Closing HTMLSession...")
    SESSION.close()
    input("\nProgram exited cleanly. As of 2023-11-20, you cannot arrive at this message on Windows.\n"
          "     - Press <Enter> to close window\n"
          "...>> ")
