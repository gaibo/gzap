from requests_html import HTMLSession
import pandas as pd
import time
import random

if __name__ == '__main__':
    # Prompt for user input
    raw_input = \
        input("Enter the Nike product link to monitor for stock...\n"
              "     - Press <Enter> to start with a default example\n"
              "...>> ")
    if raw_input == '':
        NIKE_LINK = "https://www.nike.com/t/air-max-90-se-mens-shoes-sXJXK0/DV2614-100"     # Air Max 90 "Moving Co"
    else:
        NIKE_LINK = raw_input

    # Create requests_html session
    # TODO: I prefer the "going down, going up, going sideways" recursive tree flexibility of BeautifulSoup;
    #       I know requests_html probably has the functionality, I just can't find it;
    #       I should rewrite this with Selenium+BS as a future feature
    session = HTMLSession()
    COLLECTION_DATAFRAME = pd.DataFrame()   # Initialize storage DF

    # Repeatedly scrape Nike for sizes
    while True:
        # Send request, get response
        nike_response = session.get(NIKE_LINK)
        # Render Nike's extremely complicated JavaScript
        # NOTE: stock cannot be ascertained without rendering JavaScript
        # TODO: make render retry like 5 times with lower timeout, because this is really unstable
        nike_response.html.render(timeout=20)    # Nike takes between default 8 and 15 seconds
        curr_time = pd.Timestamp('now')     # Time of rendering, i.e. when site "loads"

        # Find shoe's "select sizes" section and check which sizes are out-of-stock
        # NOTE: these CSS selectors were tailored specifically from inspecting Nike in browser,
        #       so don't read too much into "mt2-sm css hzulvp", "div input", etc.
        # TODO: you would have to click the enabled (in-stock) sizes to trigger the JavaScript that tells
        #       you whether something is low-stock; that could be a fun future feature, with Selenium
        select_sizes_section_search = nike_response.html.find("div.mt2-sm.css-hzulvp")
        select_sizes_section = select_sizes_section_search[0]
        disabled_sizes = select_sizes_section.find(":disabled")
        enabled_sizes = select_sizes_section.find(":enabled")
        try:
            out_of_stock = [float(s.attrs['value'].split(':')[-1]) for s in disabled_sizes]     # Try convert '9' to 9
            in_stock = [float(s.attrs['value'].split(':')[-1]) for s in enabled_sizes]
            if COLLECTION_DATAFRAME.empty:
                # Re-initialize column order with sorted numeric sizes
                COLLECTION_DATAFRAME = pd.DataFrame(columns=sorted(out_of_stock + in_stock))
        except ValueError:
            out_of_stock = [s.attrs['value'].split(':')[-1] for s in disabled_sizes]    # Keep 'M' as string
            in_stock = [s.attrs['value'].split(':')[-1] for s in enabled_sizes]
            if COLLECTION_DATAFRAME.empty:
                # Re-initialize hard-coded column order to non-numeric sizes (may include non-existent sizes)
                COLLECTION_DATAFRAME = pd.DataFrame(columns=['XS', 'S', 'M', 'L', 'XL', '2XL', '3XL'])

        # Record in DataFrame
        COLLECTION_DATAFRAME.loc[curr_time, out_of_stock] = False
        COLLECTION_DATAFRAME.loc[curr_time, in_stock] = True
        print(COLLECTION_DATAFRAME, end='\n\n')

        # Sleep for approximately 4 hours (+- 15 minute "human error" range)
        # TODO: sleep does not play well with PyCharm's console's response to Ctrl+C; either way, I'd like to
        #       do cleanup and retain the DataFrame. I'll rewrite this using threading
        time.sleep(4*60*60 + random.uniform(-15*60, 15*60))
