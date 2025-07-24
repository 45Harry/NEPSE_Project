import cloudscraper

def stock_data(symbol):
    url = "https://www.nepsealpha.com/trading/1/history"
    params = {
        "fsk": "rpEzO8wdmCtGJiAY",
        "symbol": symbol, 
        "resolution": "1D",
        "pass": "ok"
    }
  
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url, params=params)
    response.raise_for_status()
    return response.json()

if __name__ =='__main__':
    data = stock_data("NEPSE")
   # with open 
    print(data)
