from django.http import JsonResponse
from requests.exceptions import RequestException
import cloudscraper

def stock_data(request):
    url = "https://www.nepsealpha.com/trading/1/history"
    params = {
        "fsk": "rpEzO8wdmCtGJiAY",
        "symbol": request.GET.get("symbol", ""), 
        "resolution": "1D",
        "pass": "ok"
    }
    
    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, params=params)
        response.raise_for_status()
        return JsonResponse(response.json(), status=200)

    except RequestException as e:
        return JsonResponse({"error": str(e)}, status=500)
if __name__ =='__main__':

    data = stock_data()