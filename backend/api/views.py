from django.http import JsonResponse
from requests.exceptions import RequestException
from rest_framework.views import APIView
from rest_framework.response import Response
import cloudscraper


# Create your views here.

def get(self, request):
    """
    Fetch and return stock data as JSON.
    """
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
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        return Response(data, status=200)  # Return JSON response

    except RequestException as e:
        return Response({"error": str(e)}, status=500)
    


