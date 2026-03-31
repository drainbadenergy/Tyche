from binance.client import Client
# Replace with your actual keys (or use environment variables)
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

# Use testnet=True for the demo so no real money is at risk
client = Client(API_KEY, API_SECRET, testnet=True)

def get_live_prices(assets):
    try:
        prices = {}
        tickers = client.get_all_tickers()
        ticker_map = {t['symbol']: t['price'] for t in tickers}
        for a in assets:
            prices[a] = float(ticker_map.get(a, 0))
        return prices
    except:
        return None