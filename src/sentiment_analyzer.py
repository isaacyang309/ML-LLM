import requests
import logging

# We import the pipeline, but we only load it when the class initializes
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        if not pipeline:
            logger.error("Transformers library not installed! Run: pip install transformers torch")
            self.analyzer = None
            return
            
        logger.info("Loading FinBERT LLM for sentiment analysis...")
        try:
            # Loads the free, lightweight FinBERT model 
            self.analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            logger.info("FinBERT successfully loaded!")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.analyzer = None

    def fetch_latest_crypto_headline(self) -> str:
        """Fetches the latest crypto news headline from a free API."""
        try:
            # Free CryptoCompare API (No key required for basic news)
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data.get("Message") == "News list successfully returned":
                # Get the title of the absolute newest article
                latest_article = data["Data"][0]
                headline = latest_article["title"]
                return headline
            return "Bitcoin market remains relatively stable."
        except Exception as e:
            logger.error(f"Failed to fetch live news: {e}")
            return "Bitcoin market remains relatively stable."

    def get_crypto_sentiment(self, headline: str = None) -> float:
        """
        Takes a news headline and returns a sentiment score.
        Returns: 1.0 (Bullish), -1.0 (Bearish), or 0.0 (Neutral)
        """
        if not self.analyzer:
            return 0.0
            
        # If no headline is provided, fetch the live one
        if not headline:
            headline = self.fetch_latest_crypto_headline()
            logger.info(f"[FinBERT] Reading Live Headline: '{headline}'")
            
        try:
            result = self.analyzer(headline)[0]
            label = result['label']
            
            if label == 'positive':
                return 1.0
            elif label == 'negative':
                return -1.0
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0

# Quick Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = SentimentAnalyzer()
    score = llm.get_crypto_sentiment()
    print(f"Final Output Score: {score}")