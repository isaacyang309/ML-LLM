import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

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

    def fetch_relevant_crypto_headlines(self) -> list:
        """Fetches recent crypto news and filters strictly for BTC/ETH relevance."""
        keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto']
        relevant_headlines = []
        
        # Safely pull the key from the .env file!
        API_KEY = os.getenv("CCDATA_API_KEY") 
        
        if not API_KEY:
            logger.error("[API Error] CCDATA_API_KEY not found. Did you set up your .env file?")
            return []
        
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            
            # Inject the securely loaded API key
            headers = {
                "authorization": f"Apikey {API_KEY}"
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status() 
            data = response.json()
            
            if "Data" in data and isinstance(data["Data"], list):
                articles = data["Data"]
                
                for article in articles:
                    headline = article.get("title", "")
                    body = article.get("body", "")
                    tags = article.get("tags", "")
                    
                    full_search_text = f"{headline} {body} {tags}".lower()
                    
                    if any(keyword in full_search_text for keyword in keywords):
                        relevant_headlines.append(headline)
                        if len(relevant_headlines) >= 3:
                            break
                            
                return relevant_headlines
            else:
                logger.warning(f"[API Warning] Unexpected response format: {data}")
                return [] 
                
        except Exception as e:
            logger.error(f"Failed to fetch live news: {e}")
            return []
        
    def get_crypto_sentiment(self, headlines=None) -> float:
        """
        Takes a list of news headlines, scores them, and returns the average sentiment.
        Returns: 1.0 (Bullish), -1.0 (Bearish), or 0.0 (Neutral)
        """
        if not self.analyzer:
            return 0.0
            
        # Handle string input for backwards compatibility
        if isinstance(headlines, str):
            headlines = [headlines]
            
        # If no headlines provided, fetch live relevant ones
        if not headlines:
            headlines = self.fetch_relevant_crypto_headlines()
            
        # Fail-safe: If no relevant news is found, stay neutral so we don't block the technical strategy
        if not headlines:
            logger.info("[FinBERT] No relevant BTC/ETH news found right now. Defaulting to Neutral (0.0).")
            return 0.0
            
        total_score = 0.0
        valid_scores = 0
        
        try:
            for headline in headlines:
                logger.info(f"[FinBERT] Analyzing: '{headline}'")
                result = self.analyzer(headline)[0]
                label = result['label']
                
                if label == 'positive':
                    total_score += 1.0
                    valid_scores += 1
                elif label == 'negative':
                    total_score -= 1.0
                    valid_scores += 1
                else:
                    valid_scores += 1 # Neutral adds 0.0, but still counts as a valid read
                    
            if valid_scores == 0:
                return 0.0
                
            average_score = total_score / valid_scores
            logger.info(f"[FinBERT] Average Sentiment: {average_score:.2f} (Based on {valid_scores} relevant articles).")
            return average_score
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0

# Quick Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = SentimentAnalyzer()
    score = llm.get_crypto_sentiment()
    print(f"Final Output Score: {score}")