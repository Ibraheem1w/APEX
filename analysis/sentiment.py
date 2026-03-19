import numpy as np
import pandas as pd
 
# FinBERT is pretrained on financial text specifically
# using a general sentiment model like VADER here would be wrong —
# financial language is different enough that domain-specific models
# consistently outperform general ones on earnings calls, news, filings
 
# requires: pip install transformers torch
 
 
def load_finbert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
 
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model
 
 
def score_headline(headline, tokenizer, model):
    import torch
 
    inputs = tokenizer(
        headline,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
 
    with torch.no_grad():
        outputs = model(**inputs)
 
    probs = torch.softmax(outputs.logits, dim=1).squeeze()
 
    # finbert outputs: positive, negative, neutral
    positive = probs[0].item()
    negative = probs[1].item()
    neutral  = probs[2].item()
 
    # net sentiment score: positive - negative, ranges -1 to 1
    net_score = positive - negative
 
    return {
        'positive': round(positive, 4),
        'negative': round(negative, 4),
        'neutral':  round(neutral, 4),
        'score':    round(net_score, 4)
    }
 
 
def score_headlines(headlines, tokenizer, model):
    # batch scoring — faster than one at a time
    results = []
    for h in headlines:
        try:
            result = score_headline(h, tokenizer, model)
            results.append(result)
        except Exception:
            # malformed headline — skip rather than crash
            results.append({'positive': 0, 'negative': 0,
                            'neutral': 1, 'score': 0})
    return pd.DataFrame(results)
 
 
def fetch_sample_headlines():
    # placeholder headlines for testing the pipeline
    # in production replace with a real news API
    # e.g. NewsAPI, Alpha Vantage news, or Refinitiv
    return [
        "Federal Reserve signals potential rate cuts amid cooling inflation",
        "Tech stocks tumble as earnings disappoint Wall Street expectations",
        "Strong jobs report boosts market confidence in economic resilience",
        "Banking sector faces headwinds from rising credit losses",
        "S&P 500 reaches record high on strong corporate earnings",
        "Recession fears mount as manufacturing data weakens further",
        "Energy prices surge on supply constraints and geopolitical tension",
        "Consumer spending remains robust despite elevated interest rates"
    ]
 
 
def sentiment_signal(headlines=None):
    """
    Runs FinBERT on a list of financial headlines and returns
    an aggregated sentiment score usable as a trading signal.
 
    Positive score = bullish sentiment
    Negative score = bearish sentiment
    Near zero = neutral / mixed
    """
    tokenizer, model = load_finbert()
 
    if headlines is None:
        headlines = fetch_sample_headlines()
 
    scores = score_headlines(headlines, tokenizer, model)
 
    summary = {
        'avg_sentiment':   round(scores['score'].mean(), 4),
        'pct_positive':    round((scores['score'] > 0.1).mean(), 4),
        'pct_negative':    round((scores['score'] < -0.1).mean(), 4),
        'sentiment_vol':   round(scores['score'].std(), 4),
        'n_headlines':     len(headlines)
    }
 
    return summary, scores
 
 
def sentiment_as_feature(headlines, tokenizer, model):
    # returns a single float to plug into the forecasting feature matrix
    # positive = bullish signal, negative = bearish
    scores = score_headlines(headlines, tokenizer, model)
    return scores['score'].mean()