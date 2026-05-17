from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import MODEL_NAME, LABELS, MAX_LENGTH


def load_model(model_name: str = MODEL_NAME):
    """Load tokenizer and pretrained sequence classification model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict_sentiment(text: str, tokenizer, model) -> Dict[str, float | str]:
    """
    Predict sentiment for one text and return label plus class probabilities.
    Label mapping follows the original notebook:
    0 = Negative, 1 = Neutral, 2 = Positive
    """
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

    pred_idx = int(torch.argmax(torch.tensor(probs)).item())

    return {
        "Predicted_Sentiment": LABELS[pred_idx],
        "Negative_Probability": round(float(probs[0]), 6),
        "Neutral_Probability": round(float(probs[1]), 6),
        "Positive_Probability": round(float(probs[2]), 6),
        "Confidence": round(float(max(probs)), 6),
    }


def predict_many(texts: List[str], tokenizer, model) -> List[Dict[str, float | str]]:
    """Predict sentiment for a list of texts."""
    return [predict_sentiment(text, tokenizer, model) for text in texts]
