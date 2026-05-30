# Endpoints Documentation

## 1. Example of Endpoint for Text Classification

### Description

This endpoint allows users to classify a list of texts and receive predictions with associated labels and scores.

### Endpoint

```
POST /classification/texts
```

### Request Body

```json
{
  "texts": [
    "I'm feeling so healthy today!",
    "My day was not good, but not bad",
    "I hate when people do this"
  ]
}
```

### Response Body

```json
{
  "texts": [
    {
      "text": "I'm feeling so healthy today!",
      "label": "positive",
      "score": 0.9869547486305237,
      "metadata": {
        "scores": {
          "positive": 0.9869547486305237,
          "neutral": 0.009508736431598663,
          "negative": 0.0035365649964660406
        }
      }
    },
    {
      "text": "My day was not good, but not bad",
      "label": "negative",
      "score": 0.663978636264801,
      "metadata": {
        "scores": {
          "negative": 0.663978636264801,
          "neutral": 0.23965424299240112,
          "positive": 0.09636714309453964
        }
      }
    },
    {
      "text": "I hate when people do this",
      "label": "negative",
      "score": 0.9196633696556091,
      "metadata": {
        "scores": {
          "negative": 0.9196633696556091,
          "neutral": 0.07298988103866577,
          "positive": 0.007346796337515116
        }
      }
    }
  ],
  "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

### Notes

- The `texts` field in the request body should contain a list of strings representing the texts to be classified.
- The response contains a list of dictionaries, each representing a text along with its predicted label, score, and metadata.
- The `model_id` field in the response indicates the identifier of the model used for classification.
