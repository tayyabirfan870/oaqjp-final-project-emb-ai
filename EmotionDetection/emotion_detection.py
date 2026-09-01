import json

import requests


def emotion_detector(text_to_analyze):
    """Detect emotions and return their scores and dominant emotion."""
    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    headers = {
        "grpc-metadata-mm-model-id":
            "emotion_aggregated-workflow_lang_en_stock"
    }

    response = requests.post(url, json=payload, headers=headers)

    # Convert response text into a Python dictionary
    formatted_response = json.loads(response.text)

    # Extract emotion scores
    anger_score = formatted_response["emotionPredictions"][0][
        "emotion"]["anger"
    ]
    disgust_score = formatted_response["emotionPredictions"][0][
        "emotion"]["disgust"
    ]
    fear_score = formatted_response["emotionPredictions"][0][
        "emotion"]["fear"
    ]
    joy_score = formatted_response["emotionPredictions"][0][
        "emotion"]["joy"
    ]
    sadness_score = formatted_response["emotionPredictions"][0][
        "emotion"]["sadness"
    ]

    # Store emotions and their scores in a dictionary
    emotions = {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score
    }

    # Find the emotion with the highest score
    dominant_emotion = max(emotions, key=emotions.get)

    # Return the required output
    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion
    }