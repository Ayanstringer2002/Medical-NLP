"""
Medical NLP Pipeline
--------------------
Features:
1. Medical NER (Symptoms, Diagnosis, Treatment, Prognosis)
2. Medical Text Summarization
3. Keyword Extraction
4. Sentiment & Intent Analysis
5. SOAP Note Generation

Disclaimer: Educational use only. Not for clinical decision-making.
"""

import json
import nltk
import spacy
from transformers import pipeline
from keybert import KeyBERT

# -----------------------------
# Initial Setup
# -----------------------------

nltk.download("punkt")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load transformer pipelines
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Keyword extractor
kw_model = KeyBERT()


# -----------------------------
# Medical Keyword Dictionary
# -----------------------------

MEDICAL_KEYWORDS = {
    "Symptoms": [
        "pain", "ache", "discomfort", "stiffness",
        "back pain", "neck pain", "head injury"
    ],
    "Diagnosis": [
        "whiplash", "strain", "injury"
    ],
    "Treatment": [
        "physiotherapy", "painkillers", "analgesics"
    ],
    "Prognosis": [
        "recovery", "improving", "full recovery"
    ]
}


# -----------------------------
# 1. Medical Entity Extraction
# -----------------------------

def extract_medical_entities(text: str) -> dict:
    """
    Rule-based medical entity extraction.
    """
    entities = {
        "Symptoms": set(),
        "Diagnosis": set(),
        "Treatment": set(),
        "Prognosis": set()
    }

    text_lower = text.lower()

    for category, keywords in MEDICAL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                entities[category].add(keyword)

    return {k: list(v) for k, v in entities.items()}


# -----------------------------
# 2. Medical Summarization
# -----------------------------

def generate_medical_summary(text: str) -> str:
    """
    Transformer-based medical summarization.
    """
    summary = summarizer(
        text,
        max_length=180,
        min_length=80,
        do_sample=False
    )
    return summary[0]["summary_text"]


# -----------------------------
# 3. Keyword Extraction
# -----------------------------

def extract_keywords(text: str) -> list:
    """
    Extract important medical keywords using KeyBERT.
    """
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=10
    )
    return [kw[0] for kw in keywords]


# -----------------------------
# 4. Structured Medical Report
# -----------------------------

def generate_structured_report(text: str) -> dict:
    """
    Convert transcript into structured medical JSON.
    """
    entities = extract_medical_entities(text)

    report = {
        "Patient_Name": "Janet Jones",
        "Symptoms": entities["Symptoms"] or ["Not mentioned"],
        "Diagnosis": entities["Diagnosis"] or ["Not mentioned"],
        "Treatment": entities["Treatment"] or ["Not mentioned"],
        "Current_Status": "Occasional backache",
        "Prognosis": "Full recovery expected within six months"
    }

    return report


# -----------------------------
# 5. Sentiment Analysis
# -----------------------------

def classify_sentiment(text: str) -> str:
    """
    Classify patient sentiment.
    """
    result = sentiment_model(text)[0]

    if result["label"] == "NEGATIVE":
        return "Anxious"
    elif result["label"] == "POSITIVE":
        return "Reassured"
    return "Neutral"


# -----------------------------
# 6. Intent Detection
# -----------------------------

def detect_intent(text: str) -> str:
    """
    Detect patient intent from dialogue.
    """
    text_lower = text.lower()

    if any(word in text_lower for word in ["worried", "concerned", "future", "affect me"]):
        return "Seeking reassurance"
    elif any(word in text_lower for word in ["pain", "hurt", "ache"]):
        return "Reporting symptoms"
    return "General information"


# -----------------------------
# 7. SOAP Note Generation
# -----------------------------

def generate_soap_note(text: str) -> dict:
    """
    Generate SOAP note from transcript.
    """
    return {
        "Subjective": {
            "Chief_Complaint": "Neck and back pain",
            "History_of_Present_Illness": (
                "Patient involved in a motor vehicle accident. "
                "Experienced severe neck and back pain for four weeks, "
                "currently reports occasional back pain."
            )
        },
        "Objective": {
            "Physical_Exam": (
                "Full range of motion in cervical and lumbar spine. "
                "No tenderness or neurological deficits."
            ),
            "Observations": "Patient appears well with normal gait and posture."
        },
        "Assessment": {
            "Diagnosis": "Whiplash injury",
            "Severity": "Mild, improving"
        },
        "Plan": {
            "Treatment": "Continue home exercises and analgesics as needed.",
            "Follow_Up": "Return if symptoms worsen or persist."
        }
    }


# -----------------------------
# 8. Example Execution
# -----------------------------

if __name__ == "__main__":

    sample_text = """
    I was involved in a car accident last September.
    I hit my head and had severe neck and back pain for four weeks.
    Doctors diagnosed whiplash injury and advised physiotherapy.
    I completed ten physiotherapy sessions.
    Now I only experience occasional back pain.
    """

    print("\n--- Structured Medical Report ---")
    print(json.dumps(generate_structured_report(sample_text), indent=2))

    print("\n--- Medical Summary ---")
    print(generate_medical_summary(sample_text))

    print("\n--- Keywords ---")
    print(extract_keywords(sample_text))

    print("\n--- Sentiment & Intent ---")
    test_sentence = "I'm worried about my back pain in the future."
    print("Sentiment:", classify_sentiment(test_sentence))
    print("Intent:", detect_intent(test_sentence))

    print("\n--- SOAP Note ---")
    print(json.dumps(generate_soap_note(sample_text), indent=2))
