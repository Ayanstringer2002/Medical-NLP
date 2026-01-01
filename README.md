# 🏥 Medical NLP Pipeline

This project implements an AI-powered NLP pipeline for:
- Medical transcription analysis
- Clinical summarization
- Sentiment & intent detection
- SOAP note generation

---

## 🚀 Features

- Medical Named Entity Recognition (NER): Extract Symptoms, Diagnosis, Treatment, Prognosis from text.
- Transformer-based Summarization: Generate concise summaries of medical transcripts using BART.
- Sentiment & Intent Analysis: Understand patient sentiment (e.g., anxious, reassured) and intent from their notes.
- Keyword Extraction: Identify important medical terms using KeyBERT.
- SOAP Note Generation: Automatically create structured Subjective, Objective, Assessment, Plan (SOAP) notes.
- Interactive User Input: Enter medical notes or transcripts at runtime for real-time analysis.

---

## 🛠 Tech Stack

- Python 3.9+
- spaCy
- HuggingFace Transformers
- PyTorch
- KeyBERT
- scikit-learn

---

## ⚙️ Setup
- Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## How to Use
- Input your medical text when prompted. For example:
```bash
Enter the patient's medical transcript or notes:
I was involved in a car accident last September. I hit my head and had severe neck and back pain for four weeks. Doctors diagnosed whiplash injury and advised physiotherapy. I completed ten physiotherapy sessions. Now I only experience occasional back pain.
```
## pipeline Output :
- Structured Medical Report (JSON format)
- Medical Summary
- Keywords extracted
- Patient Sentiment & Intent
- Generated SOAP Note

## Sample Output :
```bash
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["back pain", "neck pain"],
  "Diagnosis": ["whiplash"],
  "Treatment": ["physiotherapy"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months"
}
```
- Summary :
```bash
"Patient involved in a car accident, experienced severe neck and back pain for four weeks, diagnosed with whiplash injury, completed physiotherapy, now reports occasional back pain."
```
- Sentiment: Reassured
- Intent: Reporting symptoms

## Notes :
- The pipeline is rule-based for medical entity extraction but uses transformers for summarization and sentiment.
- Works best with clear and concise medical transcripts.
- Always verify outputs manually if used in a research context.








