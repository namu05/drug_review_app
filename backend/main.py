import re
import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from transformers import pipeline
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

nlp = spacy.load("en_core_web_sm")
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
kw_model = KeyBERT("all-MiniLM-L6-v2")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

TOPICS = [
    "drug side effect", "drug name",
    "drug packaging", "drug efficacy", "drug form", "drug dosage"
]

CUSTOM_STOPWORDS = {
    "drug", "patient", "symptom", "occur", "taking"
}

FORCED_TOPIC_MAP = {
    "tablet": "drug form",
    "capsule": "drug form",
    "injection": "drug form",
    "syrup": "drug form",
    "ointment": "drug form",
}

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

def extract_drug_name(text: str) -> List[str]:
    doc = nlp(text)
    candidates = [ent.text for ent in doc.ents if ent.label_ in {"PRODUCT"}]
    return candidates

def extract_packaging_phrases(sentence: str) -> List[str]:
    patterns = [
        r"\\d+\\s?pk",
        r"\\d+\\s?fl\\s?oz",
        r"\\bbottles?\\b",
        r"\\bcans?\\b",
        r"\\d+\\s?mg\\b",
        r"\\btablets?\\b",
        r"\\bcapsules?\\b"
    ]
    phrases = []
    for pattern in patterns:
        phrases.extend(re.findall(pattern, sentence.lower()))
    return list(set(phrases))

def get_predicted_topics(sentence: str, threshold: float = 0.4) -> List[str]:
    result = topic_classifier(
        sentence, TOPICS, multi_label=True, hypothesis_template="This text is about {}."
    )
    return [label for label, score in zip(result["labels"], result["scores"]) if score >= threshold]

def get_keywords(sentence: str, top_n: int = 15) -> List[str]:
    keywords = kw_model.extract_keywords(
        sentence, keyphrase_ngram_range=(1, 1), stop_words='english',
        use_mmr=True, diversity=0.7, top_n=top_n
    )
    return [word for word, _ in keywords if word.lower() not in CUSTOM_STOPWORDS and len(word) > 3]

def map_keywords_to_topics(predicted_topics: List[str], keywords: List[str], threshold: float = 0.25) -> Dict[str, List[str]]:
    topic_keyword_map = {topic: [] for topic in predicted_topics}

    for kw in keywords:
        forced_topic = FORCED_TOPIC_MAP.get(kw.lower())
        if forced_topic and forced_topic in topic_keyword_map:
            topic_keyword_map[forced_topic].append(kw)

    topic_embeddings = sentence_model.encode(predicted_topics, convert_to_tensor=True)
    keyword_embeddings = sentence_model.encode(keywords, convert_to_tensor=True)
    sim = util.cos_sim(keyword_embeddings, topic_embeddings)

    for i, kw in enumerate(keywords):
        if any(kw in topic_keyword_map[tp] for tp in topic_keyword_map):
            continue
        topic_idx = sim[i].argmax().item()
        score = sim[i][topic_idx].item()
        if score >= threshold:
            topic_keyword_map[predicted_topics[topic_idx]].append(kw)

    for topic in topic_keyword_map:
        topic_keyword_map[topic] = sorted(set(topic_keyword_map[topic]))

    return topic_keyword_map

def extract_topics_and_keywords(text: str) -> Dict[str, List[str]]:
    text = preprocess(text)
    keywords = get_keywords(text)
    keywords.extend(extract_packaging_phrases(text))
    keywords.extend(extract_drug_name(text))
    keywords = list(set(keywords))
    predicted_topics = get_predicted_topics(text)
    return map_keywords_to_topics(predicted_topics, keywords)

class ReviewInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_review(input: ReviewInput):
    result = extract_topics_and_keywords(input.text)
    return {"topics_keywords": result}

@app.get("/")
def root():
    return {"message": "Drug Review API is live!"}