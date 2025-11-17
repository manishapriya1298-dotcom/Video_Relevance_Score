
#updated version 

import os
import urllib.parse
from typing import List, Dict, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from transformers import pipeline
from googleapiclient.discovery import build

# --------- CONFIG ---------
PROMO_KEYWORDS = [
    "sponsor", "brought to you by", "promotion", "offer", "visit",
    "subscribe", "discount", "coupon", "sign up", "ad", "advertisement"
]
SIM_THRESHOLD = 0.5   # similarity threshold for relevance
SEGMENT_WORDS = 30    # shorter segments improve granularity

# Initialize models once
model = SentenceTransformer("all-MiniLM-L6-v2") 
kw_model = KeyBERT()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# --------- UTILS ---------
def parse_video_id(video_url: str) -> Optional[str]:
    """Extract YouTube video ID from common URL patterns."""
    try:
        parsed = urllib.parse.urlparse(video_url)
        if parsed.hostname in ["www.youtube.com", "youtube.com"]:
            qs = urllib.parse.parse_qs(parsed.query)
            return qs.get("v", [None])[0]
        if parsed.hostname in ["youtu.be"]:
            return parsed.path.strip("/") or None
    except Exception:
        return None
    return None

def get_sample_transcript() -> str:
    return (
        "Welcome to AI in Education. Today, we cover modern teaching tools. "
        "This video is sponsored by LearnTech. Neural networks help in adaptive testing. "
        "Visit LearnTech for the best online courses. In conclusion, technology shapes the future of learning."
    )

# --------- TRANSCRIPTION ---------
def get_transcript_youtube(video_url: str) -> Optional[str]:
    """Fetch auto captions if available using youtube-transcript-api (no API key required)."""
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

    video_id = parse_video_id(video_url)
    if not video_id:
        return None

    try:
        # Try English first, then fallback to Hindi or any available language
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'hi'])
        return " ".join(seg["text"] for seg in transcript_data)
    except (TranscriptsDisabled, NoTranscriptFound):
        # Fallback to sample transcript if captions are not available
        return None
    except Exception:
        return None

def get_transcript_whisper_api(file_path: str) -> Optional[str]:
    """Transcribe using OpenAI Whisper API (requires OPENAI_API_KEY)."""
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set. Please set the environment variable.")
        return None
    openai.api_key = api_key

    try:
        with open(file_path, "rb") as f:
            transcript = openai.Audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text
    except Exception as e:
        st.error(f"Whisper API transcription failed: {e}")
        return None

# --------- METADATA ---------
def get_youtube_metadata(video_url: str) -> Optional[Dict]:
    """Fetch title, description, channel info using YouTube Data API v3 (requires YOUTUBE_API_KEY)."""
    

    api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        return None

    video_id = parse_video_id(video_url)
    if not video_id:
        return None

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        resp = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        ).execute()

        if not resp.get("items"):
            return None

        snippet = resp["items"][0]["snippet"]
        metadata = {
    "title": snippet.get("title"),
    "description": snippet.get("description"),
    "channel_title": snippet.get("channelTitle"),
    "publish_date": snippet.get("publishedAt"),
}
        return metadata

        
    except Exception:
        return None

# --------- SEMANTIC RELEVANCE ---------
def split_transcript(transcript: str, max_length: int = SEGMENT_WORDS) -> List[str]:
    words = transcript.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

def semantic_scores(title: str, description: str, segments: List[str], model) -> List[float]:
    context = f"{title or ''} {description or ''}".strip()
    context_emb = model.encode(context, convert_to_tensor=True)
    seg_embs = model.encode(segments, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(seg_embs, context_emb).cpu().numpy().flatten()
    return [float(s) for s in sims]

def label_segments(segments: List[str], scores: List[float], promo_keywords: Optional[List[str]] = None) -> List[Dict]:
    """Auto-label segments as promotional, irrelevant, or relevant."""
    promo_keywords = promo_keywords or PROMO_KEYWORDS
    labels = []
    for i, seg in enumerate(segments):
        seg_l = seg.lower()
        if any(k in seg_l for k in promo_keywords):
            seg_type = "promotional"
        elif scores[i] < SIM_THRESHOLD:
            seg_type = "irrelevant"
        else:
            seg_type = "relevant"
        labels.append({
            "index": i,
            "segment": seg,
            "score": round(scores[i], 3),
            "type": seg_type
        })
    return labels

def compute_relevance_score(segment_labels: List[Dict]) -> float:
    total = len(segment_labels)
    if total == 0:
        return 0.0
    relevant = sum(1 for d in segment_labels if d["type"] == "relevant")
    return round(100.0 * relevant / total, 1)

# --------- DYNAMIC KEYWORDS & TOPIC TAGS ---------
def auto_keywords(transcript: str, top_n: int = 5) -> List[str]:
    """Extract top keywords dynamically from transcript using KeyBERT."""
    keywords = kw_model.extract_keywords(transcript, top_n=top_n)
    return [kw for kw, score in keywords] or ["Other"]

def auto_classify(transcript: str, candidate_labels: List[str]) -> str:
    """Classify transcript into one of the candidate labels dynamically."""
    result = classifier(transcript, candidate_labels)
    return result["labels"][0]  # highest scoring label

def topic_tags(transcript: str, candidate_labels: Optional[List[str]] = None) -> Dict:
    """Dynamically extract keywords and classify transcript into a category."""
    keywords = auto_keywords(transcript, top_n=5)
    if candidate_labels:
        category = auto_classify(transcript, candidate_labels)
    else:
        category = "Uncategorized"
    return {
        "keywords": keywords,
        "category": category
    }

# --------- RESULTS ---------
def explanation_for_results(labels: List[Dict], title: str, score: float, transcript: str) -> str:
    promo = sum(1 for d in labels if d["type"] == "promotional")
    irrelev = sum(1 for d in labels if d["type"] == "irrelevant")
    tags_info = topic_tags(
        transcript,
        candidate_labels=["Tech", "Education", "Entertainment", "Cooking", "Sports", "Politics", "Travel"]
    )

    msg = f"Content strongly matches the title '{title}' - {score}% relevant."
    if promo > 0:
        msg += f" {promo} section(s) promote a product or service."
    if irrelev > 0:
        msg += f" {irrelev} section(s) were off-topic."
    msg += f" Auto-detected keywords: {', '.join(tags_info['keywords'])}. Category: {tags_info['category']}."
    return msg   # ✅ just return the string

# --------- STREAMLIT APP ---------
st.title("🎥 Video Semantic Relevance & Topic Analyzer")

video_url = st.text_input("Enter YouTube video URL:")

if video_url:
    transcript = get_transcript_youtube(video_url)
    metadata = get_youtube_metadata(video_url)

    if metadata:
        title = metadata.get("title", "Video")
        description = metadata.get("description", "")

        # ✅ fallback: use description if transcript missing
        if not transcript:
            transcript = description

        if transcript:
            st.subheader("Transcript Preview")
            st.write(transcript[:500] + "...")

            # Split transcript into segments
            segments = split_transcript(transcript)
            scores = semantic_scores(title, description, segments, model)
            labels = label_segments(segments, scores)
            relevance_score = compute_relevance_score(labels)

            # Generate explanation
            explanation = explanation_for_results(labels, title, relevance_score, transcript)

            st.subheader("Prediction Results")
            st.write(explanation)

            # Show table of segments
            df = pd.DataFrame(labels)
            df["flag"] = df["type"].apply(lambda x: "⚠️" if x in ["promotional", "irrelevant"] else "")
            st.dataframe(df)

            # Plot relevance scores (bar chart scaled 0–100)
            fig = px.bar(
                df, x="index", y="score", color="type",
                title="Segment Relevance Scores (0–100)",
                labels={"score": "Relevance (%)"}
            )
            st.plotly_chart(fig)

            # Overall relevance gauge
            st.subheader("Overall Relevance Score")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=relevance_score,
                title={'text': "Relevance (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig_gauge)

            # Relevance heatmap
            st.subheader("Relevance Heatmap")
            heatmap_data = pd.DataFrame({
                "timestamp": df["index"] * SEGMENT_WORDS,
                "score": [round(s * 100, 1) for s in scores],  # ✅ fixed
                "type": df["type"]
            })
            fig_heatmap = px.imshow(
                [heatmap_data["score"].values],
                labels=dict(x="Segment Index", y="Relevance", color="Score (%)"),
                x=heatmap_data["timestamp"],
                aspect="auto",
                color_continuous_scale="Viridis",
                zmin=0, zmax=100
            )
            st.plotly_chart(fig_heatmap)

            # Download flagged segments
            flagged_df = df[df["type"].isin(["promotional", "irrelevant"])]
            csv = flagged_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Flagged Segments", csv, "flagged_segments.csv", "text/csv")

        else:
            st.error("No transcript or description available for this video.")
    else:
        st.error("Metadata not available. Check your YOUTUBE_API_KEY.")
        