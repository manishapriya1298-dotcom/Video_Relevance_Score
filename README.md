# Video_Relevance_Score

🎙️📊 **Relevance Reactor**: Decode Meaning, Detect Drift, Deliver Insight  
🚀 **Mission**
Welcome to Relevance Reactor — a semantic intelligence engine that listens, understands, and scores the soul of your content. Whether you're analyzing a podcast, a lecture, or a product pitch, this module reveals how well your words align with your intent.
“Not all content is created equal. Some speaks truth. Some sells. We help you tell the difference.”  


🧠 **What It Does** 
🔊 Transcription (Optional for Demo)
- Converts audio/video into timestamped text using OpenAI Whisper or YouTube auto-captions
- Offline mode: Drop in a sample transcript and skip the noise  
🧬 **Semantic Relevance Analysis** 
- Embeds title, description, and transcript into a shared vector space
- Compares segments for topical alignment, drift, and promotional bias
- Labels each chunk as: Relevant, Irrelevant, or Promotional   
🎯 **Scoring & Explanation**  
- Outputs a Relevance Score (0–100)
- Generates human-readable reasoning:
- “Content strongly matches the title ‘AI in Education’ — 85% relevant. Some sections promote an unrelated product.” 
🧰 **Tech Stack- Python: Core logic and orchestration**  
- Whisper API: Transcription
- SentenceTransformers: Embeddings
- KeyBERT + Zero-shot: Promo detection & tagging
- Streamlit / Plotly: Dashboard visualization 
🌈 **Sample Output**{
  "score": 85.0,
  "explanation": "Content strongly matches the title ‘AI in Education’ — 85% relevant. Some sections promote an unrelated product.", 

🧭 **Why It Matters**  
- 🎓 Educators: Validate lecture content against curriculum goals
- 🎥 Creators: Ensure videos stay on-topic and avoid accidental promo drift
- 🧑‍⚖️ Reviewers: Score relevance at scale for audits or competitions
- 🧠 Researchers: Study semantic alignment across domains  
🧪 **Future Directions**- 
🧬 Multi-modal fusion: Combine audio tone + transcript for deeper promo detection 
- 🧭 Conversational drift tracking: Detect when speakers veer off-topic
- 🧠 Explainable AI: Visualize embedding space and decision boundaries

