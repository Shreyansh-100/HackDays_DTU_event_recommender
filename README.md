# 🎓 DTU Event Recommender System: DTU Attend

> A personalized platform that recommends campus events, workshops, and competitions to students based on their interests, search queries, and past attendance.

---

## 🚀 Overview

There are numerous events and workshops happening across the DTU campus every week, but many go unnoticed or get lost in the flood of different activities.  
This project aims to solve that by building an intelligent **event recommendation system** that helps students discover relevant opportunities aligned with their goals and interests.

The system uses **Natural Language Processing (NLP)** and **Machine Learning** to understand event descriptions, tags, and student preferences.  
It provides context-aware, hybrid recommendations combining **semantic similarity**, **collaborative filtering**, and **popularity boosts**.

---

## 🧠 Features

- 🔍 **Semantic Search:** Understands event context using Sentence Transformers (MiniLM).
- 🧩 **Personalized Feed:** Learns from the user’s attended events.
- 💬 **Hybrid Model:** Combines user history + search query embeddings.
- 📈 **Popularity Boost:** Prioritizes trending/high-impression events.
- 👥 **Collaborative Filtering:** Recommends based on similar users’ interests.
- 🕒 **Session + Persistent History:** Tracks both temporary and saved attendance.
- 🧾 **Dataset:** Pulled from Unstop DTU societies (AIMS, IEEE, IFSA, EHAX, etc.)

---

## 🧰 Tech Stack

| Component | Tech Used |
|------------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML/NLP** | Sentence Transformers (MiniLM), Scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Similarity Metrics** | Cosine Similarity |
| **Model Caching** | Pickle + NumPy |
| **Embeddings** | `all-MiniLM-L6-v2` from Hugging Face |
