
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

CSV_PATH = "scraped_1.csv"
EMB_PATH = "event_embeddings.npy"
USER_HISTORY_PATH = "user_history.json"
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading SentenceTransformer model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model
def load_events(csv_path: str = CSV_PATH) -> pd.DataFrame:
    
    # Absolute + relative file path check
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, "scraped_1.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(
                f"❌ Could not find dataset.\n"
                f"Tried:\n 1️⃣ {csv_path}\n 2️⃣ {alt_path}\n\n"
                "Make sure scraped_1.csv is in the same folder as app.py and recommender.py."
            )

    print(f"✅ Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {os.path.basename(csv_path)}")

    # Normalize column names
    mapping = {
        "Competition Name": "title",
        "Competition URL": "link",
        "Organizer": "organizer",
        "tags": "tags",
        "Status": "status",
        "Event Type": "event_type",
        "Posted Date": "posted_date",
        "Impressions": "Impressions",
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    # Ensure consistent columns
    for col in ["title", "organizer", "tags", "status", "link", "Impressions"]:
        if col not in df.columns:
            df[col] = ""

    if "event_id" not in df.columns:
        df["event_id"] = [f"E{i+1}" for i in range(len(df))]

    df["tags"] = df["tags"].fillna("").astype(str)
    df["tags"] = df["tags"].apply(lambda s: ", ".join(s.split()) if s.strip() and "," not in s else s)
    df["organizer"] = df["organizer"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)

    df["text"] = (df["title"] + " " + df["organizer"] + " " + df["tags"]).str.strip()
    if "Impressions" in df.columns:
        df["Impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)
        max_imp = df["Impressions"].max() if df["Impressions"].max() > 0 else 1
        df["popularity"] = df["Impressions"] / max_imp
    else:
        df["popularity"] = 0.0

    df["is_active"] = ~df["status"].astype(str).str.lower().eq("expired")

    # Reset index for consistency
    df = df.reset_index(drop=True)
    return df


def ensure_embeddings(df: pd.DataFrame, path: str = EMB_PATH) -> np.ndarray:
    if os.path.exists(path):
        emb = np.load(path)
        if emb.shape[0] == len(df):
            print("Loaded cached embeddings.")
            return emb
    model = get_model()
    emb = model.encode(df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    np.save(path, emb)
    return emb


# ------------------------------- Popularity Boost -------------------------------
def apply_popularity_boost(scores: np.ndarray, popularity: np.ndarray, pop_weight: float = 0.15) -> np.ndarray:
    pop = popularity.flatten()
    return scores * (1 + pop_weight * pop)


# ------------------------------- Recommenders -------------------------------
def recommend_by_query(query: str, df: pd.DataFrame, emb: np.ndarray, top_n=5, pop_weight=0.15):
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q_emb, emb).flatten()
    final_scores = apply_popularity_boost(sims, df["popularity"].values, pop_weight)
    df_scores = df.copy()
    df_scores["score"] = final_scores
    return df_scores.sort_values("score", ascending=False).head(top_n)


def recommend_by_history(attended_events: List[str], df: pd.DataFrame, emb: np.ndarray, top_n=5, pop_weight=0.15):
    if not attended_events:
        return pd.DataFrame()
    model = get_model()
    user_emb = model.encode(attended_events, convert_to_numpy=True)
    user_profile = np.mean(user_emb, axis=0, keepdims=True)
    sims = cosine_similarity(user_profile, emb).flatten()
    final_scores = apply_popularity_boost(sims, df["popularity"].values, pop_weight)
    df_scores = df.copy()
    df_scores["score"] = final_scores
    return df_scores.sort_values("score", ascending=False).head(top_n)


def recommend_hybrid(attended_events: List[str], query: str, df: pd.DataFrame, emb: np.ndarray, alpha=0.7, pop_weight=0.15, top_n=5):
    model = get_model()
    user_emb = model.encode(attended_events, convert_to_numpy=True)
    user_profile = np.mean(user_emb, axis=0)
    query_emb = model.encode([query], convert_to_numpy=True)[0]
    hybrid_vec = alpha * user_profile + (1 - alpha) * query_emb
    sims = cosine_similarity([hybrid_vec], emb).flatten()
    final_scores = apply_popularity_boost(sims, df["popularity"].values, pop_weight)
    df_scores = df.copy()
    df_scores["score"] = final_scores
    return df_scores.sort_values("score", ascending=False).head(top_n)


# ------------------------------- Collaborative Filtering -------------------------------
def build_user_event_matrix(history: Dict[str, List[str]], df: pd.DataFrame):
    records = []
    for user, events in history.items():
        for eid in events:
            records.append({"user_id": user, "event_id": eid, "attended": 1})
    if not records:
        return pd.DataFrame(), [], []
    df_hist = pd.DataFrame(records)
    pivot = df_hist.pivot_table(index="user_id", columns="event_id", values="attended", fill_value=0)
    for eid in df["event_id"].unique():
        if eid not in pivot.columns:
            pivot[eid] = 0
    pivot = pivot[df["event_id"].unique()]
    return pivot, pivot.index.tolist(), pivot.columns.tolist()


def recommend_cf(user_id: str, history: Dict[str, List[str]], df: pd.DataFrame, top_n=5):
    pivot, users, events = build_user_event_matrix(history, df)
    if pivot.empty or user_id not in users:
        return pd.DataFrame()
    item_matrix = normalize(pivot.T.values, axis=1)
    sim = cosine_similarity(item_matrix)
    event_to_idx = {eid: i for i, eid in enumerate(events)}
    attended_idx = [event_to_idx[eid] for eid in history.get(user_id, []) if eid in event_to_idx]
    if not attended_idx:
        return pd.DataFrame()
    scores = sim[:, attended_idx].sum(axis=1)
    for idx in attended_idx:
        scores[idx] = -1
    df_cf = df.set_index("event_id").loc[events].reset_index()
    df_cf["cf_score"] = scores
    return df_cf.sort_values("cf_score", ascending=False).head(top_n)


# ------------------------------- User History -------------------------------
def load_user_history(path=USER_HISTORY_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_user_history(history: Dict[str, List[str]], path=USER_HISTORY_PATH):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def add_attended_event(user_id: str, event_id: str):
    history = load_user_history()
    if user_id not in history:
        history[user_id] = []
    if event_id not in history[user_id]:
        history[user_id].append(event_id)
        save_user_history(history)
    return history
