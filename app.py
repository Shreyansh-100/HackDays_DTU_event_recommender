
import streamlit as st
import recommender as rec
from datetime import datetime

st.set_page_config(page_title="DTU", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    df = rec.load_events()
    emb = rec.ensure_embeddings(df)
    return df, emb

df, emb = load_data()

# Session + user id
if "user_id" not in st.session_state:
    st.session_state.user_id = "demo_user"

user_id = st.session_state.user_id
history = rec.load_user_history()
local_attended = st.session_state.get("local_attended", [])

st.title("🎓 DTU Attend")

tab1, tab2, tab3 = st.tabs(["🏠 Personalized", "🔍 Explore", "🤝 Collaborative"])

# --------------- Personalized Tab ---------------
with tab1:
    st.header("Your Personalized Feed")
    attended = list(set(history.get(user_id, []) + local_attended))
    attended_texts = [df.loc[df["event_id"] == eid, "text"].values[0] for eid in attended if eid in df["event_id"].values]
    if attended_texts:
        recs = rec.recommend_by_history(attended_texts, df, emb, top_n=8)
        for _, row in recs.iterrows():
            st.markdown(f"### {row['title']}")
            st.write(f"**Organizer:** {row['organizer']} | **Tags:** {row['tags']}")
            st.write(f"[Open Event]({row['link']}) | Score: {row['score']:.3f}")
            if st.button(f"Mark as Attended: {row['event_id']}", key=f"p_{row['event_id']}"):
                history = rec.add_attended_event(user_id, row["event_id"])
                st.success(f"Added {row['title']} to your history.")
    else:
        st.info("No history yet. Check 'Explore' tab to start adding events!")

# --------------- Search / Explore Tab ---------------
with tab2:
    st.header("Search Events by Interest (Semantic Search)")
    query = st.text_input("Enter your interest (e.g., 'AI', 'Finance', 'Hackathon')", "")
    if query:
        recs = rec.recommend_by_query(query, df, emb, top_n=10)
        for _, row in recs.iterrows():
            st.markdown(f"### {row['title']}")
            st.write(f"**Organizer:** {row['organizer']} | **Tags:** {row['tags']}")
            st.write(f"[Open Event]({row['link']}) | Score: {row['score']:.3f}")
            cols = st.columns([1,1,6])
            with cols[0]:
                if st.button(f"Add (Session)", key=f"sess_{row['event_id']}"):
                    local_attended.append(row["event_id"])
                    st.session_state.local_attended = local_attended
                    st.success("Added to session history.")
            with cols[1]:
                if st.button(f"Add (Save)", key=f"save_{row['event_id']}"):
                    history = rec.add_attended_event(user_id, row["event_id"])
                    st.success("Persisted to your saved history.")

with tab3:
    st.header("People similar to you also attended...")
    cf_recs = rec.recommend_cf(user_id, history, df, top_n=8)
    if cf_recs.empty:
        st.info("Collaborative suggestions will appear once multiple users mark events as attended.")
    else:
        for _, row in cf_recs.iterrows():
            st.markdown(f"### {row['title']}")
            st.write(f"**Organizer:** {row['organizer']} | **Tags:** {row['tags']}")
            st.write(f"[Open Event]({row['link']}) | CF Score: {row['cf_score']:.3f}")

st.markdown("---")
st.caption(f"Dataset size: {len(df)} | Model: SentenceTransformer ({rec.MODEL_NAME})")
st.caption(f"We have not filtered out expired events since the amount of ongoing workshops and events are too low to show implementation of the project")
