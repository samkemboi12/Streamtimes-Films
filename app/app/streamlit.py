import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional

# ---------------- Load Models & Data ----------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model_path   = Path("../models/svd_model.pkl").resolve()
    cosine_path  = Path("../models/cosine_sim.pkl").resolve()
    svd_model    = pickle.load(open(model_path,  "rb"))
    cosine_sim   = pickle.load(open(cosine_path, "rb"))

    ratings = pd.read_csv("../../data/ratings.csv").drop(columns="timestamp")
    movies  = pd.read_csv("../../data/movies.csv")
    tags    = pd.read_csv("../../data/tags.csv")
    

    # Clean titles / extract year
    movies["year"]  = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)
    movies.dropna(inplace=True)

    movie_ratings = ratings.merge(movies, on="movieId", how="left").dropna()
    tagged        = tags.merge(movie_ratings, on=["movieId", "userId"], how="left").dropna()
    merged_tags   = (
        tagged.groupby(["movieId", "title", "genres", "year"])["tag"]
        .agg(lambda t: ", ".join(set(t)))
        .reset_index(name="merged_tags")
    )
    merged_tags["genres"] = merged_tags["genres"].str.replace("|", " ")
    indices = pd.Series(merged_tags.index, index=merged_tags["title"])
    return svd_model, cosine_sim, ratings, movie_ratings, merged_tags, indices

svd_model, cosine_sim, ratings, movie_ratings, merged_tags, indices = load_assets()

# ---------------- Recommender ----------------
def recommend(user_id: int, top_n: int = 10, title: Optional[str] = None):
    rated = movie_ratings[movie_ratings["userId"] == user_id]["movieId"].unique()
    cold  = len(rated) == 0

    if not cold:  # collaborative filtering
        unrated = [m for m in movie_ratings["movieId"].unique() if m not in rated]
        preds   = [(m, svd_model.predict(user_id, m).est) for m in unrated]
        top_ids = [m for m, _ in sorted(preds, key=lambda x: x[1], reverse=True)[:top_n]]
        recs    = movie_ratings[movie_ratings["movieId"].isin(top_ids)][["title","genres","year"]].drop_duplicates()
        recs["src"] = "CF"
        return recs

    # cold user
    if title is None or title not in indices:
        return "cold_start"

    seed_idx   = indices[title]
    sims       = list(enumerate(cosine_sim[seed_idx]))
    sim_idx    = [i[0] for i in sorted(sims, key=lambda x: x[1], reverse=True)[1 : top_n+1]]
    recs       = merged_tags.iloc[sim_idx][["title","genres","year"]].copy()
    recs["src"] = "CB"
    return recs

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Hybrid Movie Recommender")

user_txt = st.text_input("🔑 User ID:", key="uid")

if user_txt:
    if not user_txt.isdigit():
        st.error("User ID must be numeric.")
        st.stop()

    user_id = int(user_txt)

    with st.spinner("🔍 Getting recommendations..."):
        result = recommend(user_id)

    # Cold‑start: ask for seed movie
    if isinstance(result, str) and result == "cold_start":
        st.warning("New user detected. Type a movie you like to get started.")
        seed = st.text_input("🎥 Movie you liked:", key="seed")
        if seed:
            with st.spinner("🔎 Finding similar movies..."):
                cb_res = recommend(-1, title=seed)
            if isinstance(cb_res, pd.DataFrame) and not cb_res.empty:
                st.subheader("🎯 Because you liked this...")
                for _, m in cb_res.iterrows():
                    st.markdown(f"**🎬 {m.title}**  \n*{m.genres}* ({int(m.year)})")
            else:
                st.info("No similar movies found.")
        st.stop()

    # Warm user
    if isinstance(result, pd.DataFrame):
        st.success("✅ Personalized picks")
        tab_cf, tab_cb = st.tabs(["Based on Activity", "Search Similar Movie"])

        with tab_cf:
            for _, m in result.iterrows():
                st.markdown(f"**🎬 {m.title}**  \n*{m.genres}* ({int(m.year)})")

        with tab_cb:
            seed2 = st.text_input("Find movies similar to:", key="cb_warm")
            if seed2:
                with st.spinner("🔄 Searching..."):
                    cb_alt = recommend(user_id, title=seed2)
                if isinstance(cb_alt, pd.DataFrame) and not cb_alt.empty:
                    for _, m in cb_alt.iterrows():
                        st.markdown(f"**🎬 {m.title}**  \n*{m.genres}* ({int(m.year)})")
                else:
                    st.info("No similar titles found.")
