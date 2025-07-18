from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pathlib import Path
import pandas as pd
import pickle

# ------------------- Load Models -------------------

model_path = Path("../models/svd_model.pkl").resolve()
cosine_sim_path = Path("../models/cosine_sim.pkl").resolve()

with open(model_path, "rb") as f:
    svd_model = pickle.load(f)

with open(cosine_sim_path, 'rb') as f:
    cosine_sim = pickle.load(f)

# ------------------- Load Data -------------------

ratings = pd.read_csv("../../data/ratings.csv")
movies = pd.read_csv("../../data/movies.csv")
tags = pd.read_csv("../../data/tags.csv")
ratings.drop('timestamp', axis=1, inplace=True)

# Clean movie titles and extract year
movies_clean = movies.copy()
movies_clean["year"] = movies_clean["title"].str.extract(r"\((\d{4})\)").astype(float)
movies_clean["title"] = movies_clean["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)
movies_clean.dropna(inplace=True)

# Merge ratings with movie metadata
movie_ratings = ratings.merge(movies_clean, on='movieId', how='left')
movie_ratings.dropna(inplace=True)

# Merge tags with ratings
tagged_movies = tags.merge(movie_ratings, on=['movieId', 'userId'], how='left')
tagged_movies.dropna(inplace=True)

# Aggregate tags per movie
merged_tags = tagged_movies.groupby(['movieId', 'title', 'genres', 'year'])['tag'] \
    .agg(lambda tags: ', '.join(set(tags))) \
    .reset_index(name='merged_tags')

# Preprocess genres for content-based
def merge_genres(x):
    return ' '.join(x.split('|'))

merged_tags['genres'] = merged_tags['genres'].apply(merge_genres)

# Create title index mapping for lookup
indices = pd.Series(merged_tags.index, index=merged_tags['title'])

# Get all known userIds
userIds = movie_ratings['userId'].unique().tolist()

# ------------------- Hybrid Recommender Logic -------------------

def mixed_hybrid_system(user_id, top_n=10, title=None):
    """
    Hybrid system: Uses CF for known users, CB for cold users.
    """
    rated_movie_ids = movie_ratings.loc[
        movie_ratings['userId'] == user_id, 'movieId'
    ].unique()
    cold_user = len(rated_movie_ids) == 0

    if not cold_user:
        # Collaborative filtering path
        all_movie_ids = movie_ratings['movieId'].unique()
        unrated_movies = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

        cf_preds = [(mid, svd_model.predict(user_id, mid).est) for mid in unrated_movies]
        top_cf = sorted(cf_preds, key=lambda x: x[1], reverse=True)[:top_n]
        cf_ids = [mid for mid, _ in top_cf]

        recs = movie_ratings[movie_ratings['movieId'].isin(cf_ids)][
            ['title', 'genres', 'year']].drop_duplicates()
        recs['source'] = 'Collaborative Filtering'
        return recs

    # Cold user: Content-based recommendation from input title
    if title is None or title not in indices:
        return "cold_start"

    seed_idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[seed_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    cb_indices = [i[0] for i in sim_scores]

    recs = merged_tags.iloc[cb_indices][['title', 'genres', 'year']].copy()
    recs['source'] = 'Content-Based (cold user)'
    return recs

# ------------------- FastAPI Setup -------------------

app = FastAPI()

# Allow Streamlit frontend or any other CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Welcome to the Movie Recommender API"}

@app.get("/recommend")
def get_recommendations(user_id: int, n: int = 10, title: Optional[str] = None):
    try:
        # Check if user is cold
        rated_movies = movie_ratings[movie_ratings['userId'] == user_id]
        cold_user = rated_movies.empty

        if cold_user:
            # Cold user
            if title is None:
                # No title provided: tell frontend to ask for it
                return {"status": "cold_start", "message": "New user. Please enter a movie you like."}

            # Title provided: check if valid
            if title not in indices:
                return {"error": f"Seed title '{title}' not found in the catalog."}

            # Recommend based on title
            recs = mixed_hybrid_system(user_id, n, title)
            return recs.to_dict(orient='records')

        # Warm user: use CF
        recs = mixed_hybrid_system(user_id, n)
        return recs.to_dict(orient='records')

    except Exception as e:
        return {"error": str(e)}


