import pandas as pd
import streamlit as st

from src.data_loader import load_data, ids_for_language
from src.content_based import ContentBasedRecommender
from src.collaborative import UserBasedCF, ItemBasedCF, SVDRecommender, hybrid_recommend
from src.baselines import weighted_popularity, new_user_recommend

st.set_page_config(page_title="Cinema Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Cinema Recommender — Hollywood × Tollywood")
st.caption("Content-based + collaborative filtering on MovieLens-format data with a Telugu catalogue. "
           "Ratings in the bundled sample are synthetic.")


@st.cache_resource
def load_all():
    movies, ratings = load_data("data")
    return movies, ratings, ContentBasedRecommender(movies), {
        "Item-based CF": ItemBasedCF().fit(ratings),
        "User-based CF": UserBasedCF().fit(ratings),
        "SVD": SVDRecommender().fit(ratings),
    }


movies, ratings, cb, cfs = load_all()
title_of = movies.set_index("movieId")["title"]
langs = ["All"] + sorted(movies["language"].unique(), key=lambda x: (x != "Telugu", x))
lang = st.sidebar.selectbox("Language filter", langs)
n = st.sidebar.slider("Number of results", 3, 25, 10)
allowed = ids_for_language(movies, lang)

t1, t2, t3, t4 = st.tabs(["🔥 Tollywood spotlight", "🎞️ Similar movies", "👤 For a user", "✨ New here?"])

with t1:
    pop = weighted_popularity(movies, ratings)
    te = pop[pop.language == "Telugu"].sort_values("weighted_rating", ascending=False).head(n)
    st.subheader("Top-rated Telugu movies")
    st.dataframe(te[["title", "genres", "director", "count", "weighted_rating"]].round(2), hide_index=True, use_container_width=True)
    st.bar_chart(te.set_index("title")["weighted_rating"])

with t2:
    default = int(movies.index[movies.title.str.contains("RRR")][0]) if movies.title.str.contains("RRR").any() else 0
    title = st.selectbox("Pick a movie", movies["title"].tolist(), index=default)
    row = movies[movies.title == title].iloc[0]
    st.write(f"**{row.title}** · {row.genres} · {row.language} · {row.director}")
    st.dataframe(cb.similar_movies(int(row.movieId), n, allowed)[["title", "genres", "language", "director", "score"]].round(3),
                 hide_index=True, use_container_width=True)

with t3:
    user = st.selectbox("User ID", sorted(ratings["userId"].unique()))
    method = st.radio("Method", ["Hybrid", "Item-based CF", "User-based CF", "SVD", "Content-based"], horizontal=True)
    liked = ratings[(ratings.userId == user) & (ratings.rating >= 4)].merge(movies, on="movieId")
    st.write("**Already loved (rating ≥ 4):** " + ", ".join(liked.sort_values("rating", ascending=False).title.head(8)))
    if method == "Hybrid":
        out = hybrid_recommend(user, ratings, cfs["Item-based CF"], cb, movies, n, allowed_ids=allowed)
    elif method == "Content-based":
        out = cb.recommend_for_user(user, ratings, n, allowed)
    else:
        out = cfs[method].recommend(user, n, allowed).merge(movies[["movieId", "title", "genres", "language", "director"]], on="movieId")
    if method == "Item-based CF":
        out["because you liked"] = [", ".join(title_of[m] for m, _ in cfs["Item-based CF"].explain(user, mid, ratings, 2)) for mid in out.movieId]
    st.dataframe(out.drop(columns="movieId").round(3), hide_index=True, use_container_width=True)

with t4:
    st.write("No history yet? Tell us what you like (cold-start recommendations).")
    all_genres = sorted({g for s in movies.genres for g in s.split("|")})
    g = st.multiselect("Favourite genres", all_genres, default=["Action", "Drama"])
    l = st.multiselect("Languages", sorted(movies.language.unique()), default=["Telugu"])
    st.dataframe(new_user_recommend(movies, ratings, g, l, n).drop(columns="movieId").round(3),
                 hide_index=True, use_container_width=True)
