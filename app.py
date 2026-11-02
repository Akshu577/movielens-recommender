"""Streamlit app: movie recommendations from the matrix factorization model, with a popularity penalty."""
import numpy as np
import pandas as pd
import streamlit as st

from src.mf import load_data, rmse, split, train_mf

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("Movie Recommender")
st.caption("Matrix factorization trained from scratch with SGD on MovieLens 100K (100,000 ratings, 943 users, "
           "1,682 movies from the 1990s). Source: github.com/Akshu577/movielens-recommender")


@st.cache_resource(show_spinner="Downloading MovieLens and training the model (about 30 seconds)...")
def setup():
    data, titles = load_data()
    train, test = split(data)
    n_users, n_items = data[:, 0].max() + 1, data[:, 1].max() + 1
    model = train_mf(train, n_users, n_items)
    popularity = np.bincount(train[:, 1], minlength=n_items)
    return data, np.array(titles), model, rmse(model, test), popularity


data, titles, (P, Q, bu, bi, mu), test_rmse, popularity = setup()
st.write(f"Test RMSE: **{test_rmse:.3f}** (always predicting the average rating scores 1.13)")

mode = st.radio("Recommend for", ["Me: I'll pick movies I like", "An existing MovieLens user"], horizontal=True)
penalty = st.slider("Popularity penalty", 0.0, 0.3, 0.0, 0.05,
                    help="Higher values push down blockbusters so lesser-known movies get a chance. "
                         "The notebook measures this trade-off.")
min_ratings = st.slider("Only movies rated at least this many times", 1, 100, 20,
                        help="Movies with very few ratings can score high by chance.")

if mode.startswith("Me"):
    popular = np.argsort(-popularity)[:300]
    liked = st.multiselect("Pick a few movies you like (3 or more works best)", sorted(set(titles[popular])))
    if not liked:
        st.stop()
    seen = [int(np.where(titles == t)[0][0]) for t in liked]
    # Fold in a new user: fit their taste vector to these items with ridge regression, items fixed
    Qs = Q[seen]
    target = 5.0 - mu - bi[seen]
    p_user = np.linalg.solve(Qs.T @ Qs + 0.5 * np.eye(Q.shape[1]), Qs.T @ target)
    scores = mu + bi + Q @ p_user
else:
    user = st.number_input("User id", 1, P.shape[0], 1) - 1
    rated = data[data[:, 0] == user]
    seen = rated[:, 1].tolist()
    top = rated[np.argsort(-rated[:, 2])][:8]
    st.write("**Movies this user rated highest:**")
    st.dataframe(pd.DataFrame({"Movie": titles[top[:, 1]], "Rating": top[:, 2]}), hide_index=True)
    scores = mu + bu[user] + bi + Q @ P[user]

scores = scores - penalty * np.log1p(popularity)
scores[seen] = -np.inf
scores[popularity < min_ratings] = -np.inf
recs = np.argsort(-scores)[:10]
st.subheader("Recommended for you")
st.dataframe(pd.DataFrame({"Movie": titles[recs], "Times rated": popularity[recs]}), hide_index=True,
             width="stretch")
