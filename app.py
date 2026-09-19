import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st

class BiasedMatrixFactorization:
    def __init__(self, n_users, n_items, k=30, lr=0.007, reg=0.04, epochs=25):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.n_users = n_users
        self.n_items = n_items

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))

st.set_page_config(page_title="Movie Matchmaker", layout="wide")

@st.cache_resource
def load_artifacts():
    path = os.path.join("model", "recsys_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
if artifacts is None:
    st.error("Model artifacts not found. Run download_and_train.py first.")
    st.stop()

model = artifacts["model"]
movie2idx = artifacts["movie2idx"]
idx2movie = artifacts["idx2movie"]
movies_dict = artifacts["movies"]

def clean_movie_title(raw_title: str) -> str:
    match = re.search(r"^(.*?),\s*(The|A|An)(\s*\(.*\))?$", raw_title, flags=re.IGNORECASE)
    if match:
        main_title = match.group(1).strip()
        article = match.group(2).strip().capitalize()
        year_part = match.group(3) if match.group(3) else ""
        return f"{article} {main_title}{year_part}".strip()
    return raw_title.strip()

# Build title lookup without genres and with natural English articles
title_to_mid = {}
for mid, meta in movies_dict.items():
    if mid in movie2idx:
        display_title = clean_movie_title(meta["title"])
        title_to_mid[display_title] = mid

all_movie_options = sorted(list(title_to_mid.keys()))

st.title("🎯 Personalized Movie Recommender")
st.write(
    "Search and select movies you enjoy. Our **Regularized Latent Factor Model with Biases** "
    "learns your taste vector in real-time and recommends candidate titles."
)

# Clean default picks
default_candidates = ["The Matrix (1999)", "The Shawshank Redemption (1994)", "Inception (2010)"]
initial_selection = [m for m in default_candidates if m in title_to_mid]

selected_titles = st.multiselect(
    "Search and select movies you have watched:",
    options=all_movie_options,
    default=initial_selection,
    help="Type movie names without trailing articles (e.g. 'The Matrix', not 'Matrix, The')."
)

if not selected_titles:
    st.info("Select at least 1 movie above to start receiving recommendations.")
    st.stop()

st.subheader("Your Ratings")
st.write("Rate each selected title (0.5 to 5.0 stars):")

cols = st.columns(min(len(selected_titles), 4))
user_ratings = {}

for idx, title in enumerate(selected_titles):
    col = cols[idx % len(cols)]
    mid = title_to_mid[title]
    user_ratings[mid] = col.slider(
        title,
        min_value=0.5,
        max_value=5.0,
        value=4.5,
        step=0.5,
        key=f"rate_{mid}"
    )

top_k = st.sidebar.slider("Number of Recommendations", min_value=5, max_value=30, value=10, step=5)
filter_genre = st.sidebar.selectbox(
    "Filter by Genre (Optional)",
    options=["All", "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"]
)

if st.button("Generate Recommendations 🚀", type="primary"):
    with st.spinner("Computing real-time latent taste vector..."):
        rated_indices = [movie2idx[m] for m in user_ratings.keys()]
        r_actual = np.array([user_ratings[m] for m in user_ratings.keys()], dtype=np.float32)

        Q_sub = model.Q[rated_indices]
        b_i_sub = model.bi[rated_indices]

        # 1. User bias
        b_u_new = float(np.mean(r_actual - (model.mu + b_i_sub)))

        # 2. Regularized closed-form solve for p_new
        y = r_actual - (model.mu + b_u_new + b_i_sub)
        reg_ident = model.reg * np.eye(model.k)
        p_new = np.linalg.solve(Q_sub.T @ Q_sub + reg_ident, Q_sub.T @ y)

        # 3. Score the catalog
        all_preds = model.mu + b_u_new + model.bi + np.dot(model.Q, p_new)

        # 4. Filter unrated & optional genre
        candidate_indices = [i for i in range(model.n_items) if i not in rated_indices]
        if filter_genre != "All":
            candidate_indices = [
                i for i in candidate_indices 
                if filter_genre in movies_dict[idx2movie[i]]["genres"]
            ]

        ranked_indices = sorted(candidate_indices, key=lambda i: all_preds[i], reverse=True)[:top_k]

        st.subheader("Recommended For You")
        results = []
        for rank, i_idx in enumerate(ranked_indices, start=1):
            mid = idx2movie[i_idx]
            interaction = float(np.dot(model.Q[i_idx], p_new))
            predicted_score = float(np.clip(all_preds[i_idx], 0.5, 5.0))
            
            results.append({
                "Rank": rank,
                "Title": clean_movie_title(movies_dict[mid]["title"]),
                "Genres": movies_dict[mid]["genres"],
                "Predicted Rating": f"{predicted_score:.2f} ⭐",
                "Popularity Bias ($b_i$)": f"{model.bi[i_idx]:+.2f}",
                "Taste Alignment ($q_i^T p_{you}$)": f"{interaction:+.2f}"
            })

        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Your Rating Deviation ($b_{user}$)", f"{b_u_new:+.2f}")
        c2.metric("Dataset Global Baseline ($\mu$)", f"{model.mu:.2f}")

        # Plain English Explanations
        with st.expander("💡 How does the model calculate your recommendations?"):
            st.markdown(
                """
                Each recommendation score is broken down into four core parts: 
                $$\\hat{r}_{xi} = \\mu + b_x + b_i + q_i^T p_x$$

                * **Dataset Global Baseline ($\\mu$):** The average rating across every single movie and user in the dataset (around 3.5 stars)[cite: 1]. It serves as the baseline before considering specific tastes[cite: 1].
                * **Your Rating Deviation ($b_{user}$ or $b_x$):** How tough or generous you are compared to the average viewer[cite: 1]. A positive score means you give higher ratings than most people, while a negative score means you are harder to please[cite: 1].
                * **Popularity Bias ($b_{item}$ or $b_i$):** How much better or worse a specific movie is rated compared to the global average[cite: 1]. Acclaimed movies have a positive bias, while universally disliked titles have a negative bias[cite: 1].
                * **Taste Alignment ($q_i^T p_{you}$):** The personalized match between your specific latent preferences ($p_{you}$) and the movie's latent characteristics ($q_i$)[cite: 1]. This captures niche compatibility beyond general popularity—like whether you love psychological sci-fi or dark comedies[cite: 1].
                """
            )