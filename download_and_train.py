import os
import io
import urllib.request
import zipfile
import pickle
import numpy as np
import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "data"
MODEL_DIR = "model"

def prepare_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    movies_path = os.path.join(DATA_DIR, "movies.csv")

    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        print("Downloading MovieLens Small dataset...")
        resp = urllib.request.urlopen(MOVIELENS_URL)
        with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
            for file in z.namelist():
                if file.endswith("ratings.csv"):
                    with open(ratings_path, "wb") as f:
                        f.write(z.read(file))
                elif file.endswith("movies.csv"):
                    with open(movies_path, "wb") as f:
                        f.write(z.read(file))
        print("Data downloaded successfully.")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies

class BiasedMatrixFactorization:
    def __init__(self, n_users, n_items, k=30, lr=0.007, reg=0.04, epochs=25):
        self.k = k
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.n_users = n_users
        self.n_items = n_items

    def fit(self, train_data):
        self.mu = float(np.mean(train_data[:, 2]))
        self.bu = np.zeros(self.n_users, dtype=np.float32)
        self.bi = np.zeros(self.n_items, dtype=np.float32)
        self.P = np.random.normal(0, 0.1, (self.n_users, self.k)).astype(np.float32)
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.k)).astype(np.float32)

        n_samples = len(train_data)
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            sq_err_sum = 0.0

            for idx in indices:
                u = int(train_data[idx, 0])
                i = int(train_data[idx, 1])
                r = float(train_data[idx, 2])

                dot = np.dot(self.P[u], self.Q[i])
                pred = self.mu + self.bu[u] + self.bi[i] + dot
                err = r - pred
                sq_err_sum += err ** 2

                # SGD Parameter Updates with L2 Regularization
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                p_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_u_old - self.reg * self.Q[i])

            rmse = np.sqrt(sq_err_sum / n_samples)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d}/{self.epochs} | Train RMSE: {rmse:.4f}")

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))

def main():
    ratings, movies = prepare_data()

    # Index Encoding
    user_ids = ratings["userId"].unique()
    movie_ids = movies["movieId"].unique()

    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
    movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
    idx2movie = {idx: mid for mid, idx in movie2idx.items()}

    ratings["u_idx"] = ratings["userId"].map(user2idx)
    ratings["i_idx"] = ratings["movieId"].map(movie2idx)
    ratings = ratings.dropna(subset=["u_idx", "i_idx"])

    train_data = ratings[["u_idx", "i_idx", "rating"]].to_numpy()

    model = BiasedMatrixFactorization(
        n_users=len(user2idx),
        n_items=len(movie2idx),
        k=32,
        lr=0.008,
        reg=0.03,
        epochs=20
    )
    
    print("Training Biased Latent Factor Model...")
    model.fit(train_data)

    # Save artifacts
    checkpoint = {
        "model": model,
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "idx2movie": idx2movie,
        "movies": movies[["movieId", "title", "genres"]].set_index("movieId").to_dict(orient="index")
    }
    with open(os.path.join(MODEL_DIR, "recsys_model.pkl"), "wb") as f:
        pickle.dump(checkpoint, f)
    print("Model saved to model/recsys_model.pkl")

if __name__ == "__main__":
    main()