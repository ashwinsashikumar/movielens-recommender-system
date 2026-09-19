# 🎬 Movie Matchmaker: Regularized Biased Latent Factor Recommender System

A movie recommendation engine built using **Regularized Matrix Factorization with Learned Baseline Biases**, inspired by latent-factor approaches popularized during the Netflix Prize and trained on the **MovieLens Latest-Small** dataset (~100k ratings across ~9,700 movies).

The system features real-time latent preference inference for previously unseen users, natural movie-title parsing, transparent score decomposition (`μ + b_u + b_i + q_iᵀp_u`), and an interactive web interface powered by Streamlit.

---

## 🚀 Live Demo

- **Web App:** [Launch Movie Matchmaker](https://movielens-recommender-system-ed4z95bxmknjqxs8dc9fgn.streamlit.app/)
- **GitHub Repository:** [View Source Code](https://github.com/ashwinsashikumar/movielens-recommender-system)

---

## 🧠 Mathematical Foundation

The recommendation model estimates the rating $\hat{r}_{ui}$ that user $u$ gives to movie $i$ by combining global and user/item-specific baseline effects with a latent interaction term:

```math
\hat{r}_{ui}
=
\mu + b_u + b_i + q_i^T p_u
```

where:

- **$\mu$ (Global Baseline):** Dataset-wide average rating across all movies and users.
- **$b_u$ (User Bias):** Trainable bias capturing how critical or generous user $u$ tends to be relative to the global average.
- **$b_i$ (Item Bias):** Trainable bias capturing whether movie $i$ tends to receive ratings above or below the global average.
- **$q_i^T p_u$ (Latent Interaction):** Inner product between the user's latent preference vector $p_u \in \mathbb{R}^k$ and the movie's latent characteristic vector $q_i \in \mathbb{R}^k$.

The latent vectors allow the model to learn hidden preference dimensions directly from observed rating patterns without requiring manually defined movie features.

---

## ⚙️ Optimization via Stochastic Gradient Descent

The model is trained by minimizing regularized squared error over the set of observed ratings $R$:

```math
\mathcal{L}
=
\frac{1}{2}
\sum_{(u,i)\in R}
\left(
r_{ui}-\hat{r}_{ui}
\right)^2
+
\frac{\lambda}{2}
\left(
\lVert p_u\rVert_2^2
+
\lVert q_i\rVert_2^2
+
b_u^2
+
b_i^2
\right)
```

The L2 regularization term penalizes excessively large parameter values and helps reduce overfitting in the sparse user-item rating matrix.

For each observed rating $r_{ui}$, the prediction error is calculated as:

```math
e_{ui}
=
r_{ui}-\hat{r}_{ui}
```

The parameters are then updated using Stochastic Gradient Descent (SGD):

```math
b_u
\leftarrow
b_u + \gamma
\left(
e_{ui}-\lambda b_u
\right)
```

```math
b_i
\leftarrow
b_i + \gamma
\left(
e_{ui}-\lambda b_i
\right)
```

```math
p_u
\leftarrow
p_u + \gamma
\left(
e_{ui}q_i-\lambda p_u
\right)
```

```math
q_i
\leftarrow
q_i + \gamma
\left(
e_{ui}p_u-\lambda q_i
\right)
```

where:

- $\gamma$ is the learning rate.
- $\lambda$ is the L2 regularization strength.

The updates for $p_u$ and $q_i$ are computed using their parameter values prior to the current SGD step.

---

## ⚡ Real-Time Cold-Start Inference

Traditional collaborative filtering models face a **cold-start problem** when a completely new user enters the system because no learned latent representation exists for that user.

Movie Matchmaker addresses this by asking a new user to rate a small set of movies through the Streamlit interface.

These ratings are used to estimate the new user's bias $b_{\mathrm{new}}$ and infer their latent preference vector $p_{\mathrm{new}}$ without retraining the complete recommendation model.

The new user's bias is estimated from the residual between their submitted ratings and the corresponding global and item baselines.

Given the latent vectors of the movies rated by the new user, the system solves a regularized least-squares problem:

```math
p_{\mathrm{new}}
=
\left(
Q_{\mathrm{sub}}^T Q_{\mathrm{sub}}
+
\lambda I
\right)^{-1}
Q_{\mathrm{sub}}^T
\left(
r_{\mathrm{actual}}
-
\mu
-
b_{\mathrm{new}}
-
b_{i,\mathrm{sub}}
\right)
```

where:

- $Q_{\mathrm{sub}}$ contains the learned latent vectors of the movies rated by the new user.
- $r_{\mathrm{actual}}$ contains the ratings submitted by the new user.
- $b_{\mathrm{new}}$ is the estimated rating bias of the new user.
- $b_{i,\mathrm{sub}}$ contains the learned item biases of the rated movies.
- $I$ is the identity matrix.
- $\lambda$ controls the strength of regularization.

The inferred vector $p_{\mathrm{new}}$ can then be used to predict the user's rating for any candidate movie $i$:

```math
\hat{r}_{\mathrm{new},i}
=
\mu
+
b_{\mathrm{new}}
+
b_i
+
q_i^T p_{\mathrm{new}}
```

Candidate movies can then be ranked according to their predicted ratings.

This enables immediate personalized recommendations **without retraining the entire matrix factorization model**.

---

## 🛠️ Project Structure

```text
movielens-recommender-system/
│
├── data/                   # Auto-downloaded MovieLens dataset
│
├── model/
│   └── recsys_model.pkl    # Serialized model factors and catalog mappings
│
├── download_and_train.py   # Data ingestion and SGD training pipeline
├── app.py                  # Streamlit application and real-time user inference
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚡ Getting Started Locally

### 1. Clone the Repository

```bash
git clone https://github.com/ashwinsashikumar/movielens-recommender-system.git
cd movielens-recommender-system
```

### 2. Set Up a Virtual Environment

#### Linux / WSL / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

If the trained model artifact is not already included in the repository, run:

```bash
python download_and_train.py
```

The training pipeline:

1. Downloads the MovieLens Latest-Small dataset.
2. Loads and preprocesses the rating data.
3. Initializes the user and movie latent factors.
4. Trains the biased matrix factorization model using SGD.
5. Stores the learned parameters and movie mappings.

The resulting model artifact is saved to:

```text
model/recsys_model.pkl
```

### 5. Run the Streamlit Interface

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

in your browser.

---

## 🔄 System Workflow

```text
MovieLens Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Biased Matrix Factorization
       │
       ▼
SGD Training
       │
       ▼
Learned Parameters
(μ, bu, bi, P, Q)
       │
       ▼
Serialized Model
       │
       ▼
Streamlit Application
       │
       ▼
New User Rates Movies
       │
       ▼
Estimate b_new
       │
       ▼
Solve for p_new
       │
       ▼
Score Candidate Movies
       │
       ▼
Rank Predicted Ratings
       │
       ▼
Top-N Recommendations
```

---

## 🎯 Key Features

- **Biased Matrix Factorization** — models global, user, and movie baseline effects alongside latent user-item interactions.

- **Latent Preference Learning** — automatically discovers hidden preference dimensions from user-rating patterns.

- **L2 Regularization** — reduces overfitting in the sparse user-item rating matrix.

- **SGD Training** — learns user factors, movie factors, and baseline biases directly from observed ratings.

- **Real-Time Cold-Start Inference** — estimates preferences for previously unseen users without retraining the complete model.

- **Closed-Form User Factor Estimation** — uses regularized least squares to infer a new user's latent representation from a small number of ratings.

- **Transparent Predictions** — predicted ratings can be decomposed into global, user-bias, movie-bias, and latent-interaction components.

- **Natural Movie Search** — users can search for and select movies using human-readable titles.

- **Interactive Interface** — Streamlit provides a lightweight frontend for rating movies and receiving personalized recommendations.

- **Persistent Model Artifacts** — learned factors, biases, and catalog mappings are serialized for fast application startup.

---

## 📊 Tech Stack

| Component | Technology |
| --- | --- |
| Recommendation Algorithm | Regularized Biased Matrix Factorization |
| Optimization | Stochastic Gradient Descent (SGD) |
| Cold-Start Inference | Ridge-Regularized Least Squares |
| Language | Python 3.10+ |
| Numerical Processing | NumPy, Pandas, SciPy |
| Web Interface | Streamlit |
| Dataset | MovieLens Latest-Small |
| Model Persistence | Pickle |

---

## 📚 Dataset

This project uses the **MovieLens Latest-Small** dataset provided by GroupLens Research.

The dataset contains approximately:

- **100,000 ratings**
- **9,700 movies**
- **600 users**

Ratings are provided on a **0.5–5.0 star scale**.

The dataset is automatically downloaded by the training pipeline and therefore does not need to be manually included in the repository.

---

## 💡 Recommendation Pipeline

At inference time, the system follows two different paths depending on whether the user was present during model training.

### Existing Users

For users already represented in the training data, predictions are obtained directly from the learned parameters:

```math
\hat{r}_{ui}
=
\mu + b_u + b_i + q_i^T p_u
```

### New Users

For previously unseen users:

1. The user rates a small collection of movies.
2. The system estimates the user's baseline bias $b_{\mathrm{new}}$.
3. The user's latent vector $p_{\mathrm{new}}$ is inferred using regularized least squares.
4. Ratings are predicted for candidate movies.
5. Movies already rated by the user are excluded.
6. Remaining movies are ranked by predicted rating.
7. The highest-ranked movies are returned as personalized recommendations.

This design separates expensive **offline model training** from lightweight **online user inference**, allowing new users to receive recommendations almost immediately.

---

## 📄 License

This project is intended for educational and portfolio purposes.

The MovieLens dataset is provided by **GroupLens Research** and is subject to its own terms of use.
