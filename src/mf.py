"""Matrix factorization with SGD, the same model as recommender.ipynb, as functions the app can reuse."""
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ml-100k"


def load_data():
    # MovieLens can't be redistributed, so download it from GroupLens on first run
    if not (DATA / "u.data").exists():
        urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-100k.zip", ROOT / "ml-100k.zip")
        zipfile.ZipFile(ROOT / "ml-100k.zip").extractall(ROOT)
    data = np.loadtxt(DATA / "u.data", dtype=int)[:, :3]
    data[:, :2] -= 1  # ids start at 0
    titles = pd.read_csv(DATA / "u.item", sep="|", encoding="latin-1", header=None, usecols=[1])[1].tolist()
    return data, titles


def split(data, seed=42):
    idx = np.random.default_rng(seed).permutation(len(data))
    cut = int(0.8 * len(data))
    return data[idx[:cut]], data[idx[cut:]]


def rmse(model, ratings):
    P, Q, bu, bi, mu = model
    u, i, r = ratings[:, 0], ratings[:, 1], ratings[:, 2]
    pred = np.clip(mu + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1), 1, 5)
    return float(np.sqrt(np.mean((r - pred) ** 2)))


def train_mf(train, n_users, n_items, k=50, lr=0.01, reg=0.05, epochs=40, seed=0):
    r = np.random.default_rng(seed)
    P = r.normal(0, 0.1, (n_users, k))
    Q = r.normal(0, 0.1, (n_items, k))
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    mu = train[:, 2].mean()
    for _ in range(epochs):
        for u, i, rating in train[r.permutation(len(train))]:
            err = rating - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            P[u], Q[i] = P[u] + lr * (err * Q[i] - reg * P[u]), Q[i] + lr * (err * P[u] - reg * Q[i])
    return P, Q, bu, bi, mu
