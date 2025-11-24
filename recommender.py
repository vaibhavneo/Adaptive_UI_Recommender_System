# recommender.py
# A lightweight hybrid recommender:
# - Item-item co-occurrence cosine from implicit interactions
# - Content TF-IDF cosine from title+tags+description
# - Popularity backfill for cold-start
# - Optional MMR diversity
#
# Data format:
#   data/catalog.csv:      item_id,title,tags,description
#   data/interactions.csv: user_id,item_id,timestamp

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_DIR = Path("data")
CATALOG_CSV = DATA_DIR / "catalog.csv"
INTERACTIONS_CSV = DATA_DIR / "interactions.csv"

def _ensure_seed_data():
    DATA_DIR.mkdir(exist_ok=True)
    if not CATALOG_CSV.exists():
        df = pd.DataFrame([
            {"item_id": 101, "title": "Inception",         "tags": "sci-fi dream nolan",    "description": "A mind-bending heist across layered dreams."},
            {"item_id": 102, "title": "The Matrix",        "tags": "sci-fi cyberpunk",      "description": "A hacker discovers the truth about reality."},
            {"item_id": 103, "title": "Interstellar",      "tags": "sci-fi space nolan",    "description": "Explorers travel through a wormhole in space."},
            {"item_id": 104, "title": "Arrival",           "tags": "sci-fi language",       "description": "A linguist communicates with alien visitors."},
            {"item_id": 105, "title": "Blade Runner 2049", "tags": "sci-fi noir cyberpunk", "description": "A blade runner unearths a long-buried secret."},
            {"item_id": 106, "title": "Gravity",           "tags": "space survival",        "description": "An astronaut fights to survive in orbit."},
            {"item_id": 107, "title": "The Martian",       "tags": "space survival",        "description": "An astronaut is stranded on Mars."},
        ])
        df.to_csv(CATALOG_CSV, index=False)
    if not INTERACTIONS_CSV.exists():
        df = pd.DataFrame([
            # simple toy history
            {"user_id": "1", "item_id": 101, "timestamp": 1},
            {"user_id": "1", "item_id": 103, "timestamp": 2},
            {"user_id": "2", "item_id": 102, "timestamp": 1},
            {"user_id": "2", "item_id": 105, "timestamp": 2},
            {"user_id": "3", "item_id": 107, "timestamp": 1},
        ])
        df.to_csv(INTERACTIONS_CSV, index=False)

@dataclass
class HybridConfig:
    alpha: float = 0.6      # weight for collaborative vs content (collab_weight = alpha)
    mmr_lambda: float = 0.2 # diversity; 0 disables MMR
    k_neighbors: int = 50   # neighbors to consider in item-item similarity
    min_interactions: int = 1  # min items to consider a user “warm”
    random_state: int = 42

class HybridRecommender:
    def __init__(self, config: HybridConfig = HybridConfig()):
        self.cfg = config
        self.item_index: Dict[int, int] = {}
        self.index_item: List[int] = []
        self.user_hist: Dict[str, List[int]] = {}
        self.popular: List[int] = []
        self.item_sim_collab: Optional[np.ndarray] = None
        self.item_sim_content: Optional[np.ndarray] = None

    def fit(self, catalog: pd.DataFrame, interactions: pd.DataFrame):
        # Index items
        items = catalog["item_id"].astype(int).tolist()
        self.index_item = items
        self.item_index = {iid: i for i, iid in enumerate(items)}
        n_items = len(items)

        # Popularity for backfill
        pop = interactions["item_id"].value_counts().reindex(items).fillna(0)
        self.popular = [int(i) for i in pop.sort_values(ascending=False).index.tolist()]

        # User history
        self.user_hist = (
            interactions
            .groupby("user_id")["item_id"]
            .apply(lambda s: [int(x) for x in s.tolist()])
            .to_dict()
        )

        # Build content matrix (TF-IDF on title+tags+description)
        corpus = (
            catalog["title"].fillna("") + " " +
            catalog["tags"].fillna("") + " " +
            catalog["description"].fillna("")
        ).tolist()
        vect = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X = vect.fit_transform(corpus)               # (n_items, n_terms)
        X = normalize(X, norm="l2", axis=1)
        # cosine similarity via dot product on normalized rows:
        self.item_sim_content = (X * X.T).toarray()  # dense for small demo

        # Collaborative: item-item co-occurrence cosine
        # Build a sparse user->items incidence then compute item co-occurrence
        user_to_items = {}
        for u, grp in interactions.groupby("user_id"):
            user_to_items[u] = [self.item_index.get(int(i)) for i in grp["item_id"] if int(i) in self.item_index]

        # item-item counts
        co_mat = np.zeros((n_items, n_items), dtype=np.float32)
        for indices in user_to_items.values():
            if not indices: 
                continue
            uniq = np.unique(indices)
            co_mat[np.ix_(uniq, uniq)] += 1.0

        # remove self counts (diagonal can be number of users; set to 0 for cosine)
        np.fill_diagonal(co_mat, 0.0)

        # cosine normalization: cos(i,j) = co_ij / sqrt(co_i * co_j)
        row_sum = np.sqrt(np.maximum(co_mat.sum(axis=1), 1e-8))
        denom = np.outer(row_sum, row_sum)
        sim = np.divide(co_mat, denom, out=np.zeros_like(co_mat), where=denom > 0)
        # keep only top-k neighbors for each item to reduce noise
        k = min(self.cfg.k_neighbors, n_items-1)
        for i in range(n_items):
            # zero-out all but top-k
            idx = np.argpartition(sim[i], -(k))[:-(k)]
            sim[i, idx] = 0.0
        self.item_sim_collab = sim

    def _score_user(self, user_id: str) -> np.ndarray:
        """
        Hybrid score for all items for a given user:
        - aggregate collaborative neighbors from user's history
        - blend with content similarity (mean over history)
        """
        n = len(self.index_item)
        s_collab = np.zeros(n, dtype=np.float32)
        s_content = np.zeros(n, dtype=np.float32)

        hist_items = self.user_hist.get(user_id, [])
        hist_idx = [self.item_index[i] for i in hist_items if i in self.item_index]
        if hist_idx:
            # Collaborative: sum similarities from history items
            s_collab = self.item_sim_collab[hist_idx].sum(axis=0)
            # Content: mean similarities from history items
            s_content = self.item_sim_content[hist_idx].mean(axis=0)
            # zero scores for already-seen
            s_collab[hist_idx] = 0.0
            s_content[hist_idx] = 0.0
        else:
            # cold user: zeros (backfill by popularity)
            pass

        # Blend
        alpha = self.cfg.alpha
        s = alpha * s_collab + (1 - alpha) * s_content
        return s

    def _mmr(self, scores: np.ndarray, k: int, sim_mat: np.ndarray, lambda_: float) -> List[int]:
        """
        Simple MMR (maximal marginal relevance) selection to improve diversity.
        """
        chosen = []
        cand = set(range(len(scores)))
        for _ in range(k):
            if not cand:
                break
            if not chosen:
                i = int(np.argmax(scores))
                chosen.append(i)
                cand.remove(i)
                continue
            # score_i - lambda * max_sim_to_chosen
            max_sim = np.max(sim_mat[np.ix_(list(cand), chosen)], axis=1) if chosen else np.zeros(len(cand))
            max_sim = np.array(max_sim).reshape(-1)
            cand_list = list(cand)
            mmr_scores = scores[cand_list] - lambda_ * max_sim
            j = int(np.argmax(mmr_scores))
            i = cand_list[j]
            chosen.append(i)
            cand.remove(i)
        return chosen

    def recommend(self, user_id: str, k: int = 5) -> List[int]:
        # warm user
        scores = self._score_user(user_id)
        if scores.sum() > 0:
            # take top-k with optional diversity
            if self.cfg.mmr_lambda > 0:
                top = self._mmr(scores, k, self.item_sim_content, self.cfg.mmr_lambda)
                return [self.index_item[i] for i in top]
            else:
                idx = np.argsort(-scores)[:k]
                return [self.index_item[i] for i in idx]
        # cold user: popularity
        return self.popular[:k]

    def similar_items(self, item_id: int, k: int = 5) -> List[int]:
        if item_id not in self.item_index:
            return self.popular[:k]
        idx = self.item_index[item_id]
        # combine collab + content for item–item
        s = self.cfg.alpha * self.item_sim_collab[idx] + (1 - self.cfg.alpha) * self.item_sim_content[idx]
        s[idx] = -1e9
        top = np.argsort(-s)[:k]
        return [self.index_item[i] for i in top]

    @staticmethod
    def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
        _ensure_seed_data()
        cat = pd.read_csv(CATALOG_CSV)
        inter = pd.read_csv(INTERACTIONS_CSV)
        return cat, inter
