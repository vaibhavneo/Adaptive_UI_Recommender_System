# train.py
# Optional: one-shot to ensure seed data and quick fit (sanity check).
from recommender import HybridRecommender, HybridConfig

if __name__ == "__main__":
    model = HybridRecommender(HybridConfig())
    cat, inter = model.load_data()
    model.fit(cat, inter)
    print("Fitted. Try a few calls:")
    print("Recommend for user 1:", model.recommend("1", k=5))
    print("Similar to 101:", model.similar_items(101, k=5))
