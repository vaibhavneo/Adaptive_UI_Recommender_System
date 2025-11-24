# serve.py
# FastAPI service exposing:
#   GET  /recommend?user=<id>&k=<k>   -> [item_ids]
#   GET  /similar?item=<id>&k=<k>     -> [item_ids]
#   POST /feedback {user_id,item_id}  -> append interaction, (optionally) fast refit
#   POST /reload                       -> hard refit from CSVs

from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
from recommender import HybridRecommender, HybridConfig, DATA_DIR, CATALOG_CSV, INTERACTIONS_CSV

app = FastAPI(title="Hybrid Recommender API")

# Global model
cfg = HybridConfig(alpha=0.6, mmr_lambda=0.2, k_neighbors=50)
model = HybridRecommender(cfg)
cat_df, inter_df = model.load_data()
model.fit(cat_df, inter_df)

class FeedbackIn(BaseModel):
    user_id: str
    item_id: int
    timestamp: int | None = None

@app.get("/recommend")
def recommend(user: str = Query(default="1"), k: int = Query(default=5, ge=1, le=50)):
    recs = model.recommend(user, k=k)
    return JSONResponse(content=recs)

@app.get("/similar")
def similar(item: int = Query(...), k: int = Query(default=5, ge=1, le=50)):
    recs = model.similar_items(item, k=k)
    return JSONResponse(content=recs)

@app.post("/feedback")
def feedback(fb: FeedbackIn):
    # append to interactions.csv
    DATA_DIR.mkdir(exist_ok=True)
    if INTERACTIONS_CSV.exists():
        df = pd.read_csv(INTERACTIONS_CSV)
    else:
        df = pd.DataFrame(columns=["user_id", "item_id", "timestamp"])
    row = {"user_id": fb.user_id, "item_id": fb.item_id, "timestamp": fb.timestamp or (df["timestamp"].max() + 1 if len(df) else 1)}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(INTERACTIONS_CSV, index=False)
    # quick refit (fast for small data)
    cat_df = pd.read_csv(CATALOG_CSV)
    inter_df = pd.read_csv(INTERACTIONS_CSV)
    model.fit(cat_df, inter_df)
    return JSONResponse(content={"ok": True})

@app.post("/reload")
def reload_model():
    # hard refit (useful if you edited CSVs by hand)
    cat_df = pd.read_csv(CATALOG_CSV)
    inter_df = pd.read_csv(INTERACTIONS_CSV)
    model.fit(cat_df, inter_df)
    return JSONResponse(content={"ok": True})
