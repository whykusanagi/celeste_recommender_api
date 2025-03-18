from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import difflib
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Celeste Recommender API")

# Load models and data
DATA_PATH = "umap_hitomi/"
df = pd.read_parquet(f"{DATA_PATH}doujin_metadata.parquet")
vectorizer = joblib.load(f"{DATA_PATH}tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load(f"{DATA_PATH}tfidf_matrix.pkl")
semantic_embeddings = np.load(f"{DATA_PATH}semantic_embeddings.npy")
faiss_index = faiss.read_index(f"{DATA_PATH}faiss_semantic.index")
model = SentenceTransformer('all-MiniLM-L6-v2')

class RecommendationRequest(BaseModel):
    query: str
    top_n: int = 5
    discoverability: bool = False

@app.post("/recommend_by_terms")
def recommend_by_terms(req: RecommendationRequest):
    input_vector = vectorizer.transform([req.query])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-req.top_n:][::-1]
    results = df.iloc[top_indices][['title', 'artist', 'tags', 'link', 'rank']].to_dict(orient='records')
    return {"results": results}

@app.post("/recommend_by_title")
def recommend_by_title(req: RecommendationRequest):
    closest_title = difflib.get_close_matches(req.query, df['title'].tolist(), n=1)
    if not closest_title:
        return {"results": []}

    target_row = df[df['title'] == closest_title[0]].iloc[0]
    input_vector = vectorizer.transform([target_row['combined_text']])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    candidate_ranks = df['rank'].values
    rank_diff = np.abs(candidate_ranks - target_row['rank'])
    rank_proximity_score = 1 / (1 + rank_diff)

    if req.discoverability:
        score = similarities * (1 + (rank_diff / rank_diff.max()))
    else:
        score = (similarities * 0.7) + (rank_proximity_score * 0.3)

    top_indices = score.argsort()[-(req.top_n + 1):][::-1]
    recommended = df.iloc[top_indices]
    recommended = recommended[recommended['title'] != target_row['title']]
    results = recommended[['title', 'artist', 'tags', 'link', 'rank']].to_dict(orient='records')
    return {"results": results}

@app.post("/recommend_semantic")
def recommend_semantic(req: RecommendationRequest):
    query_embedding = model.encode([req.query])
    faiss.normalize_L2(query_embedding)
    distances, indices = faiss_index.search(query_embedding, req.top_n)
    results = df.iloc[indices[0]][['title', 'artist', 'tags', 'link', 'rank']].to_dict(orient='records')
    return {"results": results}

@app.get("/")
def health():
    return {"status": "✅ Celeste recommender API running"}
