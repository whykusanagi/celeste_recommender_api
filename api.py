from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Celeste Recommender API")

# Load models and data
data_path = "umap_hitomi/"
required_files = [
    "doujin_metadata.parquet",
    "tfidf_vectorizer.pkl",
    "tfidf_matrix.pkl",
    "semantic_embeddings.npy",
    "faiss_semantic.index"
]

missing = [f for f in required_files if not os.path.isfile(f'{data_path}{f}')]
if missing:
    logging.error(f"Missing required files in {data_path}: {missing}")
    raise FileNotFoundError(f"Missing required files in {data_path}: {missing}")

logging.info("Loading models and data...")
df = pd.read_parquet(f"{data_path}doujin_metadata.parquet")
vectorizer = joblib.load(f"{data_path}tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load(f"{data_path}tfidf_matrix.pkl")
semantic_embeddings = np.load(f"{data_path}semantic_embeddings.npy")
faiss_index = faiss.read_index(f"{data_path}faiss_semantic.index")
model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("Models and data loaded successfully.")

class RecommendationRequest(BaseModel):
    query: str
    top_n: int = 5
    discoverability: bool = False

@app.post("/recommend_by_terms")
def recommend_by_terms(req: RecommendationRequest):
    logging.info(f"Received recommend_by_terms query: {req.query}")
    input_vector = vectorizer.transform([req.query])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-req.top_n:][::-1]
    results = df.iloc[top_indices][['title', 'artist', 'tags', 'link', 'rank']].copy()
    results['tags'] = results['tags'].apply(lambda x: [str(tag) for tag in x])
    results = results.to_dict(orient='records')
    logging.info(f"Returning {len(results)} results for terms query.")
    return {"results": results}

@app.post("/recommend_by_title")
def recommend_by_title(req: RecommendationRequest):
    logging.info(f"Received recommend_by_title query: {req.query}, discoverability={req.discoverability}")
    closest_title = difflib.get_close_matches(req.query, df['title'].tolist(), n=1)
    if not closest_title:
        logging.warning(f"No close match found for title: {req.query}")
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
    results = recommended[['title', 'artist', 'tags', 'link', 'rank']].copy()
    results['tags'] = results['tags'].apply(lambda x: [str(tag) for tag in x])
    results = results.to_dict(orient='records')
    logging.info(f"Returning {len(results)} recommendations based on title: {closest_title[0]}")
    return {"results": results}

@app.post("/recommend_semantic")
def recommend_semantic(req: RecommendationRequest):
    logging.info(f"Received recommend_semantic query: {req.query}")
    query_embedding = model.encode([req.query])
    faiss.normalize_L2(query_embedding)
    distances, indices = faiss_index.search(query_embedding, req.top_n)
    results = df.iloc[indices[0]][['title', 'artist', 'tags', 'link', 'rank']].copy()
    results['tags'] = results['tags'].apply(lambda x: [str(tag) for tag in x])
    results = results.to_dict(orient='records')
    logging.info(f"Returning {len(results)} semantic recommendations.")
    return {"results": results}

@app.get("/")
def health():
    logging.info("Health check called.")
    return {"status": "✅ Celeste recommender API running"}