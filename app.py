import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv',nrows=500000)
links = pd.read_csv('data/links.csv')
movies = pd.merge(movies, links[['movieId', 'tmdbId']], on='movieId', how='left')
ratings['user_idx'] = ratings['userId'].astype('category').cat.codes
ratings['movie_idx'] = ratings['movieId'].astype('category').cat.codes
movie_id_to_idx = dict(zip(ratings['movieId'], ratings['movie_idx']))
movie_idx_to_id = dict(zip(ratings['movie_idx'], ratings['movieId']))
sparse_user_item = csr_matrix(
    (ratings['rating'], (ratings['user_idx'], ratings['movie_idx'])),
    shape=(ratings['user_idx'].nunique(), ratings['movie_idx'].nunique())
)
svd = TruncatedSVD(n_components=50, random_state=42)
movie_latent_matrix = svd.fit_transform(sparse_user_item.T)
item_similarities = cosine_similarity(movie_latent_matrix)
def get_recommendations(movie_id, top_n=5):
    if movie_id not in movie_id_to_idx:
        return "Movie not found in ratings."
    idx = movie_id_to_idx[movie_id]
    sim_scores = list(enumerate(item_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    top_scores = [i[1] for i in sim_scores[1:top_n+1]]
    rec_movie_ids = [movie_idx_to_id[i] for i in top_indices]
    recommendations = movies[movies['movieId'].isin(rec_movie_ids)].copy()
    recommendations['similarity_score'] = top_scores
    movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
    return movie_title,recommendations[['movieId', 'title', 'tmdbId']]
from flask import Flask, render_template, request, jsonify
import difflib
app = Flask(__name__)
def get_api_recommendations(movie_id, top_n=5):
    if movie_id not in movie_id_to_idx:
        return None, None
    idx = movie_id_to_idx[movie_id]
    sim_scores = list(enumerate(item_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    top_scores = [i[1] for i in sim_scores[1:top_n+1]]
    rec_movie_ids = [movie_idx_to_id[i] for i in top_indices]
    recommendations = movies[movies['movieId'].isin(rec_movie_ids)].copy()
    recommendations['similarity_score'] = top_scores
    movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
    return movie_title, recommendations
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    search_query = data.get('movie_title', '').strip()
    all_titles = movies['title'].tolist()
    closest_matches = difflib.get_close_matches(search_query, all_titles, n=1, cutoff=0.4)
    if not closest_matches:
        return jsonify({"error": "Movie not found in the database. Try another title!"}), 404
    matched_title = closest_matches[0]
    movie_id = movies[movies['title'] == matched_title]['movieId'].iloc[0]
    movie_title, rec_df = get_api_recommendations(movie_id, top_n=6)
    if rec_df is None:
        return jsonify({"error": "Not enough rating data to recommend for this movie."}), 404
    results = rec_df[['movieId', 'title', 'tmdbId', 'similarity_score']].to_dict(orient='records')
    return jsonify({
        "searched_for": movie_title,
        "recommendations": results
    })
if __name__ == '__main__':
    print("Starting AI Recommendation Server...")
    app.run(debug=True, port=5000)