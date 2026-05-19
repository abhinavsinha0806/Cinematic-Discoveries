# Cinematic Discoveries

Cinematic Discoveries is a full-stack, collaborative filtering movie recommendation engine. Built to handle large-scale data processing, this application bridges the gap between raw data analytics and a user-facing web environment, translating millions of user rating patterns into highly accurate, personalized film suggestions.

## System Architecture

The project is structured into two distinct layers: a robust analytical backend and a responsive, modern frontend.

### 1. Machine Learning & Backend (Python, Flask, Scikit-Learn)
The backend architecture is designed to ingest and process the MovieLens 32M dataset efficiently. 
* **Data Ingestion & Transformation:** Raw CSV data is loaded via Pandas and transformed into a highly sparse user-item matrix using `scipy.sparse.csr_matrix`. This prevents memory overflow when handling the 32 million rows of user ratings.
* **Dimensionality Reduction:** To identify latent viewing patterns, the sparse matrix is processed using Truncated Singular Value Decomposition (Truncated SVD). The model compresses the data into 50 principal components, isolating the most significant variance in user preferences.
* **Similarity Computation:** The engine calculates the pairwise cosine similarity across the reduced matrix, establishing a mathematical proximity score between any two films in the database.
* **Fuzzy String Matching:** The Flask API utilizes `difflib` to ensure user inputs do not require exact string matches, providing a fault-tolerant search experience.

### 2. Frontend (HTML5, Tailwind CSS)
The user interface features a glassmorphism design system to provide a premium, cinematic feel. It communicates asynchronously with the Flask backend via Fetch API, dynamically rendering similarity scores and progress bars without requiring page reloads.

## Dataset Handling

This model is trained on the [MovieLens 32M dataset](https://grouplens.org/datasets/movielens/32m/), containing 32 million ratings and over 71,000 movies. 
