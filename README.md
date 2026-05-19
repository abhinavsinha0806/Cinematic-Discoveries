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

**Important:** Due to repository size constraints, the dataset is uploaded please access the dataset on the official website.

## Local Environment Setup

Ensure you have Python 3.8+ installed before proceeding.

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/abhinavsinha0806/cinematic-discoveries.git](https://github.com/abhinavsinha0806/cinematic-discoveries.git)
   cd cinematic-discoveries
   ```

2. **Initialize the Environment**
   Install the necessary computational and web dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Data Pipeline**
   * Download the `ml-32m.zip` dataset directly from [GroupLens](https://grouplens.org/datasets/movielens/32m/).
   * Extract the archive and create a new directory named `data` in the root folder of this project.
   * Move `movies.csv`, `ratings.csv`, and `links.csv` into the `data` directory.
   * Verify that the pandas `read_csv` functions in `app.py` point to `data/movies.csv`, etc.

4. **Launch the Application**
   ```bash
   python app.py
   ```
   The Flask server will initialize the data structures and compute the matrix. Once complete, access the interface at `http://127.0.0.1:5000/`.

## Authorship & Development

**Developed by Abhinav Sinha**

* **Data Science & Architecture:** The mathematical modeling, Truncated SVD implementation, data pipeline engineering, and backend Flask architecture are entirely my original work, developed from scratch to demonstrate practical applications of machine learning.
* **UI Development:** The conceptual design, layout, and user experience strategy are my own. I leveraged Google's Gemini as a rapid-prototyping coding assistant to translate my design requirements into the final Tailwind CSS utility classes and HTML structure.
