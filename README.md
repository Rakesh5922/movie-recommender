# CineMatch — Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> A content-based movie recommendation system built on the 
> TMDB 5000 dataset — deployed as a live interactive web app!

##  Live Demo
(https://movie-recommender-nngwuk6maziystb4wgrkny.streamlit.app)**

## 📸 Screenshots

---

##  How It Works
```
User selects a movie
        ↓
App finds movie's "DNA" tags
(genres + cast + director + keywords + overview)
        ↓
TF-IDF converts tags to vectors
        ↓
Cosine Similarity finds closest movies
        ↓
Top 10 recommendations displayed with posters!
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data processing |
| Scikit-learn | TF-IDF & Cosine Similarity |
| NLTK | Stemming |
| Streamlit | Web app framework |
| OMDb API | Movie poster fetching |

---

##  Dataset

- **Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Size:** 4,800 movies
- **Features used:** Genres, Cast, Crew, Keywords, Overview

---

##  Run Locally
```bash
# Clone the repo
git clone https://github.com/Rakesh599/movie-recommender.git

# Go into the folder
cd movie-recommender

# Install libraries
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Project Structure
```
movie-recommender/
│
├── app.py                 ← Streamlit web app
├── notebook.ipynb         ← Full EDA + Model notebook
├── tmdb_features.csv      ← Engineered features
├── tmdb_preprocessed.csv  ← Cleaned data
├── cosine_sim.pkl         ← Similarity matrix
├── requirements.txt       ← Dependencies
└── README.md              ← You are here!
```
---

## Key Learnings

- Real-world JSON parsing from CSV columns
- TF-IDF vectorization for text similarity
- Cosine similarity for content-based filtering
- IMDB weighted rating formula
- End-to-end ML project deployment

---
##  Author

**YOUR NAME**
- GitHub: https://github.com/Rakesh599
- LinkedIn:http://linkedin.in/rakesh-vallepu
