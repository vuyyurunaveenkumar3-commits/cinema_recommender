# 🎬 Cinema Recommender — Hollywood × Tollywood (MovieLens)

A complete movie recommendation system on the **MovieLens format**, extended with a
**Tollywood (Telugu) catalogue** of 83 films (Maya Bazaar → Baahubali → RRR → Pushpa 2 → Kalki 2898 AD)
alongside 93 Hollywood/world classics.

## ✨ Features
| Feature | Where |
|---|---|
| Content-based filtering (genres + **language + director + decade**, TF-IDF) | `src/content_based.py` |
| User-based, item-based **KNN** and **SVD** collaborative filtering | `src/collaborative.py` |
| **Hybrid** recommender (CF + content) | `hybrid_recommend` |
| **Language filter** (e.g. only Telugu results) | `--language Telugu` |
| **Explainable** picks — "because you liked …" | `--explain` |
| **Cold-start** recommendations for brand-new users | `newuser` command |
| Popularity baseline + RMSE / MAE / Precision@K / Recall@K + **per-language RMSE** | `src/evaluate.py` |
| 5 auto-generated charts | `src/visualize.py` → `results/` |
| Streamlit web app with 4 tabs | `app.py` |
| Unit tests (8) | `tests/` |
| Report template for submission | `docs/REPORT.md` |

## 📁 Structure
```
cinema_recommender/
├── main.py                    # CLI
├── app.py                     # Streamlit UI
├── requirements.txt
├── data/
│   ├── movies.csv             # movieId,title,genres,language,director   (176 movies)
│   ├── ratings.csv            # userId,movieId,rating,timestamp          (SYNTHETIC)
│   ├── catalogue.txt          # editable source list - add your own movies here!
│   ├── generate_sample_data.py
│   └── download_data.py       # real MovieLens + Telugu merge (optional)
├── src/                       # data_loader, content_based, collaborative, baselines, evaluate, visualize
├── tests/test_recommender.py
├── docs/REPORT.md
└── results/                   # evaluation.csv + PNG charts
```

## 🚀 Quick start
```bash
pip install -r requirements.txt

python main.py similar   --title "RRR"
python main.py recommend --user 3 --method hybrid --language Telugu --explain
python main.py recommend --user 3 --method svd
python main.py newuser   --genres Action,Drama --languages Telugu
python main.py telugu    -n 10
python main.py evaluate
python main.py plots
streamlit run app.py
python -m unittest discover tests -v
```

## ⚠️ About the data (please read)
* **Movies** — titles, genres, languages, directors and years are real.
* **Ratings are SYNTHETIC.** The official MovieLens dataset contains no Telugu
  ratings, so `generate_sample_data.py` simulates 400 users (≈45% Telugu fans,
  35% mixed, 20% English-only) with genre tastes. Results therefore show that the
  *pipeline works*, not real audience opinion. State this in your report.
* **Want real MovieLens ratings?** Run `python data/download_data.py` (needs internet).
  It downloads `ml-latest-small`, appends the Telugu films as new movie IDs and simulates
  ratings for them only (original files are backed up as `*_sample.csv`).
* **Add movies:** append a line to `data/catalogue.txt` (`title ;; genres ;; language ;; director`)
  and re-run `python data/generate_sample_data.py`.

## 🧠 How it works
**Content-based** — each movie → tokens: genres (×2 weight), language, director, decade → TF-IDF →
cosine similarity. *RRR* → *Baahubali* because of shared director, genre and language.
A user profile = feature vectors of rated movies weighted by (rating − user mean).

**Collaborative** — user × movie matrix, mean-centred per user:
user-KNN (k=30 similar users), item-KNN (k=30 similar movies), SVD (20 latent factors).

**Hybrid** — `0.6 · CF + 0.4 · content` (min-max scaled).

**Cold start** — weighted (Bayesian) popularity × overlap with chosen genres/languages.

## 📊 Sample results (80/20 split, synthetic data)
| Model | RMSE | MAE | Precision@10 | Recall@10 |
|---|---|---|---|---|
| Popularity baseline | 0.751 | 0.597 | 0.080 | 0.200 |
| User-based CF | 0.633 | 0.500 | 0.065 | 0.170 |
| Item-based CF | 0.636 | 0.506 | 0.093 | 0.221 |
| SVD | 0.683 | 0.546 | 0.118 | 0.283 |

CF models beat the popularity baseline on rating accuracy (RMSE); SVD ranks best (Precision/Recall@10).

## 🔭 Ideas to extend
Hyper-parameter tuning · add cast/music-director features · implicit-feedback ALS ·
TMDB posters via API · Docker / deployment on Streamlit Cloud
