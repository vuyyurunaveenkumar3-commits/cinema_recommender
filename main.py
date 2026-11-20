"""Cinema Recommender (Hollywood + Tollywood) - command line interface.

python main.py similar   --title "RRR"                       # content-based
python main.py recommend --user 5 --method hybrid --language Telugu
python main.py recommend --user 5 --method item --explain    # 'because you liked...'
python main.py newuser   --genres Action,Drama --languages Telugu
python main.py telugu    -n 10                               # top Telugu movies
python main.py evaluate                                      # compare all models
python main.py plots                                         # save charts to results/
"""
import argparse
import os
import pandas as pd

from src.data_loader import load_data, train_test_split_ratings, find_movie, ids_for_language
from src.content_based import ContentBasedRecommender
from src.collaborative import UserBasedCF, ItemBasedCF, SVDRecommender, hybrid_recommend
from src.baselines import PopularityRecommender, weighted_popularity, new_user_recommend
from src.evaluate import compare_models, rmse_by_language
from src import visualize

pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 55)


def build_cf(method, ratings):
    cls = {"user": UserBasedCF, "item": ItemBasedCF, "svd": SVDRecommender}
    return cls[method]().fit(ratings)


def show(df):
    print(df.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Tollywood movie recommender")
    p.add_argument("command", choices=["similar", "recommend", "newuser", "telugu", "evaluate", "plots"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--title")
    p.add_argument("--user", type=int)
    p.add_argument("--method", default="item", choices=["content", "user", "item", "svd", "hybrid"])
    p.add_argument("--language", default=None, help="restrict results, e.g. Telugu or English")
    p.add_argument("--genres", default="", help="comma list for `newuser`, e.g. Action,Drama")
    p.add_argument("--languages", default="", help="comma list for `newuser`, e.g. Telugu,English")
    p.add_argument("--explain", action="store_true", help="show why (item-based CF only)")
    p.add_argument("-n", type=int, default=10)
    args = p.parse_args()

    movies, ratings = load_data(args.data_dir)
    n_te = (movies["language"] == "Telugu").sum()
    print(f"Loaded {len(movies)} movies ({n_te} Telugu), {ratings['userId'].nunique()} users, "
          f"{len(ratings)} ratings\n")
    cb = ContentBasedRecommender(movies)
    allowed = ids_for_language(movies, args.language)
    title_of = movies.set_index("movieId")["title"]

    if args.command == "similar":
        if not args.title:
            p.error("--title is required")
        row = find_movie(movies, args.title)
        if row is None:
            raise SystemExit(f"No movie matching '{args.title}'")
        print(f"Movies similar to: {row['title']}  [{row['genres']} | {row['language']} | {row['director']}]\n")
        show(cb.similar_movies(row["movieId"], args.n, allowed))

    elif args.command == "recommend":
        if args.user is None:
            p.error("--user is required")
        liked = (ratings[(ratings.userId == args.user) & (ratings.rating >= 4)]
                 .merge(movies, on="movieId").sort_values("rating", ascending=False).head(5))
        print(f"User {args.user} top-rated movies:")
        show(liked[["title", "language", "rating"]])
        print()
        cf = None
        if args.method == "content":
            out = cb.recommend_for_user(args.user, ratings, args.n, allowed)
        elif args.method == "hybrid":
            cf = build_cf("item", ratings)
            out = hybrid_recommend(args.user, ratings, cf, cb, movies, args.n, allowed_ids=allowed)
        else:
            cf = build_cf(args.method, ratings)
            rec = cf.recommend(args.user, args.n, allowed)
            out = rec.merge(movies[["movieId", "title", "genres", "language"]], on="movieId")
        print(f"Top {args.n} recommendations ({args.method}" + (f", {args.language}" if args.language else "") + "):")
        show(out)
        if args.explain:
            item_cf = cf if isinstance(cf, ItemBasedCF) else build_cf("item", ratings)
            print("\nWhy these?")
            for mid, t in zip(out["movieId"], out["title"]):
                why = item_cf.explain(args.user, mid, ratings)
                if why:
                    print(f"  {t}  <-  because you liked: " + ", ".join(title_of[m] for m, _ in why))

    elif args.command == "newuser":
        genres = [g.strip() for g in args.genres.split(",") if g.strip()]
        langs = [l.strip() for l in args.languages.split(",") if l.strip()]
        print(f"Cold-start picks for genres={genres or 'any'} languages={langs or 'any'}:\n")
        show(new_user_recommend(movies, ratings, genres, langs, args.n))

    elif args.command == "telugu":
        df = weighted_popularity(movies, ratings)
        df = df[df.language == "Telugu"].sort_values("weighted_rating", ascending=False).head(args.n)
        print("Top Telugu movies (weighted rating):\n")
        show(df[["title", "genres", "director", "count", "weighted_rating"]])

    elif args.command == "evaluate":
        train, test = train_test_split_ratings(ratings, test_size=0.2)
        print(f"Train: {len(train)} ratings | Test: {len(test)} ratings\n")
        models = [PopularityRecommender().fit(train), UserBasedCF().fit(train),
                  ItemBasedCF().fit(train), SVDRecommender().fit(train)]
        table = compare_models(models, train, test, k=10)
        show(table)
        print("\nRMSE by language of the movie:")
        show(rmse_by_language(models, train, test, movies))
        os.makedirs("results", exist_ok=True)
        table.to_csv("results/evaluation.csv", index=False)
        visualize.model_comparison(table)
        print("\nSaved results/evaluation.csv and results/model_comparison.png")

    elif args.command == "plots":
        paths = [visualize.rating_distribution(movies, ratings), visualize.language_split(movies, ratings),
                 visualize.top_telugu(movies, ratings), visualize.genre_by_language(movies)]
        print("Saved:\n  " + "\n  ".join(paths))


if __name__ == "__main__":
    main()
