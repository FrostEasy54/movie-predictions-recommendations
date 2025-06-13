import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

# 1) Загружаем и предобрабатываем данные (как было)
df = pd.read_csv("imdb_top_1000.csv")
df["Released_Year"] = (
    pd.to_numeric(df["Released_Year"], errors="coerce").fillna(0).astype(int)
)
df["No_of_Votes"] = (
    df["No_of_Votes"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .replace("nan", "0")
    .astype(int)
)
df["Gross"] = pd.to_numeric(
    df["Gross"].astype(str).str.replace(",", "", regex=False).replace("nan", np.nan),
    errors="coerce",
).fillna(0.0)
df["Meta_score"] = df["Meta_score"].fillna(df["Meta_score"].median())
df["IMDB_Rating"] = df["IMDB_Rating"].fillna(df["IMDB_Rating"].mean())
df["Runtime"] = df["Runtime"].apply(
    lambda x: int(x.split()[0]) if isinstance(x, str) else 0
)

# 2) Кодируем жанры и актёров
df["Genre_List"] = df["Genre"].fillna("").str.split(",\s*")  # type: ignore
mlb_genre = MultiLabelBinarizer()
genre_encoded = mlb_genre.fit_transform(df["Genre_List"])

df["Stars_List"] = df[["Star1", "Star2", "Star3", "Star4"]].fillna("").values.tolist()
mlb_stars = MultiLabelBinarizer()
stars_encoded = mlb_stars.fit_transform(df["Stars_List"])

# 3) Оцениваем режиссёров
director_means = df.groupby("Director")["IMDB_Rating"].mean().to_dict()
global_mean = df["IMDB_Rating"].mean()
df["Director_Score"] = df["Director"].map(director_means).fillna(global_mean)

# 4) Составляем матрицу признаков и целевую переменную
X = np.hstack(
    [
        df[
            [
                "Released_Year",
                "Runtime",
                "Meta_score",
                "No_of_Votes",
                "Gross",
                "Director_Score",
            ]
        ].values,
        genre_encoded,
        stars_encoded,
    ]  # type: ignore
)  # type: ignore
y = df["IMDB_Rating"].values

# 5) Масштабируем числовые признаки
scaler = StandardScaler()
X[:, :6] = scaler.fit_transform(X[:, :6])

# 6) Обучаем модель
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)  # type: ignore

# 7) Готовим NearestNeighbors для рекомендаций
neighbors = NearestNeighbors(n_neighbors=6, metric="euclidean")
neighbors.fit(X)


# 8) Функция предсказания рейтинга
def predict_rating(
    director: str,
    genres: list[str],
    stars: list[str],
    released_year: int | None = None,
    runtime: int | None = None,
    meta_score: int | None = None,
    votes: int | None = None,
    gross: int | None = None,
):
    # медианы для подстановки
    med_year = int(df["Released_Year"].median())
    med_runtime = int(df["Runtime"].median())
    med_meta = float(df["Meta_score"].median())
    med_votes = int(df["No_of_Votes"].median())
    med_gross = float(df["Gross"].median())

    if not director or not genres or not stars:
        raise ValueError(
            "Необходимо указать режиссёра, как минимум один жанр и одного актёра"
        )

    released_year = released_year or med_year
    runtime = runtime or med_runtime
    meta_score = meta_score or med_meta  # type: ignore
    votes = votes or med_votes
    gross = gross or med_gross  # type: ignore

    director_score = director_means.get(director, global_mean)

    num = np.array([[released_year, runtime, meta_score, votes, gross, director_score]])
    num = scaler.transform(num)

    genre_vec = mlb_genre.transform([genres])
    stars_vec = mlb_stars.transform([stars])

    features = np.hstack([num, genre_vec, stars_vec])  # type: ignore
    return float(model.predict(features)[0])


# 9) Функция рекомендаций
def recommend(liked_titles: list[str], n_recs: int = 5) -> list[str]:
    liked_set = set(liked_titles)
    idxs = df.index[df["Series_Title"].isin(liked_titles)].tolist()
    if not idxs:
        return []

    neigh_idxs = neighbors.kneighbors(X[idxs], return_distance=False)
    recs = []
    for nbrs in neigh_idxs:
        for i in nbrs[1:]:
            title = df.loc[i, "Series_Title"]
            if title in liked_set:
                continue
            recs.append(title)
            if len(recs) >= n_recs:
                return recs
    return recs


# 10) Экспортируем
__all__ = ["df", "mlb_genre", "mlb_stars", "predict_rating", "recommend"]
