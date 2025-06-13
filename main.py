import streamlit as st
from pipeline import df, mlb_genre, mlb_stars, predict_rating, recommend

st.title("Прогнозирование пользовательских предпочтений и предиктивная аналитика")

tab1, tab2 = st.tabs(["Получить кино рекомендации", "Спрогнозировать рейтинг фильма"])

# Вкладка рекомендаций
with tab1:
    st.header("Получить кино рекомендации")
    liked = st.multiselect(
        "Выберите фильмы, которые вам нравятся:",
        df["Series_Title"].tolist(),
        placeholder="Выберите вариант",
    )
    n_recs = st.number_input("Число рекомендаций:", min_value=1, max_value=20, value=5)
    if st.button("Порекомендовать"):
        if not liked:
            st.warning("Выберите хотя бы один фильм!")
        else:
            recs = recommend(liked, n_recs)
            if recs:
                st.write("Модель рекомендует следующие фильмы:")
                for r in recs:
                    st.write(f"- {r}")
            else:
                st.info("Не удалось найти рекомендации для выбранных фильмов.")


# Вкладка предсказания рейтинга
with tab2:
    st.header("Спрогнозировать рейтинг фильма")
    director = st.selectbox(
        "Режисёр", sorted(df["Director"].unique()), placeholder="Выберите вариант"
    )
    genres = st.multiselect("Жанры", mlb_genre.classes_, placeholder="Выберите вариант")
    stars = st.multiselect(
        "Главные роли", mlb_stars.classes_, placeholder="Выберите вариант"
    )

    # Опциональные числовые параметры
    col1, col2 = st.columns(2)
    with col1:
        use_year = st.checkbox("Указать год премьеры")
        if use_year:
            year = st.number_input(
                "Год премьеры",
                min_value=1900,
                max_value=2100,
                value=int(df["Released_Year"].median()),
                key="year",
            )
        else:
            year = None

        use_runtime = st.checkbox("Указать длительность (мин)")
        if use_runtime:
            runtime = st.number_input(
                "Длительность (мин)",
                min_value=0,
                max_value=500,
                value=int(df["Runtime"].median()),
                key="runtime",
            )
        else:
            runtime = None

    with col2:
        use_meta = st.checkbox("Указать оценку на Metacritic")
        if use_meta:
            meta = st.number_input(
                "Оценка Metacritic",
                min_value=0,
                max_value=100,
                value=int(df["Meta_score"].median()),
                key="meta",
            )
        else:
            meta = None

        use_votes = st.checkbox("Указать число оценок")
        if use_votes:
            votes = st.number_input(
                "Число оценок",
                min_value=0,
                value=int(df["No_of_Votes"].median()),
                key="votes",
            )
        else:
            votes = None

        use_gross = st.checkbox("Указать сборы в США")
        if use_gross:
            gross = st.number_input(
                "Сборы в США в $",
                min_value=0,
                value=int(df["Gross"].median()),
                key="gross",
                format="%d",
            )
        else:
            gross = None

    if st.button("Спрогнозировать рейтинг"):
        try:
            rating = predict_rating(
                director, genres, stars, year, runtime, meta, votes, gross
            )
            st.success(f"Спрогнозированный рейтинг IMDB: {rating:.2f}")
        except Exception as e:
            st.error(str(e))
