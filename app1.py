import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import base64
from datetime import datetime
from googletrans import Translator  # Pour la traduction

# Initialisation du traducteur
translator = Translator()

# Fonction pour récupérer la description dans la langue choisie
def get_synopsis_in_language(description, lang='fr'):
    try:
        if lang == "fr":
            # Traduction de l'anglais vers le français si la langue choisie est le français
            translated = translator.translate(description, src='en', dest='fr')  # Traduction vers le français
            return translated.text
        else:
            return description  # Le synopsis est déjà en anglais par défaut
    except Exception as e:
        return description  # Retourner le texte original en cas d'erreur

# Configurer la mise en page
st.set_page_config(layout="wide")

# Fonction pour convertir une image locale en base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()
    return base64_image

# Ajouter un fond d'écran via CSS avec une image locale
def set_background(image_path):
    base64_image = get_base64_of_image(image_path)
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/webp;base64,{base64_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .block-container {{
            padding: 2rem;
            background: rgba(0, 0, 0, 0.75);
            border-radius: 10px;
        }}
        h1, h2, h3, h4, h5, h6, p {{
            color: white !important;
        }}
        .stSelectbox > div > div {{
            max-width: 400px;  /* Réduire la largeur de la barre de recherche */
            margin: 0 auto;  /* Centrer la barre de recherche */
        }}
        </style>
        """, unsafe_allow_html=True)

# Chemin de votre image pour le fond d'écran
image_path = r"C:\Users\flori\Desktop\FORMATION DATA ANALYST\PROJET_2\STREAMLIT\SUGGESTION_FILMS\images\cinema.png"
set_background(image_path)

# Charger les données
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(columns={'Titre': 'Title', 'Genres': 'Genres', 'Année': 'Year',
                       'Moyenne': 'Vote_average', 'Synopsis': 'Description', 'Durée': 'Duration'}, inplace=True)
    df['All_Actors'] = df[['Acteur_1', 'Acteur_2', 'Acteur_3', 'Acteur_4', 'Acteur_5']].fillna('').agg(', '.join, axis=1)
    df['Genres'] = df['Genres'].fillna('')
    df['All_Actors'] = df['All_Actors'].str.strip()
    df['Description'] = df['Description'].fillna('Description indisponible.')
    df['Title'] = df['Title'].str.strip().str.lower()
    df['URL_AFFICHE'] = df['URL_AFFICHE'].fillna('placeholder.jpg')  # Gérer les affiches manquantes
    return df

csv_path = r":\Users\flori\Desktop\FORMATION DATA ANALYST\PROJET_2\BDD\mon_dataframe.csv"
df = load_data(csv_path)

# Nettoyer la liste des acteurs (enlever les crochets et guillemets)
def clean_actor_list(actor_list):
    return ', '.join([actor.strip() for actor in actor_list.strip("[]").replace("'", "").split(",")])

# Afficher l'affiche ou un espace vide si l'affiche est manquante
def display_movie_with_synopsis(row, synopsis):
    st.markdown(f"""
        <div class="movie-container">
            <img src="{row['URL_AFFICHE']}" class="movie-poster" alt="{row['Title']}">
            <div class="movie-synopsis">
                <p>{synopsis}</p>
            </div>
            <div style="color: white; padding: 10px 0 0 0;">
                <strong>Année de sortie :</strong> {int(row['Year'])}<br>
                <strong>⭐ Note :</strong> {row['Vote_average']}<br>
                <strong> {row['Genres']}<br>  <!-- Affichage du genre -->
                <strong>Acteurs :</strong> {clean_actor_list(row['All_Actors'])}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Fonction pour les recommandations basées sur KNN
def knn_recommendations(movie_title, df, k=5):
    df['Titre_normalized'] = df['Title'].str.strip().str.lower()
    movie_title_normalized = movie_title.strip().lower()

    if movie_title_normalized not in df['Titre_normalized'].values:
        st.error(f"Le film '{movie_title}' n'a pas été trouvé.")
        return pd.DataFrame()

    features = ['Vote_average', 'Duration', 'Genres', 'All_Actors']
    X = df[features].dropna()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Vote_average', 'Duration']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Genres', 'All_Actors'])
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(n_neighbors=k, metric='euclidean'))
    ])

    X_transformed = pipeline['preprocessor'].fit_transform(X)
    pipeline['knn'].fit(X_transformed)

    movie_index = df[df['Titre_normalized'] == movie_title_normalized].index[0]
    movie_features = X.iloc[movie_index:movie_index+1]
    movie_transformed = pipeline['preprocessor'].transform(movie_features)

    distances, indices = pipeline['knn'].kneighbors(movie_transformed)

    similar_movies = df.iloc[indices[0]]
    return similar_movies

# Fonction pour afficher les drapeaux et changer la langue dans le menu de gauche
def render_language_selector():
    with st.sidebar:
        st.write("### Choisir la langue")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('🇫🇷', key="fr"):
                st.session_state.language = "fr"
        with col2:
            if st.button('🇬🇧', key="en"):
                st.session_state.language = "en"

# Initialiser la langue par défaut si elle n'est pas déjà définie
if "language" not in st.session_state:
    st.session_state.language = "fr"  # Valeur par défaut

# Initialiser le menu par défaut si elle n'est pas déjà définie
if "menu" not in st.session_state:
    st.session_state.menu = "Page d'accueil"  # Valeur par défaut

# Barre de menu en haut
st.markdown("""
    <style>
        .menu-bar {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        .menu-bar button {
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            border: none;
            background-color: #4CAF50;
            color: black !important;  /* Forcer la couleur du texte en noir */
            cursor: pointer;
        }
        .menu-bar button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Créer une ligne de boutons centrés pour le menu
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Page d'accueil", key="home"):
        st.session_state.menu = "Page d'accueil"
with col2:
    if st.button("Recherche par films", key="search_movies"):
        st.session_state.menu = "Recherche par films"
with col3:
    if st.button("Films par acteur", key="movies_by_actor"):
        st.session_state.menu = "Films par acteur"
with col4:
    if st.button("Recherche par genre", key="search_by_genre"):
        st.session_state.menu = "Recherche par genre"

# Ajouter le filtre d'année dans le menu de gauche
with st.sidebar:
    st.write("### Filtre par année")
    start_year, end_year = st.slider("Sélectionner la période", 1900, 2024, (1900, 2024), step=1)

# CSS pour l'affichage en survol et rapprocher les titres
st.markdown("""
    <style>
        .movie-container {
            display: inline-block;
            position: relative;
            margin: 5px;
            width: 200px;  /* Réduire la largeur pour rapprocher les films */
            height: 300px;  /* Réduire la hauteur pour rapprocher les films */
        }
        .movie-poster {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .movie-synopsis {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0, 0, 0, 0.75);
            color: white;
            padding: 15px;
            opacity: 0;
            visibility: hidden;
            border-radius: 10px;
            overflow-y: auto;
            max-height: 280px;  /* Ajuster la hauteur maximale */
            transition: opacity 0.3s, visibility 0.3s;
        }
        .movie-container:hover .movie-synopsis {
            opacity: 1;
            visibility: visible;
        }
    </style>
""", unsafe_allow_html=True)

# Page d'accueil
if st.session_state.menu == "Page d'accueil":
    st.markdown("<h1 style='font-size: 40px;'>🎥 Bienvenue au cinéma de la CREUSE</h1>", unsafe_allow_html=True)
    render_language_selector()  # Afficher le sélecteur de langue dans le menu de gauche

    # Barre latérale pour les filtres
    with st.sidebar:
        st.write("### Trier par")
        sort_option = st.radio("Options de tri", ["Aucun tri", "Année de sortie (croissant)", "Année de sortie (décroissant)", "Note (meilleure à moins bonne)"])

    genres_list = ['Comedy', 'Documentary', 'Action', 'Animation', 'Family']  # Liste des genres

    current_year = datetime.now().year
    recent_movies = df[(df['URL_AFFICHE'].notnull())]

    for genre in genres_list:
        st.markdown(f"### Films dans le genre : {genre}")
        genre_movies = recent_movies[recent_movies['Genres'].str.contains(genre, na=False)]

        # Appliquer le filtre d'année
        genre_movies = genre_movies[(genre_movies['Year'] >= start_year) & (genre_movies['Year'] <= end_year)]

        # Appliquer le tri uniquement si l'utilisateur choisit une option de tri
        if sort_option == "Année de sortie (croissant)":
            genre_movies = genre_movies.sort_values(by='Year', ascending=True)
        elif sort_option == "Année de sortie (décroissant)":
            genre_movies = genre_movies.sort_values(by='Year', ascending=False)
        elif sort_option == "Note (meilleure à moins bonne)":
            genre_movies['Rating'] = pd.to_numeric(genre_movies['Vote_average'], errors='coerce')
            genre_movies = genre_movies.sort_values(by='Rating', ascending=False)

        if not genre_movies.empty:
            # Afficher 5 films par ligne
            cols = st.columns(5)  # Créer 5 colonnes
            for i, (_, row) in enumerate(genre_movies.iterrows()):
                if i >= 5:  # Limiter à 5 films par ligne
                    break
                with cols[i % 5]:
                    translated_synopsis = get_synopsis_in_language(row['Description'], st.session_state.language)
                    display_movie_with_synopsis(row, translated_synopsis)

# Page Recherche par film
elif st.session_state.menu == "Recherche par films":
    st.markdown("<h1 style='font-size: 40px;'>🔍 Recherche par film</h1>", unsafe_allow_html=True)
    render_language_selector()  # Afficher le sélecteur de langue dans le menu de gauche

    # Barre latérale pour les filtres
    with st.sidebar:
        st.write("### Trier par")
        sort_option = st.radio(
            "Options de tri", ["Aucun tri", "Année de sortie (croissant)", "Année de sortie (décroissant)", "Note (meilleure à moins bonne)"], key="movie_sort_option"
        )

    # Liste des titres de films
    movie_list = df['Title'].str.strip().unique().tolist()  # Liste des titres uniques
    movie_list = [movie.title() for movie in movie_list]  # Formater les titres (optionnel)

    # Barre de recherche avec suggestions (centrée et réduite)
    col1, col2, col3 = st.columns([1, 2, 1])  # Créer 3 colonnes (la colonne du milieu est plus large)
    with col2:  # Utiliser la colonne du milieu pour centrer la barre de recherche
        movie_search = st.selectbox("Recherche un film :", ["Tout"] + movie_list, key="movie_search")

    # Si "Tout" est sélectionné, ne rien afficher
    if movie_search == "Tout":
        st.write("Veuillez sélectionner un film pour afficher les résultats.")
    else:
        # Recherche par titre exact
        movie_results = df[df['Title'].str.strip().str.lower() == movie_search.strip().lower()]

        # Appliquer le filtre d'année
        movie_results = movie_results[(movie_results['Year'] >= start_year) & (movie_results['Year'] <= end_year)]

        # Appliquer le tri uniquement si l'utilisateur choisit une option de tri
        if sort_option == "Année de sortie (croissant)":
            movie_results = movie_results.sort_values(by='Year', ascending=True)
        elif sort_option == "Année de sortie (décroissant)":
            movie_results = movie_results.sort_values(by='Year', ascending=False)
        elif sort_option == "Note (meilleure à moins bonne)":
            movie_results['Rating'] = pd.to_numeric(movie_results['Vote_average'], errors='coerce')
            movie_results = movie_results.sort_values(by='Rating', ascending=False)

        # Afficher les résultats de la recherche
        if not movie_results.empty:
            st.markdown(f"### Résultats pour '{movie_search}'")
            cols = st.columns(5)  # 5 films par ligne
            for i, (_, row) in enumerate(movie_results.iterrows()):
                with cols[i % 5]:
                    translated_synopsis = get_synopsis_in_language(row['Description'], st.session_state.language)
                    display_movie_with_synopsis(row, translated_synopsis)
        else:
            st.write("Aucun résultat trouvé pour votre recherche.")

        # Recommandation KNN (uniquement si un film spécifique est sélectionné)
        st.markdown(f"### Suggestions basées sur '{movie_search}'")
        knn_results = knn_recommendations(movie_search, df, k=10)
        knn_results = knn_results[(knn_results['Year'] >= start_year) & (knn_results['Year'] <= end_year)]

        if not knn_results.empty:
            cols = st.columns(5)  # 5 films par ligne
            for i, (_, row) in enumerate(knn_results.iterrows()):
                with cols[i % 5]:
                    translated_synopsis = get_synopsis_in_language(row['Description'], st.session_state.language)
                    display_movie_with_synopsis(row, translated_synopsis)
        else:
            st.write("Aucune suggestion disponible pour ce film.")

# Page Films par acteur
elif st.session_state.menu == "Films par acteur":
    st.markdown("<h1 style='font-size: 40px;'>🎭 Films par acteur</h1>", unsafe_allow_html=True)
    render_language_selector()  # Afficher le sélecteur de langue dans le menu de gauche

    # Barre latérale pour les filtres
    with st.sidebar:
        st.write("### Trier par")
        sort_option = st.radio(
            "Options de tri", ["Aucun tri", "Année de sortie (croissant)", "Année de sortie (décroissant)", "Note (meilleure à moins bonne)"], key="actor_sort_option"
        )

    # Liste des acteurs
    actor_list = df['All_Actors'].str.split(', ').explode().dropna().unique().tolist()
    actor_list = [clean_actor_list(actor) for actor in actor_list]  # Nettoyer les acteurs
    actor_filter = st.selectbox("Choisissez un acteur :", ["Tous"] + actor_list, key="actor_selectbox")

    if actor_filter != "Tous":
        actor_movies = df[df['All_Actors'].str.contains(actor_filter, na=False, regex=False)]
        # Appliquer le filtre d'année
        actor_movies = actor_movies[(actor_movies['Year'] >= start_year) & (actor_movies['Year'] <= end_year)]

        # Appliquer le tri uniquement si l'utilisateur choisit une option de tri
        if sort_option == "Année de sortie (croissant)":
            actor_movies = actor_movies.sort_values(by='Year', ascending=True)
        elif sort_option == "Année de sortie (décroissant)":
            actor_movies = actor_movies.sort_values(by='Year', ascending=False)
        elif sort_option == "Note (meilleure à moins bonne)":
            actor_movies['Rating'] = pd.to_numeric(actor_movies['Vote_average'], errors='coerce')
            actor_movies = actor_movies.sort_values(by='Rating', ascending=False)

        if not actor_movies.empty:
            st.markdown(f"### Films avec {actor_filter}")
            cols = st.columns(5)  # 5 films par ligne
            for i, (_, row) in enumerate(actor_movies.iterrows()):
                with cols[i % 5]:
                    translated_synopsis = get_synopsis_in_language(row['Description'], st.session_state.language)
                    display_movie_with_synopsis(row, translated_synopsis)
    else:
        st.write("Veuillez sélectionner un acteur pour afficher les films.")

# Page Recherche par genre
elif st.session_state.menu == "Recherche par genre":
    st.markdown("<h1 style='font-size: 40px;'>🎬 Films par genre</h1>", unsafe_allow_html=True)
    render_language_selector()  # Afficher le sélecteur de langue dans le menu de gauche

    # Barre latérale pour les filtres
    with st.sidebar:
        st.write("### Trier par")
        sort_option = st.radio(
            "Options de tri", ["Aucun tri", "Année de sortie (croissant)", "Année de sortie (décroissant)", "Note (meilleure à moins bonne)"], key="genre_sort_option"
        )

    genres_list = ['Tout', 'Comedy', 'Documentary', 'Action', 'Animation', 'Family']  # Liste des genres
    genre_filter = st.selectbox("Choisissez un genre :", genres_list, key="genre_selectbox")

    if genre_filter != "Tout":
        genre_movies = df[df['Genres'].str.contains(genre_filter, na=False)]
        # Appliquer le filtre d'année
        genre_movies = genre_movies[(genre_movies['Year'] >= start_year) & (genre_movies['Year'] <= end_year)]

        # Appliquer le tri uniquement si l'utilisateur choisit une option de tri
        if sort_option == "Année de sortie (croissant)":
            genre_movies = genre_movies.sort_values(by='Year', ascending=True)
        elif sort_option == "Année de sortie (décroissant)":
            genre_movies = genre_movies.sort_values(by='Year', ascending=False)
        elif sort_option == "Note (meilleure à moins bonne)":
            genre_movies['Rating'] = pd.to_numeric(genre_movies['Vote_average'], errors='coerce')
            genre_movies = genre_movies.sort_values(by='Rating', ascending=False)

        if not genre_movies.empty:
            st.markdown(f"### Films dans le genre : {genre_filter}")
            cols = st.columns(5)  # 5 films par ligne
            for i, (_, row) in enumerate(genre_movies.iterrows()):
                with cols[i % 5]:
                    translated_synopsis = get_synopsis_in_language(row['Description'], st.session_state.language)
                    display_movie_with_synopsis(row, translated_synopsis)
    else:
        st.write("Veuillez sélectionner un genre pour afficher les films.")