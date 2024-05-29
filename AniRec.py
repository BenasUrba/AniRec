# Instructions:
# Run the code, then click the local host link provided. This should bring you to the AniRec homepage, you can input a
# username to get personalized recommendations based on your watch order,
# or find some on the MyAnimeList website. Also you can test out the anime ID specific
# recommendations by inputting the anime ID, for example, 5114 = 'Fullmetal Alchemist Brotherhood'.

from flask import Flask, render_template, request
from surprise import Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import requests
import pandas as pd
import numpy as np

app = Flask(__name__)

svd_df = pd.read_csv('Datasets/100+Users5+Anime2.csv')

image_df = pd.read_csv('Datasets/anime_images.csv')

anime_data = pd.read_csv('Datasets/anime_data2.csv')

svd_model = joblib.load('Models/SVDModel1.pkl')

popular_animes = [5114, 1, 52991, 9253, 38524, 11061, 41467, 4181, 28851, 35180, 51009, 49387]

@app.route('/')
def homepage():
    global anime_data, popular_animes
    return render_template('homepage.html', popular_animes=popular_animes, get_anime_image=get_anime_image, anime_data=anime_data)


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    global svd_df, anime_data, svd_model

    # gets the values from username and anime_id fields
    username = request.form.get('username')
    anime_id = request.form.get('anime_id')
    username_error = False
    anime_id_error = False

    # Checks if both fields have input, if so then return an error
    if username and anime_id:
        username_error = True
        anime_id_error = True

        error_message = "Please input either your MyAnimeList username or an anime ID, not both"
        return render_template('homepage.html', popular_animes=popular_animes, get_anime_image=get_anime_image,
                               anime_data=anime_data, error_message=error_message, username_error=username_error, anime_id_error=anime_id_error)

    elif username:
        # Access code used for accessing and using MyAnimeList API
        # Please generate an access token using the TokenGenerator.py and access it through token.json
        access_token = "Access Token"
        anime_list = get_user_anime_list(username, access_token)

        if anime_list:
            ratings, seen_anime = get_user_ratings(anime_list)
            if ratings:
                user_df = pd.DataFrame(ratings, columns=['user_id', 'anime_id', 'rating'])
                svd_df = pd.concat([svd_df, user_df], ignore_index=True)
        else:
            # If username has anime list set on private or the username provided is not a valid username, then return
            # an error
            username_error = True
            error_message = f"Could not retrieve information for { username }. Make sure you entered the correct username, and your MyAnimeList anime list is set to public"
            return render_template('homepage.html', popular_animes=popular_animes, get_anime_image=get_anime_image,
                                   anime_data=anime_data, error_message=error_message, username_error=username_error,
                                   anime_id_error=anime_id_error)

    elif anime_id:
        if not anime_id.isdigit() or int(anime_id) not in anime_data['MAL_ID'].values:
            # if the anime ID is not integers or the anime ID does not exist in the dataset, then return an error
            anime_id_error = True
            error_message = f"Invalid anime ID: {anime_id}. Please enter a valid anime ID (e.g., 5114)"
            return render_template('homepage.html', popular_animes=popular_animes, get_anime_image=get_anime_image,
                                   anime_data=anime_data, error_message=error_message, username_error=username_error,
                                   anime_id_error=anime_id_error)
        else:
            anime_id = int(anime_id)
            seen_anime = [anime_id]

            # If recommendations for a specific anime ID is used, then use content based model to generate recommendations
            content_recs, content_ratings = content_recommender(seen_anime)

            recommendations = content_recs[:51]
            return render_template('recommendations.html', recommendations=recommendations,
                                   get_anime_image=get_anime_image, anime_data=anime_data, username=username)
    else:
        # Any other cases, for example if submit button is clicked with no values entered at all, it returns an error
        username_error = True
        anime_id_error = True
        error_message = "Please enter a MyAnimeList username or an anime ID"
        return render_template('homepage.html', popular_animes=popular_animes, get_anime_image=get_anime_image,
                               anime_data=anime_data, error_message=error_message, username_error=username_error,
                               anime_id_error=anime_id_error)

    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 10))
    data = Dataset.load_from_df(svd_df[['user_id', 'anime_id', 'rating']], reader=reader)

    # SVD model is fit on the new data, which includes the new users data
    svd_model.fit(data.build_full_trainset())

    # Generate the recommendations
    svd_recs = svd_recommender(seen_anime, svd_model)
    content_recs, content_ratings = content_recommender(seen_anime)
    recommendations = hybrid_recommender(svd_recs, content_recs, content_ratings)

    return render_template('recommendations.html', recommendations=recommendations, get_anime_image=get_anime_image, anime_data=anime_data, username=username)


# Function used to retrieve a users anime list using the MyAnimeList API
def get_anime_image(id):
    global image_df
    row = image_df[image_df['anime_id'] == id]
    if not row.empty:
        return row['jpg_large'].iloc[0]
    else:
        return None


def get_user_anime_list(username, access_token):
    url = f'https://api.myanimelist.net/v2/users/{username}/animelist?fields=list_status{{score}}'
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'nsfw': 'true', 'limit': 1000, 'sort': 'list_score'}
    response = requests.get(url, headers=headers,params=params)

    if response.status_code == 200:
        anime_list = response.json()
        return anime_list
    else:
        print(f"Failed to retrieve anime list for {username}")
        return None

# Function which takes in a users anime list and retrieves the ratings that a user has given for each anime
def get_user_ratings(anime_list):
    user_id = 10000
    if anime_list is None:
        return None
    ratings = []
    seen_animes = []
    for item in anime_list['data']:
        id = item['node']['id']
        try:
            status = item['list_status']['status']
            if status not in ['completed', 'watching']:
                continue
            rating = item['list_status']['score'] if 'list_status' in item and 'score' in item['list_status'] else None
            if rating is not None:
                ratings.append((user_id, id, rating))
            seen_animes.append(id)
        except KeyError:
            continue

    return ratings, seen_animes

# SVD recommender which takes in a list of seen anime IDs and returns the recommendations sorted by the highest predicted rating
def svd_recommender(seen_anime, svd_model):
    user_id = 10000

    anime_ids = svd_df['anime_id'].unique()

    unrated_animes = set(anime_ids) - set(seen_anime)

    predictions = [svd_model.predict(user_id, anime_id) for anime_id in unrated_animes]

    predictions.sort(key=lambda x: x.est, reverse=True)

    return predictions


content_df = pd.read_csv('Datasets/updated_anime_data21.csv')

numerical_feats = ['Score', 'Episodes']
categorical_feats = ['Type', 'Studios', 'Source', 'Creator']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_feats),
    ('cat', OneHotEncoder(), categorical_feats),
    ('tfidf', TfidfVectorizer(stop_words='english'), 'Genres')
])

preprocessed_X = preprocessor.fit_transform(content_df)

# Cosine similarity is calculated
cosine_similarity = cosine_similarity(preprocessed_X, preprocessed_X)

# Content recommender which takes in a list of anime IDs that the user has seen and returns the recommendations
# with the highest similarity
def content_recommender(id_list, df=content_df, cosine_similarity=cosine_similarity):
    total_sim_score = np.zeros(len(df))
    df_ids = df['MAL_ID']
    for mal_id in id_list:
        if mal_id in df_ids.values:
            idx = df[df['MAL_ID'] == mal_id].index[0]
            total_sim_score += cosine_similarity[idx]

    total_sim_score[np.isin(df_ids, id_list)] = 0

    min_score = np.min(total_sim_score)
    max_score = np.max(total_sim_score)
    normalized_scores = (total_sim_score - min_score) / (max_score - min_score)

    top_indices = normalized_scores.argsort()[::-1]

    estimated_ratings = normalized_scores[top_indices]

    return df.iloc[top_indices]['MAL_ID'], estimated_ratings


# Hybrid recommender which takes the recommendations from the SVD and Content-based models and the content ratings
# (cosine similarity scores), then multiplies the content ratings by 10 to get a 0-10 range, these scores are then added
# to the SVD predicted scores and divided by 2 to get the overall score, the highest recommendations are then recommended.
def hybrid_recommender(svd_recs, content_rec, content_ratings):

    svd_preds = {pred.iid: pred.est for pred in svd_recs}

    normalized_content_scores = [rating * 10 for rating in content_ratings]

    content_ratings_dic = dict(zip(content_rec, normalized_content_scores))

    svd_weights = 0.5

    hybrid_scores = {iid: (svd_weights * svd_preds.get(iid, 0)) + ((1 - svd_weights) * content_ratings_dic.get(iid, 0)) for iid in set(svd_preds) | set(content_ratings_dic)}

    sorted_scores = sorted(hybrid_scores.items(), key=lambda item : item[1], reverse=True)

    top_recommendations = [iid for iid, _ in sorted_scores[:51]]

    return top_recommendations


if __name__ == '__main__':
    app.run(debug=True)
